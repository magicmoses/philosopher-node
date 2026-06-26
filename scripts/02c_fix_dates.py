"""
Fill in missing born/died years for philosophers where the LLM returned null.

These are mostly ancient Eastern philosophers (Laozi, Vyasa, Sunzi, etc.)
where exact dates are genuinely debated — we use the scholarly best-estimate
or traditional date, noting the uncertainty in a comment.

Also patches obviously-wrong null died values for philosophers known to be
deceased (e.g. contemporary figures who died recently).

Usage:
    python scripts/02c_fix_dates.py
"""

import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

ENRICHED = Path(__file__).parent.parent / "data" / "enriched" / "philosophers.json"
MODEL    = "claude-sonnet-4-6"


def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    philosophers = json.load(open(ENRICHED, encoding="utf-8"))
    missing = [p for p in philosophers if p.get("born") is None]

    if not missing:
        print("No missing born years — nothing to do.")
        return

    print(f"Fixing {len(missing)} missing born years: {', '.join(p['name'] for p in missing)}")

    client = anthropic.Anthropic(api_key=api_key)

    entries = "\n".join(
        f"  {p['id']}: {p['name']} (era: {p.get('era','?')}, tradition: {p.get('tradition','?')})"
        for p in missing
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": f"""For each philosopher below, provide the best scholarly estimate for birth year.
Use negative numbers for BCE (e.g. Plato born 428 BCE → -428).
For figures with genuinely unknown dates, use the most commonly cited scholarly estimate.
For living contemporary philosophers, use null for died.

Philosophers:
{entries}

Return JSON array: [{{"id": "...", "born": <integer or null>, "died": <integer or null>}}, ...]
Return valid JSON only."""}]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()

    updates = json.loads(raw)
    update_map = {u["id"]: u for u in updates}

    patched = 0
    for p in philosophers:
        if p["id"] in update_map:
            u = update_map[p["id"]]
            if p.get("born") is None and u.get("born") is not None:
                p["born"] = u["born"]
                patched += 1
            if p.get("died") is None and u.get("died") is not None:
                p["died"] = u["died"]

    with open(ENRICHED, "w", encoding="utf-8") as f:
        json.dump(philosophers, f, indent=2, ensure_ascii=False)

    # Verify
    still_missing = [p for p in philosophers if p.get("born") is None]
    print(f"Patched {patched} born years. Still missing: {len(still_missing)}")
    if still_missing:
        for p in still_missing:
            print(f"  {p['name']} — genuinely unknown, will use era fallback")


if __name__ == "__main__":
    main()
