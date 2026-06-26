"""
Enrich parsed philosophers with LLM-generated abstracts, dates, key concepts,
and influence relationships via the Anthropic API.

Processes in batches of 15. Saves progress after each batch so the script
can be safely interrupted and resumed.

Usage:
    python scripts/02_enrich.py            # enrich all
    python scripts/02_enrich.py --dry-run  # test with 5 philosophers
"""

import json
import os
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

RAW = Path(__file__).parent.parent / "data" / "raw" / "philosophers.json"
OUT = Path(__file__).parent.parent / "data" / "enriched" / "philosophers.json"
BATCH_SIZE = 15
MODEL = "claude-sonnet-4-6"

CANONICAL_ERAS = ["Ancient", "Medieval", "Early Modern", "Modern", "Contemporary", "Various"]


def build_system_prompt(all_philosophers: list[dict]) -> str:
    """System prompt includes the full name→id index for accurate influence linking."""
    index = "\n".join(f"  {p['id']}: {p['name']}" for p in all_philosophers)
    return f"""You are enriching a philosophy knowledge graph. Your output is machine-parsed JSON — be precise and consistent.

The complete list of philosophers in our graph (id: name):
{index}

Rules:
- "influences" and "influenced_by" must only contain IDs from the list above. No external references.
- "era" must be one of: {", ".join(CANONICAL_ERAS)}
- "abstract" is 2-3 sentences: who they were, their core contribution, why they matter historically.
- "key_concepts" are 3-5 specific terms this philosopher introduced or is most associated with.
- "schools" are movement names (e.g. "German Idealism", "Stoicism", "Phenomenology").
- If a philosopher is unknown or obscure, do your best with available knowledge. Don't hallucinate dates.
- Return valid JSON only. No markdown, no explanation outside the JSON."""


def build_user_prompt(batch: list[dict]) -> str:
    entries = []
    for p in batch:
        entries.append(
            f'  - id: "{p["id"]}", name: "{p["name"]}", '
            f'tradition: "{p["tradition"]}", '
            f'domains: {json.dumps(p["domains"][:3])}, '
            f'works: {json.dumps(p["works"][:4])}'
        )

    return f"""Enrich the following {len(batch)} philosophers. Return a JSON array with one object per philosopher.

Philosophers to enrich:
{chr(10).join(entries)}

Required fields per object:
{{
  "id": "<same id as input>",
  "abstract": "<2-3 sentences>",
  "born": <year as integer or null>,
  "died": <year as integer or null>,
  "era": "<one of the canonical eras>",
  "key_concepts": ["<concept>", ...],
  "schools": ["<school/movement>", ...],
  "influenced": ["<id of philosopher THIS person influenced — their students/successors>", ...],
  "influenced_by": ["<id of philosopher who influenced THIS person — their teachers/predecessors>", ...]
}}"""


def enrich_batch(client: anthropic.Anthropic, batch: list[dict], system: str) -> list[dict]:
    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        system=system,
        messages=[{"role": "user", "content": build_user_prompt(batch)}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if the model wrapped the output
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


def merge(raw: dict, enriched: dict, valid_ids: set[str]) -> dict:
    """Merge enrichment data into the raw philosopher entry."""
    uid = raw["id"]

    def clean_ids(ids: list) -> list:
        # Remove self-references and IDs not in our graph
        return [i for i in ids if i != uid and i in valid_ids]

    return {
        **raw,
        "abstract": enriched.get("abstract", ""),
        "born": enriched.get("born"),
        "died": enriched.get("died"),
        "era": enriched.get("era", raw["eras"][0] if raw["eras"] else "Unknown"),
        "key_concepts": enriched.get("key_concepts", []),
        "schools": enriched.get("schools", []),
        "influenced": clean_ids(enriched.get("influenced", [])),
        "influenced_by": clean_ids(enriched.get("influenced_by", [])),
    }


def load_existing() -> dict[str, dict]:
    """Load partially-enriched output for resume support."""
    if OUT.exists():
        data = json.load(open(OUT, encoding="utf-8"))
        return {p["id"]: p for p in data}
    return {}


def save(philosophers: list[dict]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(philosophers, f, indent=2, ensure_ascii=False)


def main():
    dry_run = "--dry-run" in sys.argv

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    all_philosophers = json.load(open(RAW, encoding="utf-8"))
    already_done = load_existing()

    if dry_run:
        # Test with 5 representative philosophers across traditions
        test_names = {"immanuel_kant", "aristotle", "georg_wilhelm_friedrich_hegel", "laozi", "martin_heidegger"}
        to_process = [p for p in all_philosophers if p["id"] in test_names]
        print(f"Dry run: enriching {len(to_process)} philosophers")
    else:
        to_process = [p for p in all_philosophers if p["id"] not in already_done]
        print(f"Enriching {len(to_process)} philosophers ({len(already_done)} already done)")

    if not to_process:
        print("Nothing to do.")
        return

    system = build_system_prompt(all_philosophers)
    results: dict[str, dict] = dict(already_done)

    batches = [to_process[i : i + BATCH_SIZE] for i in range(0, len(to_process), BATCH_SIZE)]

    for i, batch in enumerate(batches):
        names = ", ".join(p["name"] for p in batch)
        print(f"Batch {i+1}/{len(batches)}: {names[:80]}...")

        try:
            enriched_batch = enrich_batch(client, batch, system)
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}. Skipping batch.")
            continue
        except anthropic.APIError as e:
            print(f"  API error: {e}. Waiting 10s before retry...")
            time.sleep(10)
            enriched_batch = enrich_batch(client, batch, system)

        # Index enriched results by id
        enriched_by_id = {e["id"]: e for e in enriched_batch}
        valid_ids = {p["id"] for p in all_philosophers}

        for p in batch:
            enriched = enriched_by_id.get(p["id"], {})
            results[p["id"]] = merge(p, enriched, valid_ids)

        # Save after every batch — safe to interrupt
        ordered = [results[p["id"]] for p in all_philosophers if p["id"] in results]
        save(ordered)
        print(f"  Saved {len(results)}/{len(all_philosophers)} total")

        # Respect rate limits between batches
        if i < len(batches) - 1:
            time.sleep(1)

    print(f"\nDone. {len(results)} philosophers enriched -> {OUT}")


if __name__ == "__main__":
    main()
