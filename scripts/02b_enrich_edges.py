"""
Dedicated edge enrichment pass — extracts more relationships between
philosophers already in our graph, with richer relation types.

Runs after 02_enrich.py. Reads data/enriched/philosophers.json,
produces data/enriched/edges.json, then merges the result back into
data/graph.json (requires 03_build_graph.py to have run first).

Relation types:
  student_of        formal teacher-student relationship
  built_on          extended / synthesized another's ideas
  critiqued         engaged critically with specific positions
  refuted           explicitly argued against core thesis
  collaborated_with co-authored works or sustained joint projects
  contemporary_of   same generation (~50yr window), likely mutual awareness

Only IDs that exist in our philosopher list are emitted.
Merges with existing edges — no duplicates.

Usage:
    python scripts/02b_enrich_edges.py
    python scripts/02b_enrich_edges.py --dry-run   # 8 philosophers only
"""

import json
import os
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

ENRICHED = Path(__file__).parent.parent / "data" / "enriched" / "philosophers.json"
GRAPH    = Path(__file__).parent.parent / "data" / "graph.json"
OUT      = Path(__file__).parent.parent / "data" / "enriched" / "edges.json"

MODEL      = "claude-sonnet-4-6"
BATCH_SIZE = 12

RELATION_TYPES = [
    "student_of",
    "built_on",
    "critiqued",
    "refuted",
    "collaborated_with",
    "contemporary_of",
]

SYSTEM = """You are extracting philosophical relationships for a knowledge graph.
Only use philosopher IDs from the provided index. Be generous — most major
philosophers have 5–12 meaningful connections within this list.

Relation guide:
  student_of        — formally studied under (Aristotle under Plato)
  built_on          — substantively extended or synthesized their ideas
  critiqued         — engaged critically, challenged specific positions
  refuted           — explicitly argued the core thesis is wrong
  collaborated_with — co-authored major works or long joint projects
  contemporary_of   — same generation (~50yr), demonstrably aware of each other

Edge direction: source influenced / engaged with target.
For collaborated_with and contemporary_of emit both directions.
Return ONLY valid JSON, no markdown."""


def build_index(philosophers: list[dict]) -> str:
    return "\n".join(f"  {p['id']}: {p['name']} ({p.get('born','?')}–{p.get('died','?')})"
                     for p in philosophers)


def enrich_batch(
    client: anthropic.Anthropic,
    batch: list[dict],
    index: str,
) -> list[dict]:
    entries = "\n".join(
        f"  - {p['id']} | {p['name']} | era: {p.get('era','')} | "
        f"schools: {', '.join(p.get('schools',[])[:3])}"
        for p in batch
    )

    prompt = f"""Full philosopher index (use only these IDs):
{index}

Extract ALL meaningful relationships for each philosopher below.
Aim for 5–12 edges per philosopher. Include cross-tradition connections
where real (e.g. Schopenhauer ↔ Buddhism, Leibniz ↔ Chinese thought).

Philosophers to process:
{entries}

Return a JSON array of edge objects:
[{{"source": "<id>", "target": "<id>", "relation": "<type>"}}, ...]

Include reverse edges for collaborated_with and contemporary_of."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=6000,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()

    return json.loads(raw)


def main():
    dry_run = "--dry-run" in sys.argv

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    philosophers = json.load(open(ENRICHED, encoding="utf-8"))
    valid_ids    = {p["id"] for p in philosophers}

    if dry_run:
        # Representative sample across traditions and eras
        sample = {"immanuel_kant", "aristotle", "georg_wilhelm_friedrich_hegel",
                  "laozi", "martin_heidegger", "john_rawls",
                  "arthur_schopenhauer", "simone_de_beauvoir"}
        to_process = [p for p in philosophers if p["id"] in sample]
        print(f"Dry run: {len(to_process)} philosophers")
    else:
        to_process = philosophers
        print(f"Enriching edges for {len(to_process)} philosophers...")

    index   = build_index(philosophers)
    client  = anthropic.Anthropic(api_key=api_key)
    batches = [to_process[i:i+BATCH_SIZE] for i in range(0, len(to_process), BATCH_SIZE)]

    all_edges: list[dict] = []

    for i, batch in enumerate(batches):
        names = ", ".join(p["name"] for p in batch)
        print(f"Batch {i+1}/{len(batches)}: {names[:70]}...")

        try:
            edges = enrich_batch(client, batch, index)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Error: {e}. Skipping batch.")
            continue

        # Validate IDs
        valid = [
            e for e in edges
            if e.get("source") in valid_ids
            and e.get("target") in valid_ids
            and e.get("source") != e.get("target")
            and e.get("relation") in RELATION_TYPES
        ]
        print(f"  {len(valid)} valid edges (of {len(edges)} returned)")
        all_edges.extend(valid)

        if i < len(batches) - 1:
            time.sleep(0.8)

    # Deduplicate
    seen: set[tuple] = set()
    deduped = []
    for e in all_edges:
        key = (e["source"], e["target"], e["relation"])
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    print(f"\nTotal unique edges: {len(deduped)}")

    # Breakdown by type
    from collections import Counter
    dist = Counter(e["relation"] for e in deduped)
    for rel, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {rel}: {count}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {OUT}")

    # Merge into graph.json if it exists
    if GRAPH.exists() and not dry_run:
        graph = json.load(open(GRAPH, encoding="utf-8"))

        existing_keys = {
            (e["source"], e["target"], e["relation"])
            for e in graph["edges"]
        }

        added = 0
        edge_id = max((int(e["id"][1:]) for e in graph["edges"] if e["id"][1:].isdigit()), default=0)
        for e in deduped:
            key = (e["source"], e["target"], e["relation"])
            if key not in existing_keys:
                edge_id += 1
                graph["edges"].append({
                    "id":       f"e{edge_id}",
                    "source":   e["source"],
                    "target":   e["target"],
                    "relation": e["relation"],
                })
                existing_keys.add(key)
                added += 1

        graph["meta"]["edge_count"] = len(graph["edges"])
        with open(GRAPH, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)
        print(f"Merged {added} new edges into graph.json ({len(graph['edges'])} total)")


if __name__ == "__main__":
    main()
