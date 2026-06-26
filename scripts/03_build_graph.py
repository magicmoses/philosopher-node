"""
Build the multi-type knowledge graph from enriched philosopher data.

Node types: Philosopher, Work, Concept, School, Era
Edge types: student_of, built_on, critiqued, refuted,
            authored, coined, member_of, belongs_to_era

Pipeline:
  1. Load enriched philosophers, remove isolated nodes (no influence edges)
  2. Extract Work / Concept / School / Era nodes with deduplication
  3. LLM pass: classify each influence edge as student_of/built_on/critiqued/refuted
  4. Run Louvain on the philosopher subgraph → community_id on each philosopher
  5. Assemble and save data/graph.json

Usage:
    python scripts/03_build_graph.py
"""

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import anthropic
import networkx as nx
from dotenv import load_dotenv

load_dotenv()

try:
    import community as community_louvain
except ImportError:
    print("Missing: pip install python-louvain")
    sys.exit(1)

ENRICHED = Path(__file__).parent.parent / "data" / "enriched" / "philosophers.json"
OUT = Path(__file__).parent.parent / "data" / "graph.json"
MODEL = "claude-sonnet-4-6"

RELATION_TYPES = ["student_of", "built_on", "critiqued", "refuted"]

ERAS = [
    {"id": "era_ancient",      "name": "Ancient",      "range": "~600 BCE – 500 CE"},
    {"id": "era_medieval",     "name": "Medieval",     "range": "500 – 1400"},
    {"id": "era_early_modern", "name": "Early Modern", "range": "1400 – 1800"},
    {"id": "era_modern",       "name": "Modern",       "range": "1800 – 1945"},
    {"id": "era_contemporary", "name": "Contemporary", "range": "1945 – present"},
    {"id": "era_various",      "name": "Various",      "range": "Multiple eras"},
]

ERA_ID = {e["name"]: e["id"] for e in ERAS}


# ── Helpers ──────────────────────────────────────────────────────────────────

def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def strip_fences(text: str) -> str:
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


# ── Step 1: Filter singletons ─────────────────────────────────────────────────

def filter_singletons(philosophers: list[dict]) -> tuple[list[dict], list[str]]:
    kept = [p for p in philosophers if p.get("influenced") or p.get("influenced_by")]
    removed = [p["name"] for p in philosophers if p not in kept]
    return kept, removed


# ── Step 2: Extract secondary nodes ──────────────────────────────────────────

def extract_secondary_nodes(philosophers: list[dict]) -> tuple[list, list, list]:
    """Return (work_nodes, concept_nodes, school_nodes) with deduplication."""
    works: dict[str, dict] = {}
    concepts: dict[str, dict] = {}
    schools: dict[str, dict] = {}

    for p in philosophers:
        for title in p.get("works", []):
            wid = f"work_{slug(title)}"
            if wid not in works:
                works[wid] = {"id": wid, "type": "Work", "name": title, "philosopher_ids": []}
            if p["id"] not in works[wid]["philosopher_ids"]:
                works[wid]["philosopher_ids"].append(p["id"])

        for concept in p.get("key_concepts", []):
            cid = f"concept_{slug(concept)}"
            if cid not in concepts:
                concepts[cid] = {"id": cid, "type": "Concept", "name": concept, "philosopher_ids": []}
            if p["id"] not in concepts[cid]["philosopher_ids"]:
                concepts[cid]["philosopher_ids"].append(p["id"])

        for school in p.get("schools", []):
            sid = f"school_{slug(school)}"
            if sid not in schools:
                schools[sid] = {"id": sid, "type": "School", "name": school, "philosopher_ids": []}
            if p["id"] not in schools[sid]["philosopher_ids"]:
                schools[sid]["philosopher_ids"].append(p["id"])

    return list(works.values()), list(concepts.values()), list(schools.values())


# ── Step 3: LLM relation classification ──────────────────────────────────────

CLASSIFY_SYSTEM = """You are classifying influence relationships in a philosophy knowledge graph.
For each (source → target) pair, pick the single most accurate relation type:

- student_of : source was formally taught by or directly studied under target
- built_on   : source extended, developed, or synthesized target's ideas (default when positive)
- critiqued  : source engaged critically with target, refining or challenging specific positions
- refuted    : source argued explicitly against target's core thesis

When genuinely unsure, use built_on. Return valid JSON only."""


def classify_batch(client: anthropic.Anthropic, pairs: list[dict]) -> dict[str, str]:
    """
    pairs: [{"edge_id": "kant->hume", "source": "Kant", "target": "Hume"}, ...]
    Returns {edge_id: relation_type}
    """
    lines = "\n".join(
        f'  {p["edge_id"]}: {p["source"]} → {p["target"]}'
        for p in pairs
    )
    prompt = f"""Classify each influence edge. Return a JSON array of objects with "edge_id" and "relation".

Edges:
{lines}

Valid relation values: student_of, built_on, critiqued, refuted"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=CLASSIFY_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    result = json.loads(strip_fences(response.content[0].text))
    return {item["edge_id"]: item["relation"] for item in result}


def classify_all_edges(
    client: anthropic.Anthropic,
    philosophers: list[dict],
    batch_size: int = 40,
) -> dict[str, str]:
    """Build all influence pairs and classify them in batches."""
    id_to_name = {p["id"]: p["name"] for p in philosophers}

    all_pairs = []
    for p in philosophers:
        for target_id in p.get("influenced", []):
            if target_id in id_to_name:
                edge_id = f"{p['id']}->{target_id}"
                all_pairs.append({
                    "edge_id": edge_id,
                    "source": p["name"],
                    "target": id_to_name[target_id],
                })

    print(f"Classifying {len(all_pairs)} influence edges...")
    classified: dict[str, str] = {}
    batches = [all_pairs[i : i + batch_size] for i in range(0, len(all_pairs), batch_size)]

    for i, batch in enumerate(batches):
        print(f"  Batch {i+1}/{len(batches)} ({len(batch)} edges)...")
        try:
            result = classify_batch(client, batch)
            classified.update(result)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Parse error: {e}. Defaulting batch to built_on.")
            for pair in batch:
                classified[pair["edge_id"]] = "built_on"

        if i < len(batches) - 1:
            time.sleep(0.5)

    return classified


# ── Step 4: Louvain communities ───────────────────────────────────────────────

def detect_communities(philosophers: list[dict]) -> dict[str, int]:
    G = nx.Graph()
    for p in philosophers:
        G.add_node(p["id"])
    for p in philosophers:
        for target in p.get("influenced", []):
            if G.has_node(target):
                G.add_edge(p["id"], target)

    # Louvain requires undirected — we already built an undirected graph
    partition = community_louvain.best_partition(G)
    return partition


# ── Step 5: Assemble graph.json ───────────────────────────────────────────────

def assemble(
    philosophers: list[dict],
    works: list[dict],
    concepts: list[dict],
    schools: list[dict],
    edge_types: dict[str, str],
    communities: dict[str, int],
    removed_singletons: list[str],
) -> dict:
    phil_ids = {p["id"] for p in philosophers}
    nodes = []
    edges = []
    edge_counter = 0

    def add_edge(source: str, target: str, relation: str):
        nonlocal edge_counter
        edges.append({"id": f"e{edge_counter}", "source": source, "target": target, "relation": relation})
        edge_counter += 1

    # Philosopher nodes
    for p in philosophers:
        nodes.append({
            "id": p["id"],
            "type": "Philosopher",
            "name": p["name"],
            "abstract": p.get("abstract", ""),
            "born": p.get("born"),
            "died": p.get("died"),
            "era": p.get("era", "Unknown"),
            "tradition": p.get("tradition", "Western"),
            "domains": p.get("domains", []),
            "community_id": communities.get(p["id"], -1),
        })
        # belongs_to_era
        era_id = ERA_ID.get(p.get("era", ""), "era_various")
        add_edge(p["id"], era_id, "belongs_to_era")

    # Era nodes
    for era in ERAS:
        nodes.append({**era, "type": "Era"})

    # Work nodes + authored edges
    for w in works:
        nodes.append({"id": w["id"], "type": "Work", "name": w["name"]})
        for pid in w["philosopher_ids"]:
            if pid in phil_ids:
                add_edge(pid, w["id"], "authored")

    # Concept nodes + coined edges
    for c in concepts:
        nodes.append({"id": c["id"], "type": "Concept", "name": c["name"]})
        for pid in c["philosopher_ids"]:
            if pid in phil_ids:
                add_edge(pid, c["id"], "coined")

    # School nodes + member_of edges
    for s in schools:
        nodes.append({"id": s["id"], "type": "School", "name": s["name"]})
        for pid in s["philosopher_ids"]:
            if pid in phil_ids:
                add_edge(pid, s["id"], "member_of")

    # Typed influence edges
    for p in philosophers:
        for target_id in p.get("influenced", []):
            if target_id in phil_ids:
                edge_id = f"{p['id']}->{target_id}"
                relation = edge_types.get(edge_id, "built_on")
                add_edge(p["id"], target_id, relation)

    return {
        "meta": {
            "version": 2,
            "philosopher_count": len(philosophers),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "singletons_removed": removed_singletons,
        },
        "nodes": nodes,
        "edges": edges,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    all_philosophers = json.load(open(ENRICHED, encoding="utf-8"))
    print(f"Loaded {len(all_philosophers)} philosophers.")

    philosophers, removed = filter_singletons(all_philosophers)
    print(f"Removed {len(removed)} isolated nodes: {', '.join(removed[:5])}{'...' if len(removed) > 5 else ''}")
    print(f"Working with {len(philosophers)} connected philosophers.")

    works, concepts, schools = extract_secondary_nodes(philosophers)
    print(f"Extracted: {len(works)} works, {len(concepts)} concepts, {len(schools)} schools")

    client = anthropic.Anthropic(api_key=api_key)
    edge_types = classify_all_edges(client, philosophers)
    print(f"Classified {len(edge_types)} edges.")

    # Print relation type distribution
    from collections import Counter
    dist = Counter(edge_types.values())
    for rel, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {rel}: {count}")

    print("Running community detection...")
    communities = detect_communities(philosophers)
    n_communities = len(set(communities.values()))
    print(f"Found {n_communities} communities.")

    graph = assemble(philosophers, works, concepts, schools, edge_types, communities, removed)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    print(f"\nSaved -> {OUT}")
    print(f"  {graph['meta']['node_count']} nodes, {graph['meta']['edge_count']} edges")


if __name__ == "__main__":
    main()
