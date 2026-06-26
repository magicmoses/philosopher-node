"""
Re-run community detection on the full edge set in graph.json,
then use an LLM agent to autonomously validate, name, and merge communities.

Why a separate script: 03_build_graph.py only uses the original influenced/
influenced_by edges from philosophers.json. After 02b_enrich_edges.py added
~2100 more edges directly to graph.json, this script re-clusters on the
complete 2570-edge graph — which produces meaningfully different communities.

Louvain uses only intellectual lineage edges (built_on, student_of,
collaborated_with, critiqued) with weights. contemporary_of and refuted
are excluded — temporal proximity / opposition ≠ tradition membership.

The agent checkpoint:
  Receives all raw Louvain communities with member lists and degree stats.
  Autonomously decides: names, merges (targets 7–10 final communities),
  tradition labels, and short descriptions for the UI.
  Writes data/communities_final.json.
  Updates community_id on philosopher nodes in graph.json.

Usage:
    python scripts/03b_recluster.py
"""

import json
import os
import sys
import time
from collections import Counter, defaultdict
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

GRAPH  = Path(__file__).parent.parent / "data" / "graph.json"
COMMUNITIES_OUT = Path(__file__).parent.parent / "data" / "communities_final.json"
MODEL  = "claude-sonnet-4-6"

# Edge weights for community detection.
# Only intellectual lineage edges — temporal/opposition edges excluded.
LOUVAIN_WEIGHTS = {
    "student_of":        3.0,   # strongest lineage signal
    "collaborated_with": 2.0,   # sustained joint work
    "built_on":          1.0,   # standard influence
    "critiqued":         0.3,   # connection but not tradition membership
}

COMMUNITY_COLORS = [
    "#4e9af1",  # blue
    "#f4845f",  # coral
    "#2ec4b6",  # teal
    "#c77dff",  # purple
    "#80b918",  # olive
    "#ffd166",  # yellow
    "#ef476f",  # pink
    "#e76f51",  # orange
    "#a8dadc",  # light blue
    "#50fa7b",  # green
    "#ff9f1c",  # amber
    "#b5838d",  # mauve
]


# ── Community detection ───────────────────────────────────────────────────────

def build_louvain_graph(graph: dict) -> nx.Graph:
    """Build an undirected weighted graph using only intellectual lineage edges."""
    phil_ids = {n["id"] for n in graph["nodes"] if n["type"] == "Philosopher"}

    G = nx.Graph()
    for pid in phil_ids:
        G.add_node(pid)

    for e in graph["edges"]:
        w = LOUVAIN_WEIGHTS.get(e["relation"])
        if w is None:
            continue
        src, tgt = e["source"], e["target"]
        if src not in phil_ids or tgt not in phil_ids:
            continue
        if G.has_edge(src, tgt):
            G[src][tgt]["weight"] += w
        else:
            G.add_edge(src, tgt, weight=w)

    return G


def run_louvain(G: nx.Graph) -> dict[str, int]:
    return community_louvain.best_partition(G, weight="weight", random_state=42)


def pre_split_by_era(
    partition: dict[str, int],
    nodes_by_id: dict[str, dict],
    max_span_years: int = 550,
) -> dict[str, int]:
    """
    Split any Louvain community that spans more than max_span_years into
    sub-communities based on era boundaries.
    Ensures Ancient Greek (~pre-500 CE) and Medieval Scholastic (500-1400) are
    never in the same community.
    """
    from collections import defaultdict

    # Group by community
    raw: dict[int, list[str]] = defaultdict(list)
    for pid, cid in partition.items():
        raw[cid].append(pid)

    new_partition = dict(partition)
    next_id = max(partition.values()) + 1

    ERA_SPLIT_BOUNDARIES = [500, 1400]  # CE boundaries that matter

    for cid, members in raw.items():
        born_years = [
            (pid, nodes_by_id[pid].get("born"))
            for pid in members
            if nodes_by_id.get(pid, {}).get("born") is not None
        ]
        if not born_years:
            continue

        years = [y for _, y in born_years]
        span = max(years) - min(years)
        if span <= max_span_years:
            continue

        # Split across era boundaries
        buckets: dict[str, list[str]] = defaultdict(list)
        for pid, yr in born_years:
            if yr < 500:
                bucket = "ancient"
            elif yr < 1400:
                bucket = "medieval"
            elif yr < 1700:
                bucket = "early_modern"
            else:
                bucket = "modern"
            buckets[bucket].append(pid)

        # Also handle philosophers with unknown birth year
        for pid in members:
            if nodes_by_id.get(pid, {}).get("born") is None:
                era = nodes_by_id.get(pid, {}).get("era", "Unknown")
                bucket = {
                    "Ancient": "ancient", "Medieval": "medieval",
                    "Early Modern": "early_modern",
                }.get(era, "modern")
                buckets[bucket].append(pid)

        bucket_list = [b for b in buckets.values() if b]
        if len(bucket_list) <= 1:
            continue

        # First bucket keeps the original cid, rest get new ids
        first = True
        for bucket_members in bucket_list:
            if first:
                first = False
                continue
            for pid in bucket_members:
                new_partition[pid] = next_id
            next_id += 1

    n_new = len(set(new_partition.values())) - len(set(partition.values()))
    if n_new:
        print(f"  Pre-split: created {n_new} additional communities by era boundary")

    return new_partition


# ── Agent: autonomous community validator ─────────────────────────────────────

AGENT_SYSTEM = """You are an expert in the history of philosophy tasked with validating
automatically detected communities in a philosophy knowledge graph.

Your job:
1. Review the raw Louvain communities and produce 8–10 final clusters.
2. Each cluster must have a clear shared identity — same tradition AND roughly the same era.
3. Assign a precise name (2–4 words) and a 1-sentence UI description per cluster.

Hard rules — never break these:
- NEVER merge Ancient Greek philosophy (Plato, Aristotle, Stoics — pre-500 CE) with
  Medieval Scholasticism (Aquinas, Augustine, Anselm — 500–1400 CE). They are separated
  by 800 years and distinct in method and context. Keep them as separate clusters.
- NEVER merge Western and Eastern traditions into one cluster.
- No cluster should span more than ~600 years of intellectual history.
- Small communities (< 4 members) must merge into the nearest intellectual neighbour.
- Names must be specific: "Ancient Greek Philosophy" beats "Classical Philosophy";
  "Scholastic Tradition" beats "Medieval Philosophy".
- Never use generic labels like "Community N", "Western Philosophy", or "Modern Thought".

Target: 8–10 final clusters, each with a clear, defensible intellectual identity.
Return ONLY valid JSON array, no markdown, no explanation."""


def agent_validate_communities(
    client: anthropic.Anthropic,
    raw_partition: dict[str, int],
    nodes_by_id: dict[str, dict],
    degree: Counter,
) -> list[dict]:
    """
    Calls Claude as an autonomous agent to name and merge communities.
    Returns a list of final community dicts.
    """
    # Build community summaries for the agent
    raw_communities: dict[int, list] = defaultdict(list)
    for pid, cid in raw_partition.items():
        raw_communities[cid].append(pid)

    summaries = []
    for cid, members in sorted(raw_communities.items(), key=lambda x: -len(x[1])):
        top = sorted(members, key=lambda p: degree.get(p, 0), reverse=True)[:8]
        top_names = [nodes_by_id[p]["name"] for p in top if p in nodes_by_id]
        era_counts = Counter(
            nodes_by_id[p].get("era", "Unknown")
            for p in members if p in nodes_by_id
        )
        summaries.append({
            "louvain_id": cid,
            "size": len(members),
            "top_members": top_names,
            "era_distribution": dict(era_counts),
        })

    prompt = f"""Review these {len(summaries)} raw Louvain communities from a philosophy knowledge graph.
Merge and name them to produce 7–10 meaningful final communities.

Raw communities (sorted by size, largest first):
{json.dumps(summaries, indent=2)}

Return a JSON array where each object represents a FINAL community:
[
  {{
    "name": "...",
    "description": "One sentence for a UI tooltip.",
    "tradition": "Western" | "Eastern" | "Both",
    "louvain_ids": [list of raw louvain_id integers to merge into this community]
  }},
  ...
]

Merge all communities with < 5 members into the most intellectually similar larger one.
Every louvain_id must appear in exactly one final community."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=3000,
        system=AGENT_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()

    return json.loads(raw)


# ── Assemble final communities ────────────────────────────────────────────────

def build_final_communities(
    agent_output: list[dict],
    raw_partition: dict[str, int],
    nodes_by_id: dict[str, dict],
    degree: Counter,
) -> tuple[list[dict], dict[str, str]]:
    """
    Returns:
      communities_final: list of community dicts (for communities_final.json)
      pid_to_community:  philosopher_id → final community id string
    """
    # Build louvain_id → final community index
    louvain_to_final: dict[int, int] = {}
    for idx, comm in enumerate(agent_output):
        for lid in comm.get("louvain_ids", []):
            louvain_to_final[lid] = idx

    # Map philosopher → final community
    pid_to_final: dict[str, int] = {}
    for pid, louvain_id in raw_partition.items():
        pid_to_final[pid] = louvain_to_final.get(louvain_id, len(agent_output))

    # Build final community list with member data
    final_members: dict[int, list] = defaultdict(list)
    for pid, fidx in pid_to_final.items():
        final_members[fidx].append(pid)

    communities_final = []
    for idx, comm in enumerate(agent_output):
        members = final_members.get(idx, [])
        top = sorted(members, key=lambda p: degree.get(p, 0), reverse=True)[:6]
        top_names = [nodes_by_id[p]["name"] for p in top if p in nodes_by_id]
        cid = f"c{idx}"
        communities_final.append({
            "id":          cid,
            "name":        comm["name"],
            "description": comm.get("description", ""),
            "tradition":   comm.get("tradition", "Western"),
            "color":       COMMUNITY_COLORS[idx % len(COMMUNITY_COLORS)],
            "size":        len(members),
            "members":     members,
            "top_names":   top_names,
        })

    # Sort by size descending
    communities_final.sort(key=lambda c: -c["size"])

    pid_to_community_id: dict[str, str] = {}
    for comm in communities_final:
        for pid in comm["members"]:
            pid_to_community_id[pid] = comm["id"]

    return communities_final, pid_to_community_id


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    graph = json.load(open(GRAPH, encoding="utf-8"))
    nodes_by_id = {n["id"]: n for n in graph["nodes"]}
    phil_count = sum(1 for n in graph["nodes"] if n["type"] == "Philosopher")
    print(f"Loaded graph: {phil_count} philosophers, {len(graph['edges'])} edges")

    # Compute degree for top-member ranking
    degree: Counter = Counter()
    for e in graph["edges"]:
        if e["relation"] in LOUVAIN_WEIGHTS:
            degree[e["source"]] += 1
            degree[e["target"]] += 1

    print("Building Louvain graph (intellectual edges only)...")
    G = build_louvain_graph(graph)
    print(f"  Louvain graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("Running Louvain community detection...")
    raw_partition = run_louvain(G)
    raw_partition = pre_split_by_era(raw_partition, nodes_by_id)
    raw_counts = Counter(raw_partition.values())
    n_raw = len(raw_counts)
    print(f"  Detected {n_raw} raw communities")
    for cid, size in sorted(raw_counts.items(), key=lambda x: -x[1])[:10]:
        top = sorted([p for p, c in raw_partition.items() if c == cid],
                     key=lambda p: degree.get(p, 0), reverse=True)[:4]
        names = ", ".join(nodes_by_id[p]["name"] for p in top if p in nodes_by_id)
        print(f"    [{cid:2d}] {size:3d} members — {names}")

    print("\nCalling agent to validate and name communities...")
    client = anthropic.Anthropic(api_key=api_key)
    agent_output = agent_validate_communities(client, raw_partition, nodes_by_id, degree)
    print(f"  Agent produced {len(agent_output)} final communities:")
    for comm in agent_output:
        lids = comm.get("louvain_ids", [])
        print(f"    '{comm['name']}' (merging {lids})")

    communities_final, pid_to_community = build_final_communities(
        agent_output, raw_partition, nodes_by_id, degree
    )

    # Save communities_final.json
    COMMUNITIES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(COMMUNITIES_OUT, "w", encoding="utf-8") as f:
        json.dump(communities_final, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {COMMUNITIES_OUT}")

    # Patch philosopher nodes in graph.json with new community_id
    patched = 0
    for node in graph["nodes"]:
        if node["type"] == "Philosopher":
            new_cid = pid_to_community.get(node["id"], "c_unknown")
            if node.get("community_id") != new_cid:
                node["community_id"] = new_cid
                patched += 1
    with open(GRAPH, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    print(f"Patched {patched} philosopher nodes in graph.json")

    print("\nFinal communities:")
    for c in communities_final:
        print(f"  {c['name']:<45} {c['size']:3d} members  ({c['tradition']})")


if __name__ == "__main__":
    main()
