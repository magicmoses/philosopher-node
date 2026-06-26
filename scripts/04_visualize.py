"""
Generate the interactive knowledge graph visualization.

Reads data/graph.json and writes output/graph.html — a self-contained
single-file using Cytoscape.js + fcose layout.

Three levels:
  L1  Community overview  — cluster cards, click to filter
  L2  Philosopher graph   — fcose layout, communities visually separated
  L3  Detail sidebar      — click any node for works, concepts, connections

Usage:
    python scripts/04_visualize.py
"""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

GRAPH      = Path(__file__).parent.parent / "data" / "graph.json"
COMM_FILE  = Path(__file__).parent.parent / "data" / "communities_final.json"
OUT        = Path(__file__).parent.parent / "output" / "graph.html"
DOCS_OUT   = Path(__file__).parent.parent / "docs" / "index.html"

# ── Community names (Louvain IDs → human-readable) ─────────────────────────
# IDs come from 03_build_graph.py — update if graph is rebuilt.
COMMUNITY_NAMES = {
     0: "Analytic Philosophy",
     6: "Continental Philosophy",
    21: "Enlightenment & Political Thought",
     3: "Classical & Medieval",
    13: "Pragmatism & Process Thought",
    23: "Philosophy of Science",
    14: "Cognitive Science & Naturalism",
     1: "Chinese Philosophy",
    30: "Epistemology",
     4: "Japanese Buddhism",
     7: "Indian Philosophy",
     8: "Metaethics",
    -1: "Specialized Domains",
}

# Merge communities smaller than this into neighbours or "Specialized Domains"
MIN_COMMUNITY_SIZE = 4

COMMUNITY_COLORS = [
    "#4e9af1",  # blue           — Analytic
    "#f4845f",  # coral          — Continental
    "#2ec4b6",  # teal           — Enlightenment
    "#c77dff",  # purple         — Classical & Medieval
    "#80b918",  # olive          — Pragmatism
    "#ffd166",  # yellow         — Phil of Science
    "#ef476f",  # pink           — Cognitive Science
    "#e76f51",  # orange         — Chinese
    "#a8dadc",  # light blue     — Epistemology
    "#50fa7b",  # green          — Japanese Buddhism
    "#ff9f1c",  # amber          — Indian
    "#b5838d",  # mauve          — Metaethics
    "#6c757d",  # grey           — Specialized Domains
]

ERA_COLORS = {
    "Ancient":      "#c9a227",
    "Medieval":     "#a0522d",
    "Early Modern": "#1a6b8a",
    "Modern":       "#2e7d32",
    "Contemporary": "#3a5a8c",
    "Various":      "#e65c00",
    "Unknown":      "#555555",
}

EDGE_COLORS = {
    "student_of":        "#50fa7b",  # green
    "built_on":          "#4a9eff",  # blue
    "critiqued":         "#ffa500",  # orange
    "refuted":           "#ff5555",  # red
    "collaborated_with": "#c77dff",  # purple
    "contemporary_of":   "#555566",  # dark grey (temporal only, subtle)
}


# ── Data processing ──────────────────────────────────────────────────────────

def load():
    graph = json.load(open(GRAPH, encoding="utf-8"))
    # communities_final.json is written by 03b_recluster.py
    if COMM_FILE.exists():
        raw = json.load(open(COMM_FILE, encoding="utf-8"))
        communities = {c["id"]: c for c in raw}
    else:
        communities = {}
    return graph, communities


def build_communities(phil_nodes, edges):
    """
    Map Louvain community IDs to named clusters.
    Small communities are merged into the large neighbour with most shared edges.
    Returns (communities dict, philosopher_id → final_community_id mapping).
    """
    raw: dict[int, list] = defaultdict(list)
    for p in phil_nodes:
        raw[p["community_id"]].append(p)

    degree = Counter()
    edge_index: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        if e["relation"] in ("built_on", "critiqued", "student_of", "refuted"):
            degree[e["source"]] += 1
            degree[e["target"]] += 1
            edge_index[e["source"]].append(e["target"])
            edge_index[e["target"]].append(e["source"])

    large_ids = {cid for cid, m in raw.items() if len(m) >= MIN_COMMUNITY_SIZE}
    phil_cid  = {p["id"]: p["community_id"] for p in phil_nodes}

    reassign: dict[int, int] = {}
    for cid, members in raw.items():
        if cid in large_ids:
            continue
        counts: Counter = Counter()
        for p in members:
            for nb in edge_index.get(p["id"], []):
                nb_cid = phil_cid.get(nb, -1)
                if nb_cid in large_ids:
                    counts[nb_cid] += 1
        reassign[cid] = counts.most_common(1)[0][0] if counts else -1

    # Build final communities
    final: dict[int, dict] = {}
    ordered = sorted(large_ids, key=lambda c: -len(raw[c]))

    for idx, cid in enumerate(ordered):
        members = list(raw[cid])
        name = COMMUNITY_NAMES.get(cid, f"Cluster {cid}")
        top  = sorted(members, key=lambda p: degree.get(p["id"], 0), reverse=True)
        final[cid] = {
            "id": cid, "name": name,
            "color": COMMUNITY_COLORS[idx % len(COMMUNITY_COLORS)],
            "members": [p["id"] for p in members],
            "top_names": [p["name"] for p in top[:6]],
            "size": len(members),
        }

    # Reassign small → large
    for small_cid, target_cid in reassign.items():
        dest = target_cid if target_cid in final else -1
        if dest not in final:
            final[-1] = {
                "id": -1, "name": COMMUNITY_NAMES[-1],
                "color": COMMUNITY_COLORS[-1],
                "members": [], "top_names": [], "size": 0,
            }
        for p in raw[small_cid]:
            final[dest]["members"].append(p["id"])
            final[dest]["size"] += 1
            if len(final[dest]["top_names"]) < 6:
                final[dest]["top_names"].append(p["name"])

    # Reverse map
    pid_to_cid: dict[str, int] = {}
    for cid, data in final.items():
        for pid in data["members"]:
            pid_to_cid[pid] = cid

    return final, pid_to_cid


def compute_sector_positions(philosophers, pid_to_cid):
    """
    Compact sector seeding: clusters arranged in a circle, members spread
    in a tight sub-circle around the cluster centre.
    cose refinement then tightens everything up.
    """
    from collections import defaultdict

    # Sort clusters largest first so biggest ones get cardinal positions
    cluster_ids = sorted(set(pid_to_cid.values()), key=lambda c: -sum(
        1 for p in philosophers if pid_to_cid.get(p["id"]) == c
    ))
    n_clusters = max(len(cluster_ids), 1)
    RING_R = 280   # tight ring — cose will spread further if needed

    centres: dict[str, tuple] = {}
    for i, cid in enumerate(cluster_ids):
        angle = (i / n_clusters) * 2 * math.pi - math.pi / 2
        centres[cid] = (math.cos(angle) * RING_R, math.sin(angle) * RING_R)

    # Group members per cluster
    cluster_members: dict[str, list] = defaultdict(list)
    for p in philosophers:
        cluster_members[pid_to_cid.get(p["id"], "c_unknown")].append(p["id"])

    positions: dict[str, dict] = {}
    for cid, members in cluster_members.items():
        cx, cy = centres.get(cid, (0, 0))
        nm = max(len(members), 1)
        inner_r = 20 + math.sqrt(nm) * 13   # scales with cluster size
        for j, pid in enumerate(members):
            a = (j / nm) * 2 * math.pi
            h = (abs(hash(pid)) % 1000) / 1000.0
            r = inner_r * (0.6 + 0.8 * h)
            positions[pid] = {"x": cx + math.cos(a) * r, "y": cy + math.sin(a) * r}

    return positions


# Philosophers always shown with a label regardless of graph degree.
# These are culturally prominent but may have fewer graph connections.
ALWAYS_LABEL = {
    "adam_smith", "niccol_machiavelli", "s_ren_kierkegaard",
    "simone_de_beauvoir", "kongzi", "michel_foucault",
    "mary_wollstonecraft", "blaise_pascal", "hugo_grotius",
}

LABEL_DEGREE_THRESHOLD = 18  # permanent label for nodes with degree >= this


def short_label(name: str) -> str:
    """Use last name only for names longer than 18 characters."""
    if len(name) <= 18:
        return name
    parts = name.split()
    # Keep last part (surname), but if it's a particle ("de", "von", "van")
    # keep last two parts e.g. "Simone de Beauvoir" → "de Beauvoir"
    if len(parts) >= 2 and parts[-2].lower() in ("de", "von", "van", "van der"):
        return f"{parts[-2]} {parts[-1]}"
    return parts[-1]


def build_elements(graph, communities, pid_to_cid, positions):
    degree = Counter()
    for e in graph["edges"]:
        if e["relation"] in ("built_on", "critiqued", "student_of", "refuted"):
            degree[e["source"]] += 1
            degree[e["target"]] += 1

    phil_ids = {n["id"] for n in graph["nodes"] if n["type"] == "Philosopher"}
    nodes_list = []

    for n in graph["nodes"]:
        if n["type"] != "Philosopher":
            continue
        cid     = pid_to_cid.get(n["id"], -1)
        comm    = communities.get(cid, {})
        pos     = positions.get(n["id"], {"x": 0, "y": 0})
        deg     = degree.get(n["id"], 0)
        show_lbl = deg >= LABEL_DEGREE_THRESHOLD or n["id"] in ALWAYS_LABEL

        nodes_list.append({
            "data": {
                "id":              n["id"],
                "label":           short_label(n["name"]) if show_lbl else n["name"],
                "type":            "Philosopher",
                "era":             n.get("era", "Unknown"),
                "tradition":       n.get("tradition", "Western"),
                "abstract":        n.get("abstract", ""),
                "born":            n.get("born"),
                "died":            n.get("died"),
                "community_id":    cid,
                "community_label": comm.get("name", ""),
                "color":           comm.get("color", "#888"),
                "era_color":       ERA_COLORS.get(n.get("era", ""), "#555"),
                "degree":          deg,
                "show_label":      show_lbl,
            },
            "position": pos,
        })

    # Sort ascending by degree — higher-degree nodes render last → their labels
    # appear on top via Cytoscape's painter's algorithm, preventing overlap.
    nodes_list.sort(key=lambda e: e["data"]["degree"])
    elements = list(nodes_list)

    # Build born-year lookup for temporal sanity check
    born = {n["id"]: n.get("born") for n in graph["nodes"] if n["type"] == "Philosopher"}

    DIRECTIONAL = {"built_on", "student_of", "critiqued", "refuted"}

    for e in graph["edges"]:
        rel = e["relation"]

        # contemporary_of removed entirely from visualization
        if rel == "contemporary_of":
            continue

        if rel not in EDGE_COLORS:
            continue
        if e["source"] not in phil_ids or e["target"] not in phil_ids:
            continue

        # Arrow points FROM influenced (newer) TO influencer (older).
        # graph.json stores: source=influencer, target=influenced → reverse.
        if rel in DIRECTIONAL:
            src, tgt = e["target"], e["source"]
        else:
            src, tgt = e["source"], e["target"]

        # Temporal sanity check: after reversal, src must be ≥ tgt in birth year.
        # If src is older than tgt by >50 years, the LLM had the direction wrong — skip.
        if rel in DIRECTIONAL:
            src_born = born.get(src)
            tgt_born = born.get(tgt)
            if src_born and tgt_born and (src_born < tgt_born - 50):
                continue   # anachronistic edge — skip silently

        src_cid = pid_to_cid.get(src, -1)
        tgt_cid = pid_to_cid.get(tgt, -1)
        elements.append({"data": {
            "id":       e["id"],
            "source":   src,
            "target":   tgt,
            "relation": rel,
            "color":    EDGE_COLORS[rel],
            "intra":    src_cid == tgt_cid,
        }})

    return elements


def build_sidebar_data(graph):
    nodes_by_id = {n["id"]: n for n in graph["nodes"]}
    works    = defaultdict(list)
    concepts = defaultdict(list)
    schools  = defaultdict(list)

    for e in graph["edges"]:
        tgt = nodes_by_id.get(e["target"], {})
        if e["relation"] == "authored":
            works[e["source"]].append(tgt.get("name", ""))
        elif e["relation"] == "coined":
            concepts[e["source"]].append(tgt.get("name", ""))
        elif e["relation"] == "member_of":
            schools[e["source"]].append(tgt.get("name", ""))

    result = {}
    for n in graph["nodes"]:
        if n["type"] != "Philosopher":
            continue
        result[n["id"]] = {
            "works":    works[n["id"]][:8],
            "concepts": concepts[n["id"]][:8],
            "schools":  schools[n["id"]][:6],
        }
    return result


# ── HTML template ─────────────────────────────────────────────────────────────

def render_html(elements, communities, sidebar_data, meta):
    elements_json    = json.dumps(elements,    ensure_ascii=False)
    communities_json = json.dumps(
        {str(k): v for k, v in communities.items()}, ensure_ascii=False
    )
    sidebar_json     = json.dumps(sidebar_data, ensure_ascii=False)
    era_colors_json  = json.dumps(ERA_COLORS,   ensure_ascii=False)

    n_phil  = meta["philosopher_count"]
    n_edges = sum(1 for e in elements if "source" in e.get("data", {}))
    n_comm  = len(communities)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Philosophy Knowledge Graph</title>

<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.29.2/cytoscape.min.js"></script>

<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  background: #0d1117;
  color: #e6edf3;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}}

/* ── Header ─────────────────────────────────────────────────── */
header {{
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 10px 20px;
  background: #161b22;
  border-bottom: 1px solid #30363d;
  flex-shrink: 0;
}}

.header-title {{ font-size: 14px; font-weight: 600; white-space: nowrap; }}

.view-tabs {{
  display: flex;
  gap: 2px;
  background: #0d1117;
  border-radius: 6px;
  padding: 3px;
}}
.tab {{
  padding: 4px 14px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  border: none;
  background: transparent;
  color: #8b949e;
  transition: all 0.15s;
  white-space: nowrap;
}}
.tab.active {{ background: #21262d; color: #e6edf3; }}
.tab:hover:not(.active) {{ color: #c9d1d9; }}

.header-stats {{ font-size: 11px; color: #8b949e; margin-left: auto; white-space: nowrap; }}

/* ── Cluster filter bar ─────────────────────────────────────── */
#filter-bar {{
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 20px;
  background: #161b22;
  border-bottom: 1px solid #30363d;
  flex-shrink: 0;
  flex-wrap: wrap;
  min-height: 38px;
}}
.filter-label {{ font-size: 10px; color: #8b949e; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; white-space: nowrap; }}
.chip {{
  padding: 3px 10px;
  border-radius: 99px;
  font-size: 11px;
  cursor: pointer;
  border: 1.5px solid transparent;
  transition: opacity 0.15s, transform 0.1s;
  white-space: nowrap;
  font-weight: 500;
}}
.chip.on  {{ opacity: 1; }}
.chip.off {{ opacity: 0.3; }}
.chip:hover {{ opacity: 0.85; transform: translateY(-1px); }}

/* ── Main content ────────────────────────────────────────────── */
.main {{ display: flex; flex: 1; overflow: hidden; position: relative; }}

#cy {{ flex: 1; background: #0d1117; }}

/* ── Community grid (L1) ─────────────────────────────────────── */
#community-grid {{
  display: none;
  flex: 1;
  padding: 24px 32px;
  overflow-y: auto;
  flex-wrap: wrap;
  gap: 14px;
  align-content: flex-start;
  justify-content: center;
}}
#community-grid.visible {{ display: flex; }}

.cluster-card {{
  flex: 0 1 260px;
  padding: 18px 20px;
  border-radius: 12px;
  cursor: pointer;
  border: 1px solid transparent;
  transition: transform 0.15s, box-shadow 0.2s;
}}
.cluster-card:hover {{
  transform: translateY(-3px);
  box-shadow: 0 10px 32px rgba(0,0,0,0.5);
}}
.cluster-name {{ font-size: 15px; font-weight: 700; margin-bottom: 4px; }}
.cluster-size {{ font-size: 11px; opacity: 0.65; margin-bottom: 10px; }}
.cluster-members {{ font-size: 11px; opacity: 0.75; line-height: 1.6; }}

/* ── Info card (fixed overlay, no layout dependency) ─────────── */
#info-card {{
  display: none;
  position: fixed;
  top: 96px;
  right: 20px;
  width: 290px;
  max-height: calc(100vh - 130px);
  overflow-y: auto;
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 10px;
  padding: 18px 20px;
  z-index: 300;
  box-shadow: 0 8px 32px rgba(0,0,0,0.55);
}}
#info-card.open {{ display: block; }}

.s-name  {{ font-size: 16px; font-weight: 700; line-height: 1.3; margin-bottom: 3px; }}
.s-dates {{ font-size: 11px; color: #8b949e; margin-bottom: 10px; }}
.s-era   {{
  display: inline-block;
  padding: 2px 9px;
  border-radius: 99px;
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 12px;
}}
.s-abstract {{ font-size: 12px; color: #c9d1d9; line-height: 1.65; margin-bottom: 14px; }}
.s-section {{ margin-bottom: 12px; }}
.s-section-title {{
  font-size: 9px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #8b949e;
  margin-bottom: 5px;
}}
.tag-row {{ display: flex; flex-wrap: wrap; gap: 4px; }}
.tag {{
  padding: 2px 8px;
  border-radius: 8px;
  font-size: 11px;
  background: #21262d;
  color: #c9d1d9;
}}
.tag.link {{ cursor: pointer; transition: background 0.12s; }}
.tag.link:hover {{ background: #30363d; }}
.work-list {{ list-style: none; }}
.work-list li {{
  font-size: 11px;
  color: #c9d1d9;
  padding: 3px 0;
  border-bottom: 1px solid #21262d;
  font-style: italic;
}}
.no-data {{ font-size: 11px; color: #444; }}

/* ── Legend ──────────────────────────────────────────────────── */
#legend {{
  display: flex;
  gap: 14px;
  align-items: center;
  padding: 7px 20px;
  background: #161b22;
  border-top: 1px solid #30363d;
  flex-shrink: 0;
  flex-wrap: wrap;
}}
.leg {{ display: flex; align-items: center; gap: 5px; font-size: 10px; color: #8b949e; }}
.leg-line {{ width: 20px; height: 2px; border-radius: 1px; }}

/* ── Tooltip ─────────────────────────────────────────────────── */
#tooltip {{
  position: fixed;
  background: #1c2128;
  border: 1px solid #30363d;
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 12px;
  color: #e6edf3;
  pointer-events: none;
  z-index: 999;
  display: none;
  max-width: 180px;
}}

.preset-btn {{
  padding: 3px 10px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 500;
  cursor: pointer;
  border: 1px solid #30363d;
  background: transparent;
  color: #8b949e;
  transition: all 0.15s;
  white-space: nowrap;
}}
.preset-btn:hover  {{ color: #e6edf3; border-color: #555; }}
.preset-btn.active {{ background: #21262d; color: #e6edf3; border-color: #555; }}

#preset-desc {{
  width: 100%;
  font-size: 10px;
  color: #555;
  padding: 0 20px 5px;
  background: #161b22;
  margin-top: -3px;
}}

/* ── Search ── */
#search-wrap {{
  position: relative;
  margin-left: auto;
}}
#search-input {{
  background: #0d1117;
  border: 1px solid #30363d;
  border-radius: 6px;
  color: #e6edf3;
  font-size: 12px;
  padding: 5px 10px;
  width: 200px;
  outline: none;
  transition: border-color 0.15s;
}}
#search-input:focus  {{ border-color: #58a6ff; }}
#search-input::placeholder {{ color: #555; }}
#search-dropdown {{
  position: absolute;
  top: calc(100% + 4px);
  right: 0;
  width: 240px;
  background: #1c2128;
  border: 1px solid #30363d;
  border-radius: 8px;
  overflow: hidden;
  z-index: 500;
  display: none;
  box-shadow: 0 8px 24px rgba(0,0,0,0.5);
}}
.search-item {{
  padding: 8px 12px;
  font-size: 12px;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  transition: background 0.1s;
}}
.search-item:hover {{ background: #30363d; }}
.search-item-name  {{ color: #e6edf3; }}
.search-item-era   {{ font-size: 10px; color: #8b949e; }}

::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: #30363d; border-radius: 3px; }}
</style>
</head>
<body>

<header>
  <span class="header-title">Philosophy Knowledge Graph</span>
  <div class="view-tabs">
    <button class="tab" id="tab-l1" onclick="setView('l1')">Clusters</button>
    <button class="tab active" id="tab-l2" onclick="setView('l2')">Graph</button>
  </div>
  <div id="search-wrap">
    <input id="search-input" type="text" placeholder="Search philosopher…" autocomplete="off" spellcheck="false">
    <div id="search-dropdown"></div>
  </div>
  <span class="header-stats">{n_phil} thinkers &nbsp;·&nbsp; {n_edges} connections &nbsp;·&nbsp; {n_comm} clusters</span>
</header>

<div id="filter-bar">
  <span class="filter-label">Cluster</span>
  <div id="chips"></div>
  <div style="width:1px;background:#30363d;align-self:stretch;margin:0 4px"></div>
  <span class="filter-label">Edges</span>
  <div id="edge-chips"></div>
  <div style="width:1px;background:#30363d;align-self:stretch;margin:0 4px"></div>
  <span class="filter-label">View</span>
  <button class="preset-btn active" id="preset-none"      onclick="setPreset('none')">Focus</button>
  <button class="preset-btn"        id="preset-influences" onclick="setPreset('influences')">Influences</button>
  <button class="preset-btn"        id="preset-debates"   onclick="setPreset('debates')">Debates</button>
  <button class="preset-btn"        id="preset-all"       onclick="setPreset('all')">All</button>
</div>
<div id="preset-desc">Click any thinker to explore their connections</div>

<div class="main">
  <div id="community-grid"></div>
  <div id="cy"></div>
  <div id="info-card"></div>
</div>

<div id="legend">
  <span style="font-size:10px;color:#8b949e;font-weight:700;text-transform:uppercase;letter-spacing:.08em">Edges</span>
  <div class="leg"><div class="leg-line" style="background:#4a9eff"></div>built on</div>
  <div class="leg"><div class="leg-line" style="background:#ffa500"></div>critiqued</div>
  <div class="leg"><div class="leg-line" style="background:#50fa7b"></div>student of</div>
  <div class="leg"><div class="leg-line" style="background:#ff5555"></div>refuted</div>
  <div class="leg"><div class="leg-line" style="background:#c77dff"></div>collaborated</div>
</div>

<div id="tooltip"></div>

<script>
const ELEMENTS    = {elements_json};
const COMMUNITIES = {communities_json};
const SIDEBAR     = {sidebar_json};
const ERA_COLORS  = {era_colors_json};

// ── State ────────────────────────────────────────────────────────────────────
let cy;
let activeView = 'l2';
let activeCid  = null;  // currently filtered community id (null = all)
let selected   = null;

// ── Cytoscape init ───────────────────────────────────────────────────────────
function initCy() {{
  cy = cytoscape({{
    container: document.getElementById('cy'),
    elements: ELEMENTS,

    style: [
      // Philosopher nodes
      {{
        selector: 'node[type = "Philosopher"]',
        style: {{
          'background-color':   'data(color)',
          'width':              e => nodeSize(e.data('degree')),
          'height':             e => nodeSize(e.data('degree')),
          'border-width':       0,
          'label':              'data(label)',
          'font-size':               6,
          'color':                   '#c9d1d9',
          'text-valign':             'bottom',
          'text-margin-y':           5,
          'text-wrap':               'none',
          // Permanent label for well-known nodes; opaque background prevents overlap
          'text-opacity':            e => e.data('show_label') ? 0.9 : 0,
          'text-background-color':   '#0d1117',
          'text-background-opacity': e => e.data('show_label') ? 0.95 : 0,
          'text-background-padding': '2px',
          'min-zoomed-font-size':    5,
          // Higher-degree nodes paint on top → their labels win overlap battles
          'z-index':                 e => e.data('degree'),
          'transition-property': 'opacity, border-width, border-color',
          'transition-duration': '0.12s',
        }}
      }},

      // Ghost layer — always visible, very faint grey, no arrows
      // Gives the "connected network" feeling without clutter
      {{
        selector: 'edge',
        style: {{
          'line-color':          '#2a2d3a',
          'target-arrow-shape':  'none',
          'curve-style':         'bezier',
          'opacity':             0.55,
          'width':               0.5,
          'transition-property': 'opacity, width, line-color, target-arrow-color',
          'transition-duration': '0.15s',
        }}
      }},
      // contemporary_of — even fainter (purely temporal)
      {{
        selector: 'edge[relation = "contemporary_of"]',
        style: {{ 'opacity': 0.18, 'width': 0.4 }}
      }},
      // Ambient: preset active — colour restored, arrows appear
      {{
        selector: 'edge.ambient',
        style: {{
          'line-color':         'data(color)',
          'target-arrow-color': 'data(color)',
          'target-arrow-shape': 'triangle',
          'arrow-scale':        0.6,
          'opacity': e => {{
            const r = e.data('relation');
            if (r === 'student_of' || r === 'refuted') return 0.65;
            if (r === 'critiqued')         return 0.45;
            if (r === 'collaborated_with') return 0.55;
            return 0.25;
          }},
          'width': e => e.data('relation') === 'built_on' ? 1.0 : 1.5,
        }}
      }},
      // Hover edge — bright, full colour
      {{
        selector: 'edge.hover-edge',
        style: {{
          'line-color':         'data(color)',
          'target-arrow-color': 'data(color)',
          'target-arrow-shape': 'triangle',
          'arrow-scale':        0.7,
          'opacity':            0.85,
          'width':              1.8,
        }}
      }},

      // Neighbour label — shown when a connected node is selected/highlighted
      {{
        selector: 'node.neighbor-label',
        style: {{
          'text-opacity':            1,
          'text-background-color':   '#0d1117',
          'text-background-opacity': 0.92,
          'text-background-padding': '2px',
          'font-size':               6,
          'color':                   '#c9d1d9',
          'z-index':                 500,
        }}
      }},
      // Hover — always show label, pop to top z-index so it never hides under others
      {{
        selector: 'node.hovered',
        style: {{
          'border-width':            2,
          'border-color':            '#ffffff88',
          'text-opacity':            1,
          'text-background-color':   '#0d1117',
          'text-background-opacity': 0.92,
          'text-background-padding': '3px',
          'font-size':               8,
          'color':                   '#ffffff',
          'z-index':                 9999,
        }}
      }},

      // Selection highlight
      {{
        selector: 'node.selected',
        style: {{
          'border-width':            2.5,
          'border-color':            '#ffffff',
          'text-opacity':            1,
          'text-background-color':   '#0d1117',
          'text-background-opacity': 0.92,
          'text-background-padding': '3px',
          'font-size':               9,
          'color':                   '#ffffff',
          'z-index':                 9999,
        }}
      }},
      {{
        selector: 'edge.highlighted',
        style: {{ 'opacity': 1.0, 'width': 2.2 }}
      }},
      {{
        selector: '.dimmed',
        style: {{ 'opacity': 0.05 }}
      }},
    ],

    layout: {{
      name:    'preset',
      animate: false,
    }},

    wheelSensitivity: 0.3,
  }});

  // Refine from sector seeds with cose.
  // High gravity keeps everything concentrated; differential idealEdgeLength
  // pulls intra-cluster nodes tight while letting inter-cluster edges breathe.
  cy.ready(() => {{
    cy.layout({{
      name:             'cose',
      animate:          true,
      animationDuration: 1000,
      fit:              true,
      padding:          60,
      randomize:        false,       // start from sector seeds
      nodeRepulsion:    7000,        // moderate — less spreading
      idealEdgeLength:  e => e.data('intra') ? 40 : 160,
      edgeElasticity:   e => e.data('intra') ? 0.6 : 0.15,
      nestingFactor:    0.1,
      gravity:          0.75,        // high gravity = compact, no pyramid
      numIter:          1500,
      initialTemp:      80,          // don't stray far from seeds
      coolingFactor:    0.99,
      minTemp:          1.0,
    }}).on('layoutstop', () => {{
      removeOverlaps(cy);
      initSearch();
    }}).run();
  }});

  // Events
  // Single tap → highlight connections, label neighbours, open info card
  cy.on('tap', 'node[type = "Philosopher"]', e => selectNode(e.target));
  // Tap on empty background → clear everything
  cy.on('tap', e => {{ if (e.target === cy) deselect(); }});
  cy.on('mouseover', 'node[type = "Philosopher"]', e => {{
    const node = e.target;
    node.addClass('hovered');
    // Show this node's edges on hover (only active types)
    node.connectedEdges().forEach(edge => {{
      if (activeEdgeTypes.has(edge.data('relation')) || activeEdgeTypes.size === 0) {{
        edge.addClass('hover-edge');
      }}
    }});
    showTooltip(e);
  }});
  cy.on('mouseout', 'node[type = "Philosopher"]', e => {{
    const node = e.target;
    node.removeClass('hovered');
    // Only remove hover-edge if not also highlighted (from selection)
    node.connectedEdges().not('.highlighted').removeClass('hover-edge');
    hideTooltip();
  }});
}}

function nodeSize(deg) {{
  return Math.max(5, Math.min(12, 5 + deg * 0.45));
}}

// ── Selection & sidebar ───────────────────────────────────────────────────────
function highlightNode(node) {{
  if (selected) selected.removeClass('selected');
  // Clear previous neighbour labels
  cy.nodes().removeClass('neighbor-label');

  selected = node;
  node.addClass('selected');
  cy.elements().addClass('dimmed').removeClass('highlighted');
  node.removeClass('dimmed');

  const hood = node.closedNeighborhood();
  hood.removeClass('dimmed');
  hood.edges().addClass('highlighted').removeClass('dimmed');

  // Label all connected nodes for this view
  hood.nodes().forEach(n => {{
    if (n.id() !== node.id()) n.addClass('neighbor-label');
  }});
}}

function selectNode(node) {{
  highlightNode(node);
  renderSidebar(node.data());
}}

function deselect() {{
  if (selected) selected.removeClass('selected');
  selected = null;
  cy.elements().removeClass('dimmed highlighted');
  cy.edges().removeClass('hover-edge');
  cy.nodes().removeClass('neighbor-label');
  document.getElementById('info-card').classList.remove('open');
}}

function renderSidebar(d) {{
  const card  = document.getElementById('info-card');
  const extra = SIDEBAR[d.id] || {{}};
  const eraColor = ERA_COLORS[d.era] || '#555';
  const dash  = '—';
  const dates = (d.born || d.died)
    ? '(' + (d.born || '?') + ' – ' + (d.died || 'present') + ')'
    : '';

  function esc(s) {{
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }}

  function tagRow(items) {{
    if (!items || !items.length) return '<span class="no-data">' + dash + '</span>';
    return '<div class="tag-row">' +
      items.map(s => '<span class="tag">' + esc(s) + '</span>').join('') +
      '</div>';
  }}

  function workRows(items) {{
    if (!items || !items.length) return '<span class="no-data">' + dash + '</span>';
    return '<ul class="work-list">' +
      items.map(w => '<li>' + esc(w) + '</li>').join('') +
      '</ul>';
  }}

  function linkRow(items) {{
    if (!items || !items.length) return '<span class="no-data">' + dash + '</span>';
    return '<div class="tag-row">' +
      items.map(s => '<span class="tag link" data-jump="' + esc(s) + '">' + esc(s) + '</span>').join('') +
      '</div>';
  }}

  const node    = cy.getElementById(d.id);
  // After edge reversal: outgoers = philosophers this one built on (sources)
  //                      incomers = philosophers who built on this one
  const outNbrs = node.outgoers('node[type="Philosopher"]').map(n => n.data('label'));
  const inNbrs  = node.incomers('node[type="Philosopher"]').map(n => n.data('label'));

  card.innerHTML =
    '<div class="s-name">' + esc(d.label) + '</div>' +
    '<div class="s-dates">' + dates + '</div>' +
    '<span class="s-era" style="background:' + eraColor + '28;color:' + eraColor + '">' + esc(d.era) + '</span>' +
    '<p class="s-abstract">' + esc(d.abstract || dash) + '</p>' +
    '<div class="s-section"><div class="s-section-title">Key Works</div>' + workRows(extra.works) + '</div>' +
    '<div class="s-section"><div class="s-section-title">Key Concepts</div>' + tagRow(extra.concepts) + '</div>' +
    '<div class="s-section"><div class="s-section-title">Movements</div>' + tagRow(extra.schools) + '</div>' +
    '<div class="s-section"><div class="s-section-title">Built on / responded to (' + outNbrs.length + ')</div>' + linkRow(outNbrs) + '</div>' +
    '<div class="s-section"><div class="s-section-title">Influenced (' + inNbrs.length + ')</div>' + linkRow(inNbrs) + '</div>';

  // Attach jump handlers via data attribute (avoids inline-onclick quote issues)
  card.querySelectorAll('[data-jump]').forEach(el => {{
    el.addEventListener('click', () => jumpTo(el.getAttribute('data-jump')));
  }});

  card.classList.add('open');
}}

function jumpTo(label) {{
  const node = cy.nodes().filter(n => n.data('label') === label).first();
  if (node.length) {{
    cy.animate({{ center: {{ eles: node }}, zoom: Math.max(cy.zoom(), 1.5) }}, {{ duration: 300 }});
    selectNode(node);
  }}
}}

// ── Tooltip ───────────────────────────────────────────────────────────────────
const tt = document.getElementById('tooltip');
let ttMove;
function showTooltip(e) {{
  const d = e.target.data();
  tt.textContent = d.label + (d.born ? ` (${{d.born}})` : '');
  tt.style.display = 'block';
  document.removeEventListener('mousemove', ttMove);
  ttMove = ev => {{
    tt.style.left = (ev.clientX + 14) + 'px';
    tt.style.top  = (ev.clientY - 32) + 'px';
  }};
  document.addEventListener('mousemove', ttMove);
}}
function hideTooltip() {{
  tt.style.display = 'none';
  document.removeEventListener('mousemove', ttMove);
}}

// ── Filter chips ──────────────────────────────────────────────────────────────
function buildChips() {{
  const container = document.getElementById('chips');
  Object.entries(COMMUNITIES)
    .sort((a, b) => b[1].size - a[1].size)
    .forEach(([cid, c]) => {{
      const chip = document.createElement('span');
      chip.className = 'chip on';
      chip.style.background    = c.color + '22';
      chip.style.color         = c.color;
      chip.style.borderColor   = c.color + '66';
      chip.textContent         = c.name;
      chip.dataset.cid         = cid;
      chip.addEventListener('click', () => filterByCommunity(cid));
      container.appendChild(chip);
    }});
}}

function filterByCommunity(cid) {{
  if (activeCid === cid) {{
    // Reset
    activeCid = null;
    document.querySelectorAll('.chip').forEach(c => c.classList.replace('off','on') || c.classList.add('on'));
    cy.nodes().style('display', 'element');
    cy.edges().style('display', 'element');
    cy.fit(60);
  }} else {{
    activeCid = cid;
    document.querySelectorAll('.chip').forEach(c => {{
      c.classList.toggle('on',  c.dataset.cid === cid);
      c.classList.toggle('off', c.dataset.cid !== cid);
    }});
    const members = new Set(COMMUNITIES[cid]?.members || []);
    cy.nodes('[type="Philosopher"]').forEach(n => {{
      n.style('display', members.has(n.id()) ? 'element' : 'none');
    }});
    // Edges: keep ambient if active, but only within visible cluster
    cy.edges().forEach(e => {{
      const vis = members.has(e.source().id()) && members.has(e.target().id());
      if (!vis) e.removeClass('ambient hover-edge highlighted');
    }});
    cy.fit(cy.nodes(':visible'), 60);
    deselect();
  }}
}}

// ── Community grid (L1) ───────────────────────────────────────────────────────
function renderGrid() {{
  const grid = document.getElementById('community-grid');
  grid.innerHTML = '';
  Object.entries(COMMUNITIES)
    .sort((a, b) => b[1].size - a[1].size)
    .forEach(([cid, c]) => {{
      const card = document.createElement('div');
      card.className = 'cluster-card';
      card.style.background   = c.color + '14';
      card.style.borderColor  = c.color + '40';
      card.innerHTML = `
        <div class="cluster-name" style="color:${{c.color}}">${{c.name}}</div>
        <div class="cluster-size" style="color:${{c.color}}">${{c.size}} thinkers</div>
        <div class="cluster-members">${{c.top_names.join(' · ')}}</div>
      `;
      card.addEventListener('click', () => {{
        setView('l2');
        setTimeout(() => filterByCommunity(cid), 50);
      }});
      grid.appendChild(card);
    }});
}}

// ── View switch ───────────────────────────────────────────────────────────────
function setView(v) {{
  activeView = v;
  document.getElementById('tab-l1').classList.toggle('active', v === 'l1');
  document.getElementById('tab-l2').classList.toggle('active', v === 'l2');
  document.getElementById('cy').style.display            = v === 'l2' ? 'block' : 'none';
  document.getElementById('community-grid').classList.toggle('visible', v === 'l1');
  document.getElementById('filter-bar').style.display    = v === 'l2' ? 'flex'  : 'none';
  document.getElementById('legend').style.display        = v === 'l2' ? 'flex'  : 'none';
  if (v === 'l1') renderGrid();
}}

// ── Edge type filters ────────────────────────────────────────────────────────
const EDGE_META = {{
  student_of:        {{ label: 'student of',   color: '#50fa7b' }},
  built_on:          {{ label: 'built on',     color: '#4a9eff' }},
  critiqued:         {{ label: 'critiqued',    color: '#ffa500' }},
  refuted:           {{ label: 'refuted',      color: '#ff5555' }},
  collaborated_with: {{ label: 'collaborated', color: '#c77dff' }},
}};

// contemporary_of excluded from all presets — 1151 edges of pure temporal noise
const PRESETS = {{
  none:      {{ types: [],                               desc: 'Click any thinker to explore their connections' }},
  influences: {{ types: ['student_of', 'built_on'],       desc: 'Intellectual lineage — who built on whom and teacher-student bonds' }},
  debates:   {{ types: ['critiqued', 'refuted'],          desc: 'Where thinkers clashed — critiques and explicit refutations' }},
  all:       {{ types: ['student_of', 'built_on', 'critiqued', 'refuted', 'collaborated_with'], desc: 'All intellectual connections (no temporal edges)' }},
}};

let activeEdgeTypes = new Set();  // empty = Focus mode (edges on-demand only)

function buildEdgeChips() {{
  const container = document.getElementById('edge-chips');
  Object.entries(EDGE_META).forEach(([rel, meta]) => {{
    const chip = document.createElement('span');
    chip.className = 'chip on';
    chip.style.background  = meta.color + '22';
    chip.style.color       = meta.color;
    chip.style.borderColor = meta.color + '66';
    chip.textContent       = meta.label;
    chip.dataset.rel       = rel;
    chip.addEventListener('click', () => toggleEdgeType(rel));
    container.appendChild(chip);
  }});
}}

function toggleEdgeType(rel) {{
  if (activeEdgeTypes.has(rel)) {{
    activeEdgeTypes.delete(rel);
  }} else {{
    activeEdgeTypes.add(rel);
  }}
  applyEdgeVisibility();
  syncEdgeChips();
  // Clear preset highlight when manually toggling
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
}}

function applyEdgeVisibility() {{
  cy.edges().forEach(e => {{
    const rel = e.data('relation');
    const active = activeEdgeTypes.has(rel);
    // Always keep in DOM (display:element) — opacity does the hiding
    e.style('display', 'element');
    e.toggleClass('ambient', active && activeEdgeTypes.size > 0);
  }});
}}

function syncEdgeChips() {{
  document.querySelectorAll('#edge-chips .chip').forEach(c => {{
    c.classList.toggle('on',  activeEdgeTypes.has(c.dataset.rel));
    c.classList.toggle('off', !activeEdgeTypes.has(c.dataset.rel));
  }});
}}

function setPreset(name) {{
  const preset = PRESETS[name] || PRESETS.all;
  activeEdgeTypes = new Set(preset.types);
  applyEdgeVisibility();
  syncEdgeChips();
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('preset-' + name)?.classList.add('active');
  document.getElementById('preset-desc').textContent = preset.desc;
}}

// ── Search ────────────────────────────────────────────────────────────────────
function initSearch() {{
  const input    = document.getElementById('search-input');
  const dropdown = document.getElementById('search-dropdown');

  // Build lookup: lowercase name → node data
  const index = [];
  cy.nodes('[type = "Philosopher"]').forEach(n => {{
    index.push({{ id: n.id(), label: n.data('label'), era: n.data('era') }});
  }});
  index.sort((a, b) => a.label.localeCompare(b.label));

  function showResults(query) {{
    const q = query.toLowerCase().trim();
    if (!q) {{ dropdown.style.display = 'none'; return; }}

    const matches = index
      .filter(p => p.label.toLowerCase().includes(q))
      .slice(0, 8);

    if (!matches.length) {{ dropdown.style.display = 'none'; return; }}

    dropdown.innerHTML = matches.map(p => `
      <div class="search-item" data-id="${{p.id}}">
        <span class="search-item-name">${{p.label}}</span>
        <span class="search-item-era">${{p.era}}</span>
      </div>`).join('');

    dropdown.querySelectorAll('.search-item').forEach(item => {{
      item.addEventListener('click', () => {{
        const node = cy.getElementById(item.dataset.id);
        if (node.length) {{
          // Switch to graph view if needed
          if (activeView !== 'l2') setView('l2');
          setTimeout(() => {{
            cy.animate({{ center: {{ eles: node }}, zoom: 2.2 }}, {{ duration: 400 }});
            selectNode(node);
          }}, activeView !== 'l2' ? 100 : 0);
        }}
        input.value = item.querySelector('.search-item-name').textContent;
        dropdown.style.display = 'none';
      }});
    }});

    dropdown.style.display = 'block';
  }}

  input.addEventListener('input', e => showResults(e.target.value));
  input.addEventListener('keydown', e => {{
    if (e.key === 'Escape') {{ dropdown.style.display = 'none'; input.blur(); }}
    if (e.key === 'Enter') {{
      const first = dropdown.querySelector('.search-item');
      if (first) first.click();
    }}
  }});

  // Close on outside click
  document.addEventListener('click', e => {{
    if (!e.target.closest('#search-wrap')) dropdown.style.display = 'none';
  }});
}}

// ── Overlap removal ───────────────────────────────────────────────────────────
// Bidirectional push — no temporal Y constraint since we use cluster layout.
// PAD is generous enough to leave breathing room for labels below nodes.
function removeOverlaps(cy) {{
  const PAD   = 18;  // extra space for labels below nodes
  const nodes = cy.nodes('[type = "Philosopher"]:visible');

  for (let iter = 0; iter < 100; iter++) {{
    let moved = false;
    for (let i = 0; i < nodes.length; i++) {{
      for (let j = i + 1; j < nodes.length; j++) {{
        const a = nodes[i], b = nodes[j];
        const ap = a.position(), bp = b.position();
        const dx = bp.x - ap.x;
        const dy = bp.y - ap.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
        const minDist = (a.width() + b.width()) / 2 + PAD;

        if (dist < minDist) {{
          const push = (minDist - dist) / 2 + 0.5;
          const nx = (dx / dist) * push;
          const ny = (dy / dist) * push;
          a.position({{ x: ap.x - nx, y: ap.y - ny }});
          b.position({{ x: bp.x + nx, y: bp.y + ny }});
          moved = true;
        }}
      }}
    }}
    if (!moved) break;
  }}
}}

// ── Boot ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {{
  initCy();
  buildChips();
  buildEdgeChips();
}});
</script>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    graph, communities = load()

    phil_nodes = [n for n in graph["nodes"] if n["type"] == "Philosopher"]

    # Build pid → community_id from graph nodes (set by 03b_recluster.py)
    pid_to_cid: dict[str, str] = {
        n["id"]: n.get("community_id", "c_unknown") for n in phil_nodes
    }

    # Fallback: if communities_final.json missing, rebuild from internal data
    if not communities:
        communities, pid_to_cid = build_communities(phil_nodes, graph["edges"])

    positions = compute_sector_positions(phil_nodes, pid_to_cid)
    elements = build_elements(graph, communities, pid_to_cid, positions)
    sidebar  = build_sidebar_data(graph)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    DOCS_OUT.parent.mkdir(parents=True, exist_ok=True)
    html = render_html(elements, communities, sidebar, graph["meta"])
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    with open(DOCS_OUT, "w", encoding="utf-8") as f:
        f.write(html)

    n_edges = sum(1 for e in elements if "source" in e.get("data", {}))
    print(f"Saved -> {OUT}")
    print(f"  {len(phil_nodes)} philosophers  {n_edges} edges  {len(communities)} clusters")
    print()
    print("Clusters:")
    for cid, c in sorted(communities.items(), key=lambda x: -x[1]["size"]):
        print(f"  {c['name']:<45} {c['size']:3d} thinkers  ({c.get('tradition','')})")


if __name__ == "__main__":
    main()
