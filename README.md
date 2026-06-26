# Philosophy Knowledge Graph

> 217 philosophers. 1,000 edges. 10 clusters.

![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=flat-square&logo=python&logoColor=white) ![NetworkX](https://img.shields.io/badge/NetworkX-013243?style=flat-square) ![Anthropic](https://img.shields.io/badge/Anthropic_Claude-191919?style=flat-square&logo=anthropic&logoColor=white) ![Cytoscape.js](https://img.shields.io/badge/Cytoscape.js-F5A623?style=flat-square) ![GitHub Pages](https://img.shields.io/badge/GitHub_Pages-222222?style=flat-square&logo=github&logoColor=white)

**[Live Demo](https://magicmoses.github.io/philosopher-node)** · [GitHub](https://github.com/magicmoses/philosopher-node)

---

<img width="738" height="616" alt="grafik" src="https://github.com/user-attachments/assets/fbeeb44b-77d4-400b-918b-f15314152ff9" />


## What is it?

An interactive knowledge graph of philosophy. Each node represents a philosopher, each edge captures a relationship: who built on whom, who studied under whom, who critiqued whom. Ten clusters emerge from the graph. Explore.

---

## Why?

Traditional philosophy timelines are linear. This project instead represents philosophy as a network of intellectual influence, criticism, mentorship, and shared traditions, making hidden structures and communities easier to explore.

---

## Usage

- Hover a node to highlight its connections and label all connected thinkers
- Click a node once to freeze all its connections & display all connected philosopher names
- Detail panel for every philosopher on the side: abstract, key works, concepts, movements
- Search by name (top right)
- Filter bar: Focus / Influences / Debates / All edge presets

---

## Pipeline

```
01_parse.py          awesome-philosophy README -> raw philosopher list
02_enrich.py         abstracts, dates, key concepts, schools per philosopher
02b_enrich_edges.py  dense edge extraction: 6 relation types across all pairs
02c_fix_dates.py     fill missing birth years for ancient philosophers
03_build_graph.py    constructs the internal multi-type knowledge graph
03b_recluster.py     weighted Louvain on intellectual lineage edges
                     LLM agent names and merges raw communities autonomously
04_visualize.py      single-file Cytoscape.js HTML, sector seeding + cose layout
```

---

## Technical & Design

- Community detection runs on a weighted subgraph (`student_of=3.0, collaborated_with=2.0, built_on=1.0, critiqued=0.3`). 
- Era-boundary pre-split prevents Louvain from merging Ancient Greek and Medieval Scholastic philosophy despite their strong influence edges. Any raw community spanning >550 years is split at era boundaries (500 CE, 1400 CE) before the agent sees it.
- LLM-generated relationships are filtered through temporal consistency checks before entering the graph: Temporal sanity check filters anachronistic edges produced by the LLM (e.g. Plato influencing Schopenhauer in the wrong direction). After direction reversal, any edge where the influenced philosopher is >50 years older than the influencer is dropped.
- `contemporary_of` edges are excluded from clustering and visualization in this version.
- Ghost edges (opacity 0.55, no arrowheads, near-black) give the connectivity feel without clutter. Color and arrows appear on hover or preset activation.
- Nodes are rendered in ascending degree order so labels of highly connected philosophers remain visible.

---

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env  # add ANTHROPIC_API_KEY

python scripts/01_parse.py
python scripts/02_enrich.py
python scripts/02b_enrich_edges.py
python scripts/02c_fix_dates.py
python scripts/03_build_graph.py
python scripts/03b_recluster.py
python scripts/04_visualize.py
# generates output/graph.html
open output/graph.html
```

## Acknowledgement

The initial list of philosophers is based on the excellent list in [awesome-philosophy](https://github.com/HussainAther/awesome-philosophy) repository. Some entries were excluded during development because no sufficiently supported relationships to other philosophers in the dataset could be established. 

---

## Author
@magicmoses
