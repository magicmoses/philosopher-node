"""
Parse the awesome-philosophy README into structured philosopher data.

Source: https://github.com/HussainAther/awesome-philosophy
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

README_URL = "https://raw.githubusercontent.com/HussainAther/awesome-philosophy/master/README.md"
OUT = Path(__file__).parent.parent / "data" / "raw" / "philosophers.json"

# Maps README section headings to (tradition, era) tuples
SECTION_MAP = {
    "Classical ethics": ("Western", "Ancient"),
    "Christian and Medieval ethics": ("Western", "Medieval"),
    "Modern ethics": ("Western", "Modern"),
    "Postmodern ethics": ("Western", "Contemporary"),
    "Bioethics": ("Western", "Contemporary"),
    "Meta-ethics (Metaethics)": ("Western", "Contemporary"),
    "Epistemology": ("Western", "Modern"),
    "Logic": ("Western", "Modern"),
    "Aesthetics": ("Western", "Modern"),
    "Metaphysics": ("Western", "Modern"),
    "Philosophy of the mind": ("Western", "Contemporary"),
    "Classical philosophy": ("Western", "Ancient"),
    "Christian and Medieval": ("Western", "Medieval"),
    "Early modern": ("Western", "Early Modern"),
    "Phenomenology and existentialism": ("Western", "Contemporary"),
    "Hermeneutics and deconstruction": ("Western", "Contemporary"),
    "Structuralism and post-structuralism": ("Western", "Contemporary"),
    "Critical theory and Marxism": ("Western", "Contemporary"),
    "Chinese philosophy": ("Eastern", "Various"),
    "Indian philosophy": ("Eastern", "Various"),
    "Islamic philosophy": ("Eastern", "Medieval"),
    "Japanese philosophy": ("Eastern", "Various"),
    "Education": ("Western", "Contemporary"),
    "Religion": ("Western", "Contemporary"),
    "Science": ("Western", "Contemporary"),
    "Mathematics": ("Western", "Contemporary"),
    "Physics": ("Western", "Contemporary"),
    "Computer science": ("Western", "Contemporary"),
    "Neuroscience": ("Western", "Contemporary"),
    "Chemistry": ("Western", "Contemporary"),
    "Biology": ("Western", "Contemporary"),
    "Sociology": ("Western", "Contemporary"),
    "Psychology": ("Western", "Contemporary"),
    "Economics": ("Western", "Contemporary"),
    "Art": ("Western", "Contemporary"),
    "Music": ("Western", "Contemporary"),
    "Literatue": ("Western", "Contemporary"),
    "Language": ("Western", "Contemporary"),
    "History": ("Western", "Contemporary"),
    "Medicine": ("Western", "Contemporary"),
    "Law": ("Western", "Contemporary"),
    "Politics": ("Western", "Contemporary"),
}


def fetch_readme() -> str:
    with urllib.request.urlopen(README_URL) as r:
        return r.read().decode("utf-8")


def parse_entry(line: str) -> tuple[str, list[str]] | None:
    """Extract (author, [works]) from a list entry line."""
    if not line.startswith("* "):
        return None

    content = line[2:].strip()

    # Anonymous texts like "The Upanishads", URLs, bracket-links
    if content.startswith('"') or content.startswith("[") or content.startswith("http"):
        return None

    # Name = everything before the first quoted work
    quote_pos = content.find('"')
    raw_name = content[:quote_pos].strip().rstrip(",") if quote_pos != -1 else content.strip().rstrip(",")
    works = re.findall(r'"([^"]+)"', content)

    return raw_name, works


def normalize_name(name: str) -> str:
    """Collapse known formatting variants to a canonical name."""
    replacements = {
        "D. M. Armstrong": "D.M. Armstrong",
        "Daniel Dennett ": "Daniel Dennett",
        "Daniel C. Dennett": "Daniel Dennett",
        "David Kellogg Lewis": "David K. Lewis",
        "B. F. Skinner": "B.F. Skinner",
        "Willard Van Orman Quine": "Willard van Orman Quine",
        "Saul Kripke,": "Saul Kripke",
        "J. L. Austin,": "J.L. Austin",
        "Jean-Paul Sartre,": "Jean-Paul Sartre",
        "Kurt Gödel,": "Kurt Gödel",
        "Ruth Garrett Millikan": "Ruth Millikan",
        "Erwin Schrödinger, What is Life? The Physical Aspect of the Living Cell\"": "Erwin Schrödinger",
        "J. L. Austin": "J.L. Austin",
        "J. L. Mackie": "J.L. Mackie",
        "J. L. Schellenberg": "J.L. Schellenberg",
        "D.M. Armstrong": "D.M. Armstrong",
        "G. E. Moore": "G.E. Moore",
        "G. E. M. Anscombe": "G.E.M. Anscombe",
        "P. F. Strawson": "P.F. Strawson",
        "W. D. Ross": "W.D. Ross",
        "H. P. Grice": "H.P. Grice",
        "A. J. Ayer": "A.J. Ayer",
        "H.L.A. Hart": "H.L.A. Hart",
        "B.F. Skinner": "B.F. Skinner",
        "R.G. Collingwood": "R.G. Collingwood",
    }
    # Malformed README entries: "Name, Work Title Without Quotes..."
    if name not in replacements and len(name) > 50 and "," in name:
        name = name.split(",")[0].strip()
    return replacements.get(name, name)


def split_co_authors(name: str) -> list[str]:
    """'Horkheimer and Adorno' → ['Horkheimer', 'Adorno']"""
    if " and " in name:
        return [n.strip() for n in name.split(" and ")]
    # "Deleuze and Guattari" already handled by " and "
    # Multi-author with commas: "James Ladyman, Don Ross, ..."
    if name.count(",") >= 2:
        return [n.strip() for n in name.split(",")]
    return [name]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def parse(readme: str) -> list[dict]:
    philosophers: dict[str, dict] = {}
    current_section = ""

    for line in readme.splitlines():
        # Track current section heading
        heading_match = re.match(r"^#{2,4} (.+)", line)
        if heading_match:
            current_section = heading_match.group(1).strip()
            continue

        if not line.startswith("* "):
            continue

        result = parse_entry(line)
        if not result:
            continue

        raw_name, works = result
        tradition, era = SECTION_MAP.get(current_section, ("Western", "Unknown"))
        domain = current_section

        for name in split_co_authors(raw_name):
            name = normalize_name(name)
            if len(name) < 3 or name[0].islower():
                continue

            uid = slug(name)
            if uid not in philosophers:
                philosophers[uid] = {
                    "id": uid,
                    "name": name,
                    "tradition": tradition,
                    "eras": [era],
                    "domains": [domain] if domain else [],
                    "works": works,
                }
            else:
                # Philosopher appears in multiple sections — merge
                entry = philosophers[uid]
                if era not in entry["eras"]:
                    entry["eras"].append(era)
                if domain and domain not in entry["domains"]:
                    entry["domains"].append(domain)
                for w in works:
                    if w not in entry["works"]:
                        entry["works"].append(w)

    return sorted(philosophers.values(), key=lambda p: p["name"])


def main():
    print("Fetching README...", flush=True)
    readme = fetch_readme()

    philosophers = parse(readme)
    print(f"Parsed {len(philosophers)} unique philosophers.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(philosophers, f, indent=2, ensure_ascii=False)

    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
