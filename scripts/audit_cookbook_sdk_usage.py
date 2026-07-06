#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COOKBOOKS_DIR = ROOT / "cookbooks"
CALL_RE = re.compile(r"\bjinko\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def iter_code_sources(notebook_path: Path) -> list[str]:
    notebook = json.loads(notebook_path.read_text())
    sources: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, list):
            sources.append("".join(source))
        else:
            sources.append(str(source))
    return sources


def main() -> int:
    usage: dict[str, set[str]] = defaultdict(set)
    helper_notebooks: set[str] = set()

    for notebook_path in sorted(COOKBOOKS_DIR.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in notebook_path.parts:
            continue
        notebook_key = str(notebook_path.relative_to(ROOT))
        for source in iter_code_sources(notebook_path):
            if "import jinko_helpers as jinko" in source:
                helper_notebooks.add(notebook_key)
            for match in CALL_RE.finditer(source):
                usage[match.group(1)].add(notebook_key)

    print("Legacy helper import notebooks:")
    for notebook_key in sorted(helper_notebooks):
        print(f"- {notebook_key}")

    print("\nLegacy helper call inventory:")
    for helper_name in sorted(usage):
        print(f"\n{helper_name}")
        for notebook_key in sorted(usage[helper_name]):
            print(f"- {notebook_key}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
