#!/usr/bin/env python
"""
Pull edits made in `notebooks/*.ipynb` back into `src/*.nbsrc`.

The build goes src -> notebooks, so editing a notebook directly is normally a
way to lose work: the next `build_notebooks.py` overwrites it. This script is
the other direction, so you can edit in Jupyter (or Colab) and keep the sources
authoritative afterwards.

    python sync_from_notebooks.py            # sync every notebook
    python sync_from_notebooks.py 02         # just NB2
    python sync_from_notebooks.py --check    # report differences, change nothing

The generated setup cell is recognised and written back as the `#%% setup`
marker, so REPO_URL stays templated in one place. Empty cells are dropped.

Round-trip is verified after each write: the script rebuilds from the new
source and compares against the notebook it read, cell for cell.
"""

import json
import sys
from pathlib import Path

import build_notebooks as bn

ROOT = Path(__file__).resolve().parent
SETUP_MARKER = "# Setup. Nothing here is part of the lecture"


def notebook_for(src_path):
    sub = "solutions" if src_path.parent.name == "solutions" else ""
    return ROOT / "notebooks" / sub / f"{src_path.stem}.ipynb"


def to_nbsrc(cells):
    """Serialise notebook cells back into the .nbsrc text format."""
    out = []
    for c in cells:
        text = "".join(c["source"]).rstrip("\n")
        if not text.strip():
            continue                                  # drop empty scratch cells
        if c["cell_type"] == "code" and text.lstrip().startswith(SETUP_MARKER):
            # The generated block ends at `ms.hello()`. Anything a human has
            # appended below it is THEIR code and must be kept -- split rather
            # than swallow it.
            head, sep, tail = text.partition("ms.hello()")
            out.append("#%% setup\n")                 # keep REPO_URL templated
            if sep and tail.strip():
                # lines appended below the generated block stay in the SAME
                # cell -- the build re-appends whatever follows the marker
                out.append(tail.strip() + "\n\n")
            elif not sep:
                print(f"    note: setup cell not recognised, kept verbatim")
                out.pop()
                out.append("#%% code\n" + text + "\n\n")
            continue
        out.append(f"#%% {'md' if c['cell_type'] == 'markdown' else 'code'}\n")
        out.append(text + "\n\n")
    return "".join(out).rstrip("\n") + "\n"


def cell_texts(cells):
    return [(c["cell_type"], "".join(c["source"]).strip()) for c in cells]


def sync(src_path, check_only=False):
    nb_path = notebook_for(src_path)
    if not nb_path.exists():
        return None
    nb_cells = json.load(open(nb_path))["cells"]
    old = cell_texts(bn.parse(src_path))
    new = cell_texts([c for c in nb_cells if "".join(c["source"]).strip()])

    if old == new:
        return ("unchanged", 0)

    n_diff = sum(1 for a, b in zip(old, new) if a != b) + abs(len(old) - len(new))
    if check_only:
        return ("WOULD SYNC", n_diff)

    src_path.write_text(to_nbsrc(nb_cells))

    # verify: rebuilding from the new source must reproduce the notebook
    rebuilt = cell_texts(bn.parse(src_path))
    return ("synced" if rebuilt == new else "SYNCED BUT ROUND-TRIP MISMATCH", n_diff)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv
    srcs = sorted((ROOT / "src").glob("*.nbsrc")) + \
        sorted((ROOT / "src" / "solutions").glob("*.nbsrc"))
    if args:
        srcs = [s for s in srcs if any(a in s.stem for a in args)]

    print("checking:" if check else "syncing notebooks -> src:")
    bad = 0
    for s in srcs:
        result = sync(s, check)
        if result is None:
            continue
        status, n = result
        if "MISMATCH" in status:
            bad += 1
        note = f"  ({n} cells)" if n else ""
        print(f"  {s.stem:<42} {status}{note}")
    if bad:
        sys.exit(f"\n{bad} file(s) failed round-trip verification")
