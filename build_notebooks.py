#!/usr/bin/env python
"""
Build the Colab notebooks from the `src/*.nbsrc` sources.

Why a build step? Every notebook must be standalone: a Colab user opens exactly
one notebook, in a fresh VM, with no repository checked out. But the dataset
generator should have a single source of truth (`mlschool_data.py`) rather than
six drifting copies. So the sources carry a marker

    #%% inject mlschool_data

which is expanded at build time into a `%%writefile mlschool_data.py` cell
holding the current module source.

Source format (`src/NAME.nbsrc`), markers must start at column 0:

    #%% md          -> markdown cell, raw markdown until the next marker
    #%% code        -> code cell
    #%% setup       -> the standard clone-and-install cell (see REPO_URL below)
    #%% inject X    -> code cell that writes module X.py to disk (legacy; used
                       only by notebooks that must run with no repository)

Usage:  python build_notebooks.py [name ...]
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# SET THIS before distributing the notebooks. It is the only place the URL
# appears; every notebook's setup cell is generated from it.
# NOTE: must be the HTTPS URL, not git@github.com:... -- Colab has no SSH key.
REPO_URL = "https://github.com/drinkingkazu/a3net-lecture2.git"
REPO_DIR = "a3net-lecture2"
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
OUT = ROOT / "notebooks"

KERNEL = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
    "colab": {"provenance": [], "toc_visible": True},
    "accelerator": "GPU",
}


def _cell(kind, source):
    text = source.strip("\n")
    lines = [l + "\n" for l in text.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": lines}


SETUP_CELL = """# --- setup: make the course package importable (run once per session) ------
# On Colab: clones the repository and installs it.
# Locally inside a checkout: finds it and uses it in place -- no second copy.
# Already importable: does nothing. Safe to re-run either way.
REPO_URL = "{repo_url}"
REPO_DIR = "{repo_dir}"

import os, subprocess, sys


def _find_checkout(start=None):
    \"\"\"Walk up from `start` looking for a directory containing mlschool/.\"\"\"
    d = os.path.abspath(start or os.getcwd())
    while True:
        if os.path.isfile(os.path.join(d, "mlschool", "__init__.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


try:
    import mlschool                       # already installed, or already on sys.path
except ModuleNotFoundError:
    root = _find_checkout()               # are we sitting inside the repo already?
    if root is None:                      # no -- fetch it (this is the Colab path)
        if not os.path.isdir(REPO_DIR):
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR],
                           check=True)
        root = os.path.abspath(REPO_DIR)
        try:                              # nice-to-have; sys.path below is enough
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", root],
                           check=True)
        except subprocess.CalledProcessError:
            print("pip install failed; falling back to sys.path (usually fine)")
    sys.path.insert(0, root)
    import mlschool

import mlschool as ms
print("mlschool", ms.__version__, "from", os.path.dirname(ms.__file__))
print("device:", ms.device())"""


def _setup():
    return _cell("code", SETUP_CELL.format(repo_url=REPO_URL, repo_dir=REPO_DIR))


def _inject(module_name):
    """Expand `#%% inject X` into a %%writefile cell holding module X's source.

    X may be a bare module name (`mlschool_data`, read from the repo root) or a
    path (`mlschool/slides.py`), in which case the file is written to the
    notebook's working directory under its basename.
    """
    if module_name.endswith(".py") or "/" in module_name:
        path = ROOT / module_name
        module_name = path.stem
        src = path.read_text()
        return _cell("code", f"%%writefile {module_name}.py\n"
                             f"# Written to disk by the notebook so it is "
                             f"standalone on Colab.\n" + src)
    src = (ROOT / f"{module_name}.py").read_text()
    # %%writefile must be the very first line of the cell.
    body = (
        f"%%writefile {module_name}.py\n"
        f"# Shared dataset module for the ML summer school (lecture 2).\n"
        f"# Written to disk by the notebook so the notebook is standalone on Colab.\n"
        + src
    )
    return _cell("code", body)


def parse(path):
    cells, kind, buf = [], None, []

    def flush():
        if kind is not None and "".join(buf).strip():
            cells.append(_cell(kind, "".join(buf)))

    for raw in path.read_text().split("\n"):
        if raw.startswith("#%%"):
            flush()
            tag = raw[3:].strip()
            buf = []
            if tag == "setup":
                cells.append(_setup())
                kind = None
            elif tag.startswith("inject"):
                cells.append(_inject(tag.split()[1]))
                kind = None
            elif tag == "md":
                kind = "markdown"
            elif tag == "code":
                kind = "code"
            else:
                raise ValueError(f"{path.name}: unknown marker {raw!r}")
        else:
            if kind is not None:
                buf.append(raw + "\n")
    flush()
    return cells


def build(path, out_dir):
    cells = parse(path)
    for i, c in enumerate(cells):
        c["id"] = f"cell{i:03d}"          # required by nbformat >= 4.5
    nb = {"cells": cells, "metadata": KERNEL,
          "nbformat": 4, "nbformat_minor": 5}
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / (path.stem + ".ipynb")
    dst.write_text(json.dumps(nb, indent=1) + "\n")
    n_md = sum(c["cell_type"] == "markdown" for c in nb["cells"])
    n_code = len(nb["cells"]) - n_md
    print(f"  {dst.relative_to(ROOT)}  ({n_md} md + {n_code} code cells)")


if __name__ == "__main__":
    wanted = sys.argv[1:]
    # (source directory, output directory) pairs
    trees = [(SRC, OUT), (SRC / "solutions", OUT / "solutions")]
    files = [(f, out) for src, out in trees for f in sorted(src.glob("*.nbsrc"))]
    if wanted:
        files = [(f, o) for f, o in files if any(w in f.stem for w in wanted)]
    if not files:
        sys.exit("no matching .nbsrc files")
    print("building:")
    for f, out in files:
        build(f, out)
