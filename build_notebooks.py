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

import hashlib
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


SETUP_CELL = """# Setup. Nothing here is part of the lecture -- it just makes `mlschool`
# importable (cloning the course repo if we are on Colab) and imports the usual
# suspects. Run it and move on.
REPO = "{repo_url}"
import os, subprocess, sys
try:
    import mlschool
except ModuleNotFoundError:
    here = [os.path.abspath(d) for d in (".", "..", "../..")]
    root = next((d for d in here
                 if os.path.isfile(os.path.join(d, "mlschool", "__init__.py"))), None)
    if root is None:                                   # not inside a checkout: fetch it
        subprocess.run(["git", "clone", "--depth", "1", REPO, "{repo_dir}"], check=True)
        root = os.path.abspath("{repo_dir}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", root])
    sys.path.insert(0, root)

import mlschool as ms
import time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
DEVICE = ms.device()
ms.hello()"""


def _setup(extra=""):
    """The clone-and-install cell, plus anything written under the marker.

    Lines following `#%% setup` are appended to the same cell. That is how a
    notebook keeps e.g. its slide-deck call fused with setup while REPO_URL
    stays templated in exactly one place.
    """
    src = SETUP_CELL.format(repo_url=REPO_URL, repo_dir=REPO_DIR)
    if extra.strip():
        src = src.rstrip("\n") + "\n\n" + extra.strip() + "\n"
    return _cell("code", src)


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
        body = "".join(buf)
        if kind == "setup":
            cells.append(_setup(body))            # body is appended to the cell
        elif kind is not None and body.strip():
            cells.append(_cell(kind, body))

    for raw in path.read_text().split("\n"):
        if raw.startswith("#%%"):
            flush()
            tag = raw[3:].strip()
            buf = []
            if tag == "setup":
                kind = "setup"                    # emitted by the next flush()
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


def digest(cells):
    """Whitespace-insensitive fingerprint of a notebook's cells."""
    body = "\n\x00".join(f"{c['cell_type']}:{''.join(c['source']).strip()}"
                         for c in cells if "".join(c["source"]).strip())
    return hashlib.sha1(body.encode()).hexdigest()


def hand_edited(dst):
    """True if the notebook has been edited since the build that wrote it.

    Each build stamps the digest of what it wrote into the notebook metadata.
    If the file's current digest still matches that stamp, nobody has touched
    it and it is safe to overwrite; if it differs, the difference is somebody's
    hand edit and rebuilding would throw it away.

    An unstamped notebook predates this scheme, so we cannot tell -- treat it
    as edited and make the caller decide.
    """
    try:
        nb = json.loads(dst.read_text())
    except Exception:
        return False                              # unreadable -> just rebuild
    stamp = nb.get("metadata", {}).get("mlschool_build")
    return stamp != digest(nb["cells"])


def build(path, out_dir, force=False):
    cells = parse(path)
    for i, c in enumerate(cells):
        c["id"] = f"cell{i:03d}"          # required by nbformat >= 4.5
    nb = {"cells": cells, "metadata": dict(KERNEL, mlschool_build=digest(cells)),
          "nbformat": 4, "nbformat_minor": 5}
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / (path.stem + ".ipynb")

    # Refuse to clobber hand edits. The notebook is a build artefact, but it is
    # also the thing you actually open in Jupyter -- so if it has been touched
    # since the build that wrote it, rebuilding would silently discard that
    # work. Sync the other way first (`sync_from_notebooks.py`, src <- notebook).
    if dst.exists() and not force and hand_edited(dst):
        print(f"  SKIPPED {dst.relative_to(ROOT)} -- edited since it was built; "
              f"run `python sync_from_notebooks.py {path.stem[:2]}` first, "
              f"or rebuild with --force to discard those edits")
        return False

    dst.write_text(json.dumps(nb, indent=1) + "\n")
    n_md = sum(c["cell_type"] == "markdown" for c in nb["cells"])
    n_code = len(nb["cells"]) - n_md
    print(f"  {dst.relative_to(ROOT)}  ({n_md} md + {n_code} code cells)")


if __name__ == "__main__":
    force = "--force" in sys.argv
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")]
    # (source directory, output directory) pairs
    trees = [(SRC, OUT), (SRC / "solutions", OUT / "solutions")]
    files = [(f, out) for src, out in trees for f in sorted(src.glob("*.nbsrc"))]
    if wanted:
        files = [(f, o) for f, o in files if any(w in f.stem for w in wanted)]
    if not files:
        sys.exit("no matching .nbsrc files")
    print("building:")
    skipped = sum(build(f, out, force) is False for f, out in files)
    if skipped:
        print(f"\n{skipped} notebook(s) skipped to protect hand edits.")
