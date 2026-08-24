"""
mlschool.diagrams — schematic figures for the lecture text.

These are explanatory drawings, not results. They are matplotlib rather than
images so that they stay version-controlled, restyle consistently, and can be
edited by the lecturer.

Every function returns a matplotlib Figure.
"""

import numpy as np

INK = "#101418"
BLUE = "#4cc9f0"
AMBER = "#f7b32b"
GREEN = "#3ddc97"
RED = "#ef476f"
VIOLET = "#8367c7"
GREY = "#9aa5b1"
PALE = "#dfe6ec"


def _canvas(w=10, h=4, xlim=(0, 10), ylim=(0, 4), n=1):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, n, figsize=(w, h))
    for ax in np.atleast_1d(axes):
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal"); ax.axis("off")
    return fig, axes


def _box(ax, x, y, w, h, color, label="", fs=8, alpha=1.0, tc="white", lw=0.8):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                facecolor=color, edgecolor=INK, lw=lw, alpha=alpha))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=tc, weight="bold")


def _arrow(ax, p0, p1, color=INK, lw=1.4, style="-|>", alpha=1.0, rad=0.0):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11,
                                 color=color, lw=lw, alpha=alpha,
                                 connectionstyle=f"arc3,rad={rad}"))


def _grid(ax, x0, y0, n, cell, active=None, face=PALE, on=BLUE, lw=0.5):
    """n x n grid of cells; `active` is a set of (col,row) that are filled."""
    from matplotlib.patches import Rectangle
    active = active or set()
    for r in range(n):
        for c in range(n):
            f = on if (c, r) in active else face
            ax.add_patch(Rectangle((x0 + c * cell, y0 + r * cell), cell, cell,
                                   facecolor=f, edgecolor=GREY, lw=lw))


# --------------------------------------------------------------------------
# NB0
# --------------------------------------------------------------------------
def weight_sharing():
    """MLP connects everything to everything; a CNN reuses one small kernel."""
    fig, axes = _canvas(11, 4.6, (0, 10), (0, 9), n=2)

    for ax, title in zip(axes, ["MLP: one weight per (pixel, unit) pair",
                                "CNN: one small kernel, reused at every position"]):
        _grid(ax, 0.4, 0.6, 6, 0.75)
        ax.text(2.65, 0.1, "input image", ha="center", fontsize=8, color=INK)
        ax.set_title(title, fontsize=9.5)

    # --- MLP: every pixel to every hidden unit
    ax = axes[0]
    units = [(7.6, 2.2), (7.6, 4.2), (7.6, 6.2)]
    for (ux, uy) in units:
        for r in range(6):
            for c in range(6):
                ax.plot([0.4 + (c + .5) * .75, ux - .35],
                        [0.6 + (r + .5) * .75, uy], color=RED, lw=0.16, alpha=0.30)
    for (ux, uy) in units:
        ax.add_patch(__import__("matplotlib").patches.Circle(
            (ux, uy), 0.35, facecolor=RED, edgecolor=INK, lw=0.8))
    ax.text(7.6, 7.6, "hidden units", ha="center", fontsize=8, color=INK)
    ax.text(5.0, 8.5, "36 x 3 = 108 weights\nfor a 6x6 image",
            ha="center", fontsize=8.5, color=RED, weight="bold")

    # --- CNN: one kernel at two positions
    ax = axes[1]
    from matplotlib.patches import Rectangle, Circle
    for (c0, r0), (ux, uy) in [((0, 3), (7.6, 6.2)), ((3, 0), (7.6, 2.2))]:
        ax.add_patch(Rectangle((0.4 + c0 * .75, 0.6 + r0 * .75), 2.25, 2.25,
                               facecolor="none", edgecolor=BLUE, lw=2.2))
        for r in range(r0, r0 + 3):
            for c in range(c0, c0 + 3):
                ax.plot([0.4 + (c + .5) * .75, ux - .35],
                        [0.6 + (r + .5) * .75, uy], color=BLUE, lw=0.5, alpha=0.75)
        ax.add_patch(Circle((ux, uy), 0.35, facecolor=BLUE, edgecolor=INK, lw=0.8))
    ax.text(7.6, 7.6, "feature map", ha="center", fontsize=8, color=INK)
    ax.text(5.0, 8.5, "3 x 3 = 9 weights,\nsame ones at every position",
            ha="center", fontsize=8.5, color=BLUE, weight="bold")
    ax.text(5.0, 4.2, "same\nkernel", ha="center", va="center", fontsize=8,
            color=BLUE, style="italic")
    fig.tight_layout()
    return fig


def representations():
    """The same sparse event held three different ways."""
    fig, axes = _canvas(12, 4.2, (0, 10), (0, 10), n=3)
    hits = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (6, 7)]

    ax = axes[0]
    _grid(ax, 0.5, 1.6, 9, 0.85, active=set(hits))
    ax.set_title("1. dense array", fontsize=9.5)
    ax.text(4.3, 0.7, "81 numbers stored,\n6 of them non-zero",
            ha="center", fontsize=8.5, color=INK)

    ax = axes[1]
    _box(ax, 2.0, 8.3, 6.0, 1.0, INK, "x    y    charge", fs=9)
    for k, (c, r) in enumerate(hits):
        y = 7.1 - k * 1.05
        _box(ax, 2.0, y, 6.0, 0.95, PALE, f"{c}    {r}    {12 + 7 * k}",
             fs=8.5, tc=INK)
    ax.set_title("2. sparse: coordinates + values", fontsize=9.5)
    ax.text(5.0, 0.15, "18 numbers stored", ha="center", fontsize=8.5, color=INK)

    ax = axes[2]
    xs = np.array([h[0] for h in hits]) * 0.95 + 1.2
    ys = np.array([h[1] for h in hits]) * 0.95 + 1.2
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            d = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
            if d < 1.8:
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color=BLUE, lw=1.1, zorder=1)
    ax.scatter(xs, ys, s=90, c=AMBER, edgecolors=INK, zorder=2)
    ax.set_title("3. point cloud + graph", fontsize=9.5)
    ax.text(5.0, 0.15, "same 18 numbers,\nplus edges you choose",
            ha="center", fontsize=8.5, color=INK)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# NB1
# --------------------------------------------------------------------------
def receptive_field(layers=(("conv 3x3", 3), ("conv 3x3", 5), ("pool /2", 6),
                            ("conv 3x3", 10), ("conv 3x3", 14), ("pool /2", 16))):
    """How far back into the input one deep unit can see."""
    from matplotlib.patches import Rectangle
    n_in = 21
    fig, ax = _canvas(9.6, 5.4, (-1, n_in + 5), (-1.6, len(layers) * 1.35 + 2.6), n=1)
    cell = 1.0
    centre = n_in / 2

    for c in range(n_in):                                   # the input row
        ax.add_patch(Rectangle((c * cell, 0), cell, 0.8,
                               facecolor=PALE, edgecolor=GREY, lw=0.5))
    ax.text(centre, -0.9, "input pixels", ha="center", fontsize=9)

    for i, (name, rf) in enumerate(layers):
        y = 1.35 + i * 1.35
        x0 = centre - rf / 2
        shade = 0.20 + 0.13 * i
        ax.add_patch(Rectangle((x0, y), rf * cell, 0.85, facecolor=BLUE,
                               alpha=min(shade, 0.95), edgecolor=INK, lw=1.0))
        ax.plot([x0, x0], [0.85, y], color=BLUE, lw=0.7, ls=":", alpha=0.7)
        ax.plot([x0 + rf, x0 + rf], [0.85, y], color=BLUE, lw=0.7, ls=":", alpha=0.7)
        ax.text(-0.6, y + 0.42, name, fontsize=8.5, va="center", ha="right")
        ax.text(n_in + 0.5, y + 0.42, f"{rf} px", fontsize=9, va="center",
                color="#1f6f8f", weight="bold")

    top = 1.35 + len(layers) * 1.35
    _box(ax, centre - 0.9, top, 1.8, 0.8, AMBER, "unit", fs=8, tc=INK)
    _arrow(ax, (centre, top), (centre, top - 0.45), color=GREY)
    ax.text(centre, -1.15, "one unit at the top is influenced by this many "
                           "input pixels", ha="center", fontsize=9, style="italic")
    ax.set_title("receptive field: convolutions widen it slowly, "
                 "pooling widens it fast", fontsize=10, pad=14)
    fig.tight_layout()
    return fig


def encoder_heads():
    """One shared encoder, a decoder for per-pixel output, three task heads."""
    fig, ax = _canvas(12, 5.9, (0, 24), (-0.6, 11.6), n=1)
    enc = [("1x96x96", 0.5, 8.6, BLUE), ("16x48x48", 2.9, 7.2, BLUE),
           ("32x24x24", 5.3, 5.8, BLUE), ("96x12x12", 7.7, 4.4, VIOLET)]
    for name, x, y, col in enc:
        _box(ax, x, y, 2.1, 1.4, col, name, fs=7.5)
    for i in range(3):
        _arrow(ax, (enc[i][1] + 1.05, enc[i][2]),
               (enc[i + 1][1] + 1.05, enc[i + 1][2] + 1.4), color=GREY)
    ax.text(0.5, 10.6, "ENCODER   down: context", fontsize=9, color="#1f6f8f",
            weight="bold")

    dec = [("32x24x24", 12.6, 5.8), ("16x48x48", 15.0, 7.2), ("3x96x96", 17.4, 8.6)]
    for name, x, y in dec:
        _box(ax, x, y, 2.1, 1.4, GREEN, name, fs=7.5, tc=INK)
    _arrow(ax, (9.8, 5.1), (12.6, 6.1), color=GREY)
    for i in range(2):
        _arrow(ax, (dec[i][1] + 2.1, dec[i][2] + 0.7),
               (dec[i + 1][1], dec[i + 1][2] + 0.7), color=GREY)
    ax.text(14.4, 10.6, "DECODER   up: resolution", fontsize=9, color="#1f9d6a",
            weight="bold")

    for src, dst in [(1, 0), (2, 1)]:
        _arrow(ax, (enc[src + 1][1] + 2.1, enc[src + 1][2] + 0.7),
               (dec[dst][1], dec[dst][2] + 0.7), color=AMBER, lw=1.7, rad=-0.32)
    ax.text(11.9, 11.1, "U-Net skips: concatenate, to recover spatial detail",
            fontsize=8.5, color="#b07d12", ha="center", weight="bold")

    for name, x in [("classification\nglobal MAX pool", 1.0),
                    ("energy\nglobal SUM pool", 5.6)]:
        _box(ax, x, 0.9, 4.0, 1.5, AMBER, name, fs=7.5, tc=INK)
        _arrow(ax, (8.75, 4.4), (x + 2.0, 2.4), color=GREY, rad=0.12)
    _box(ax, 17.2, 5.6, 4.4, 1.5, AMBER, "segmentation\nno pooling", fs=7.5, tc=INK)
    _arrow(ax, (19.4, 8.6), (19.4, 7.1), color=GREY)

    ax.text(12.0, -0.35, "three heads; each pooling chosen to match the physics "
                         "of its target", ha="center", fontsize=8.5, style="italic")
    fig.tight_layout()
    return fig


def skip_connection_types():
    """Two unrelated mechanisms share the name 'skip connection'."""
    fig, axes = _canvas(11, 3.9, (0, 12), (0, 7), n=2)

    ax = axes[0]
    _box(ax, 1.0, 3.0, 1.6, 1.4, BLUE, "x", fs=10)
    _box(ax, 4.2, 3.0, 2.2, 1.4, VIOLET, "f(x)", fs=9)
    from matplotlib.patches import Circle
    ax.add_patch(Circle((8.3, 3.7), 0.55, facecolor=AMBER, edgecolor=INK))
    ax.text(8.3, 3.7, "+", ha="center", va="center", fontsize=14, weight="bold")
    _box(ax, 9.8, 3.0, 1.6, 1.4, GREEN, "y", fs=10, tc=INK)
    _arrow(ax, (2.6, 3.7), (4.2, 3.7)); _arrow(ax, (6.4, 3.7), (7.75, 3.7))
    _arrow(ax, (9.0, 3.7), (9.8, 3.7))
    _arrow(ax, (1.8, 4.4), (8.3, 5.9), color=AMBER, lw=2.0, rad=-0.25)
    _arrow(ax, (8.3, 5.9), (8.3, 4.3), color=AMBER, lw=2.0)
    ax.text(6.0, 6.4, "identity bypass", fontsize=8.5, color="#b07d12",
            ha="center", weight="bold")
    ax.text(6.0, 1.5, r"$y = x + f(x)$   same resolution", ha="center", fontsize=9)
    ax.text(6.0, 0.6, "job: keep GRADIENTS alive (ResNet)", ha="center",
            fontsize=8.5, style="italic", color=INK)
    ax.set_title("residual skip  —  addition", fontsize=10)

    ax = axes[1]
    _box(ax, 0.8, 4.4, 2.4, 1.3, BLUE, "encoder\n48x48", fs=7.5)
    _box(ax, 0.8, 1.4, 2.4, 1.3, BLUE, "encoder\n24x24", fs=7.5)
    _box(ax, 5.2, 1.4, 2.4, 1.3, GREEN, "upsampled\n48x48", fs=7.5, tc=INK)
    _box(ax, 9.0, 3.0, 2.4, 1.3, VIOLET, "concat\n(2C channels)", fs=7.5)
    _arrow(ax, (2.0, 4.4), (2.0, 2.7), color=GREY)
    _arrow(ax, (3.2, 2.05), (5.2, 2.05), color=GREY)
    _arrow(ax, (7.6, 2.05), (9.0, 3.2), color=GREY)
    _arrow(ax, (3.2, 5.05), (9.0, 4.1), color=AMBER, lw=2.0, rad=-0.15)
    ax.text(6.2, 5.9, "carries the fine detail\ndownsampling threw away",
            fontsize=8.5, color="#b07d12", ha="center", weight="bold")
    ax.text(6.0, 0.5, "job: recover RESOLUTION (U-Net)", ha="center",
            fontsize=8.5, style="italic", color=INK)
    ax.set_title("U-Net skip  —  concatenation across resolutions", fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# NB2
# --------------------------------------------------------------------------
def gradient_flow():
    """Why depth needs residual connections and normalisation."""
    fig, axes = _canvas(10.5, 4.6, (0, 10), (0, 12), n=2)
    n = 7
    for ax, residual in zip(axes, [False, True]):
        for i in range(n):
            y = 0.8 + i * 1.5
            _box(ax, 3.2, y, 3.4, 1.0, BLUE if not residual else VIOLET,
                 f"layer {n - i}", fs=8)
        for i in range(n - 1):
            y = 0.8 + i * 1.5
            g = 0.55 ** (n - 1 - i) if not residual else 1.0
            _arrow(ax, (4.9, y + 1.5), (4.9, y + 1.05), color=RED,
                   lw=0.4 + 4.0 * g, alpha=0.35 + 0.65 * g)
        if residual:
            _arrow(ax, (7.4, 10.9), (7.4, 0.9), color=AMBER, lw=3.2)
            ax.text(8.7, 6.0, "gradient\nhighway", fontsize=8.5, color="#b07d12",
                    ha="center", va="center", weight="bold", rotation=90)
        else:
            for i, lab in [(5, r"$\times 0.55$"), (2, r"$\times 0.55$")]:
                ax.text(2.9, 1.3 + i * 1.5, lab, fontsize=8, color=RED, ha="right")
        ax.text(4.9, 11.6, "loss gradient enters here", fontsize=8.5, ha="center")
        bottom = "reaches layer 1 at ~$10^{-10}$" if not residual else \
                 "reaches layer 1 intact"
        ax.text(4.9, 0.1, bottom, fontsize=8.5, ha="center",
                color=RED if not residual else "#1f9d6a", weight="bold")
        ax.set_title("plain deep network" if not residual else
                     "+ residual connections & normalisation", fontsize=10)
    fig.tight_layout()
    return fig


def diagnosis_tree():
    """The decision procedure for a model that will not train."""
    fig, ax = _canvas(12, 5.4, (0, 24), (0, 11), n=1)
    _box(ax, 8.6, 9.2, 6.8, 1.3, INK, "training loss is not falling", fs=9)
    q1 = "is it pinned at ln(n classes)?"
    _box(ax, 8.0, 6.9, 8.0, 1.2, VIOLET, q1, fs=8.5)
    _arrow(ax, (12.0, 9.2), (12.0, 8.1))

    _box(ax, 0.6, 4.4, 7.2, 1.2, BLUE, "model has collapsed", fs=8.5)
    _box(ax, 16.2, 4.4, 7.2, 1.2, BLUE, "it is learning, just slowly", fs=8.5)
    _arrow(ax, (10.0, 6.9), (4.2, 5.6), rad=0.15); ax.text(6.0, 6.3, "yes", fontsize=8)
    _arrow(ax, (14.0, 6.9), (19.8, 5.6), rad=-0.15); ax.text(17.6, 6.3, "no", fontsize=8)

    leaves = [
        (0.6, 1.6, "update ratio > $10^{-2}$\n-> LR too HIGH", RED),
        (8.4, 1.6, "gradients vanish\nwith depth\n-> add norm + residual", AMBER),
        (16.2, 1.6, "update ratio < $10^{-5}$\n-> LR too LOW", RED),
    ]
    for x, y, txt, col in leaves:
        _box(ax, x, y, 7.2, 2.0, col, txt, fs=8, tc=INK)
    _arrow(ax, (3.4, 4.4), (3.4, 3.6)); _arrow(ax, (5.6, 4.4), (11.0, 3.6), rad=-0.1)
    _arrow(ax, (19.8, 4.4), (19.8, 3.6))
    ax.text(12.0, 0.6, "and always: is the data shuffled?  can the receptive field "
                       "see the evidence?", ha="center", fontsize=8.5, style="italic")
    ax.set_title("triage: the training loss is flat", fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# NB3
# --------------------------------------------------------------------------
def batching_strategies():
    """Two ways to put variable-length events into one tensor."""
    fig, axes = _canvas(11.5, 4.4, (0, 16), (0, 9), n=2)
    sizes = [6, 3, 8, 4]
    colors = [BLUE, AMBER, GREEN, VIOLET]
    N = max(sizes)

    ax = axes[0]
    from matplotlib.patches import Rectangle
    for e, (m, col) in enumerate(zip(sizes, colors)):
        y = 6.6 - e * 1.5
        for k in range(N):
            filled = k < m
            ax.add_patch(Rectangle((2.0 + k * 1.4, y), 1.3, 1.1,
                                   facecolor=col if filled else "white",
                                   edgecolor=GREY if filled else RED,
                                   hatch=None if filled else "///", lw=0.9))
        ax.text(1.7, y + 0.55, f"ev {e}", ha="right", va="center", fontsize=8)
    ax.text(2.0 + N * 1.4 / 2, 8.2, f"padded to N = {N}", ha="center", fontsize=9)
    ax.text(8.0, 0.4, "simple and rectangular;\nwasted slots, and you must "
                      "mask EVERYWHERE", ha="center", fontsize=8.5, color=RED)
    ax.set_title("A. pad + mask", fontsize=10)

    ax = axes[1]
    x = 1.2
    for e, (m, col) in enumerate(zip(sizes, colors)):
        for k in range(m):
            ax.add_patch(Rectangle((x, 4.4), 0.62, 1.2, facecolor=col,
                                   edgecolor=GREY, lw=0.7))
            ax.add_patch(Rectangle((x, 2.9), 0.62, 1.2, facecolor="white",
                                   edgecolor=GREY, lw=0.7))
            ax.text(x + 0.31, 3.5, str(e), ha="center", va="center", fontsize=7)
            x += 0.66
    ax.text(0.9, 5.0, "hits", ha="right", va="center", fontsize=8)
    ax.text(0.9, 3.5, "batch\nindex", ha="right", va="center", fontsize=8)
    ax.text(8.0, 6.6, "one long list, no gaps", ha="center", fontsize=9)
    ax.text(8.0, 0.9, "zero waste, no truncation;\nevery op becomes a scatter "
                      "over the index", ha="center", fontsize=8.5, color="#1f9d6a")
    ax.set_title("B. concatenate + batch index", fontsize=10)
    fig.tight_layout()
    return fig


def message_passing():
    """Fixed geometric neighbours vs. learned, content-dependent ones."""
    fig, axes = _canvas(11.5, 4.3, (0, 10), (0, 10), n=3)
    rng = np.random.default_rng(3)
    pts = rng.uniform(1.4, 8.6, (11, 2))
    i = 5
    d = np.hypot(*(pts - pts[i]).T)
    nbr = np.argsort(d)[1:5]

    ax = axes[0]
    for j in nbr:
        _arrow(ax, tuple(pts[j]), tuple(pts[i]), color=BLUE, lw=1.5)
    ax.scatter(*pts.T, s=70, c=PALE, edgecolors=INK, zorder=3)
    ax.scatter(*pts[i], s=130, c=AMBER, edgecolors=INK, zorder=4)
    ax.set_title("1. k-NN graph\nedges from geometry", fontsize=9.5)

    ax = axes[1]
    ax.text(5, 8.6, r"$m_{ij} = \mathrm{MLP}([h_i \,\|\, h_j - h_i])$",
            ha="center", fontsize=10)
    for k, j in enumerate(nbr):
        _box(ax, 0.6 + k * 2.3, 5.6, 2.0, 1.1, BLUE, f"$m_{{i{k}}}$", fs=9)
        _arrow(ax, (1.6 + k * 2.3, 5.5), (5.0, 4.4), color=GREY, rad=0.12)
    _box(ax, 3.0, 3.0, 4.0, 1.3, AMBER, "max / sum\n(symmetric)", fs=8.5, tc=INK)
    _arrow(ax, (5.0, 3.0), (5.0, 2.2))
    _box(ax, 3.4, 0.9, 3.2, 1.2, GREEN, r"new $h_i$", fs=9, tc=INK)
    ax.set_title("2. one round of message passing", fontsize=9.5)

    ax = axes[2]
    w = rng.random(len(pts)) ** 2
    for j in range(len(pts)):
        if j == i:
            continue
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                color=VIOLET, lw=0.3 + 3.4 * w[j], alpha=0.35 + 0.6 * w[j], zorder=1)
    ax.scatter(*pts.T, s=70, c=PALE, edgecolors=INK, zorder=3)
    ax.scatter(*pts[i], s=130, c=AMBER, edgecolors=INK, zorder=4)
    ax.set_title("3. attention\nall pairs, learned weights", fontsize=9.5)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# NB4
# --------------------------------------------------------------------------
def transfer_strategies():
    """Three ways to use a pretrained encoder on a small labelled sample."""
    fig, axes = _canvas(11.5, 3.9, (0, 12), (0, 7), n=3)
    specs = [("from scratch", RED, RED, "random", "random", "needs many labels"),
             ("linear probe", GREY, GREEN, "frozen", "trained", "very few labels"),
             ("fine-tune", VIOLET, GREEN, "trained, low LR", "trained",
              "usually the best")]
    for ax, (title, ecol, hcol, elab, hlab, note) in zip(axes, specs):
        _box(ax, 1.2, 3.4, 6.0, 2.0, ecol, f"ENCODER\n{elab}", fs=8.5)
        _box(ax, 8.0, 3.7, 3.0, 1.4, hcol, f"HEAD\n{hlab}", fs=8.5,
             tc=INK if hcol == GREEN else "white")
        _arrow(ax, (7.2, 4.4), (8.0, 4.4))
        if elab == "frozen":
            ax.text(4.2, 2.7, "❄", ha="center", fontsize=15, color="#2f6fb3")
        ax.text(6.0, 1.5, note, ha="center", fontsize=8.5, style="italic")
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# NB6
# --------------------------------------------------------------------------
def submanifold_convolution():
    """Standard sparse convolution dilates the active set; submanifold does not."""
    fig, axes = _canvas(11.5, 4.3, (0, 9), (0, 10), n=3)
    n, cell = 9, 0.9
    active = {(3, 4), (4, 4), (5, 4), (5, 5), (6, 5)}
    grown = set(active)
    for (c, r) in active:
        for dc in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if 0 <= c + dc < n and 0 <= r + dr < n:
                    grown.add((c + dc, r + dr))

    for ax, (act, title, sub) in zip(axes, [
            (active, "input", f"{len(active)} active sites"),
            (grown, "after a standard sparse conv", f"{len(grown)} active sites"),
            (active, "after a SUBMANIFOLD conv", f"{len(active)} active sites")]):
        _grid(ax, 0.3, 1.2, n, cell, active=act,
              on=BLUE if act is active else RED)
        ax.set_title(title, fontsize=9.5)
        ax.text(4.35, 0.4, sub, ha="center", fontsize=9,
                color=BLUE if act is active else RED, weight="bold")
    axes[1].text(4.35, 9.6, "sparsity destroyed", ha="center", fontsize=8.5,
                 color=RED, style="italic")
    axes[2].text(4.35, 9.6, "sparsity preserved exactly", ha="center", fontsize=8.5,
                 color="#1f9d6a", style="italic")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# NB7
# --------------------------------------------------------------------------
def calibration_concept():
    """What a reliability diagram is telling you."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    x = np.linspace(0.34, 1.0, 100)
    ax.plot([0.33, 1], [0.33, 1], "k--", lw=1.4, label="perfectly calibrated")
    ax.plot(x, 0.33 + (x - 0.33) * 0.55, color=RED, lw=2.4,
            label="overconfident (the usual case)")
    ax.plot(x, np.minimum(1.0, 0.33 + (x - 0.33) * 1.45), color=BLUE, lw=2.4,
            label="underconfident")
    ax.annotate("says 0.9, is right 0.64\nof the time", xy=(0.9, 0.643),
                xytext=(0.60, 0.355), fontsize=8.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.set_xlabel("confidence the model reports")
    ax.set_ylabel("accuracy it actually achieves")
    ax.set_xlim(0.3, 1.02); ax.set_ylim(0.3, 1.02)
    ax.set_title("reliability: does 0.9 mean 0.9?", fontsize=10)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def uncertainty_types():
    """Aleatoric noise you cannot reduce vs epistemic ignorance you can."""
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.concatenate([rng.uniform(0, 3.5, 90), rng.uniform(6.5, 10, 90)])
    noise = 0.12 + 0.16 * x
    y = np.sin(x) + rng.normal(0, noise)
    gx = np.linspace(0, 10, 300)
    mu = np.sin(gx)
    alea = 0.12 + 0.16 * gx
    epis = 1.15 * np.exp(-((gx - 5.0) ** 2) / 3.0)
    ax.fill_between(gx, mu - alea - epis, mu + alea + epis, color=VIOLET,
                    alpha=0.20, label="+ epistemic (no training data here)")
    ax.fill_between(gx, mu - alea, mu + alea, color=BLUE, alpha=0.35,
                    label="aleatoric (irreducible measurement noise)")
    ax.plot(gx, mu, color=INK, lw=1.8, label="prediction")
    ax.scatter(x, y, s=8, color=AMBER, edgecolors="none", label="training data")
    ax.axvspan(3.5, 6.5, color=GREY, alpha=0.10)
    ax.text(5.0, 2.35, "no data", ha="center", fontsize=9, color=GREY)
    ax.set_xlabel("input"); ax.set_ylabel("target")
    ax.set_title("two kinds of uncertainty, and only one shrinks with more data",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


ALL = [weight_sharing, representations, receptive_field, encoder_heads,
       skip_connection_types, gradient_flow, diagnosis_tree, batching_strategies,
       message_passing, transfer_strategies, submanifold_convolution,
       calibration_concept, uncertainty_types]
