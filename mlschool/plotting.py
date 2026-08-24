"""
mlschool.plotting — display helpers.

Nothing here is conceptual. It exists so that the notebooks can show one line
(`ms.plot_grid(events)`) instead of twenty lines of matplotlib that nobody
should have to read during a lecture.
"""

import numpy as np

from .data import CLASS_NAMES, SEG_COLORS, SEG_NAMES, SIZE

BG = "#101418"


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------
def plot_event(ds, i, ax=None, mode="charge", title=None):
    """Draw one event. mode = 'charge' | 'seg'."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 3.2))
    if mode == "charge":
        img = np.ma.masked_where(ds["image"][i] == 0, ds["image"][i])
        cm = plt.cm.viridis.copy()
        cm.set_bad(BG)
        ax.imshow(img, cmap=cm, origin="lower", interpolation="nearest")
    else:
        ax.imshow(ds["seg"][i], cmap=ListedColormap(SEG_COLORS), vmin=0, vmax=2,
                  origin="lower", interpolation="nearest")
    if title is None and "label" in ds and "energy" in ds:
        title = f"{CLASS_NAMES[ds['label'][i]]}\nE={ds['energy'][i]:.2f}"
    ax.set_title(title or "", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def plot_grid(ds, idx=None, n=6, titles=None):
    """A row of events: charge on top, per-pixel semantics below."""
    import matplotlib.pyplot as plt
    idx = list(range(n)) if idx is None else list(idx)
    fig, axes = plt.subplots(2, len(idx), figsize=(2.0 * len(idx), 4.4))
    axes = np.atleast_2d(axes)
    for k, i in enumerate(idx):
        plot_event(ds, i, axes[0, k], "charge",
                   None if titles is None else titles[k])
        plot_event(ds, i, axes[1, k], "seg", title="")
    axes[0, 0].set_ylabel("charge", fontsize=9)
    axes[1, 0].set_ylabel("semantics", fontsize=9)
    fig.tight_layout()
    return fig


def plot_class_examples(ds, names=CLASS_NAMES, per_class=2):
    """Two examples of each class, charge and semantics."""
    idx = [int(np.where(ds["label"] == c)[0][k])
           for c in range(len(names)) for k in range(per_class)]
    return plot_grid(ds, idx)


def plot_points(coords, values, ax=None, discrete=False, title="", size=6):
    """Scatter a point cloud on a detector-like background."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    if ax is None:
        _, ax = plt.subplots(figsize=(3.4, 3.4))
    cmap = ListedColormap(SEG_COLORS[1:]) if discrete else "viridis"
    ax.scatter(coords[:, 0], coords[:, 1], c=values, s=size, cmap=cmap)
    ax.set_xlim(0, SIZE); ax.set_ylim(0, SIZE)
    ax.set_aspect("equal"); ax.set_facecolor(BG)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    return ax


def plot_segmentation_comparison(ds, preds, idx, row_labels):
    """Input charge, truth, and one row per model prediction."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(SEG_COLORS)
    rows = 2 + len(preds)
    fig, axes = plt.subplots(rows, len(idx), figsize=(2.3 * len(idx), 2.3 * rows))
    axes = np.atleast_2d(axes)
    for k, i in enumerate(idx):
        plot_event(ds, i, axes[0, k], "charge")
        axes[1, k].imshow(ds["seg"][i], cmap=cmap, vmin=0, vmax=2, origin="lower")
        for r, p in enumerate(preds):
            axes[2 + r, k].imshow(p[i], cmap=cmap, vmin=0, vmax=2, origin="lower")
        for r in range(1, rows):
            axes[r, k].set_xticks([]); axes[r, k].set_yticks([])
    for r, name in enumerate(["input charge", "truth"] + list(row_labels)):
        axes[r, 0].set_ylabel(name, fontsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# training diagnostics
# --------------------------------------------------------------------------
PANELS = [("train_loss", "training loss", True),
          ("val_loss", "validation loss", True),
          ("val_acc", "validation accuracy", False),
          ("grad_norm", "gradient norm", True),
          ("update_ratio", r"update ratio  $|\Delta\theta|/|\theta|$", True),
          ("dead", "dead-unit fraction", False)]


def plot_histories(runs, title=None, figsize=(13, 6.5)):
    """The six-panel instrument panel, one line per run."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    for ax, (key, label, logy) in zip(axes.ravel(), PANELS):
        for r in runs:
            ax.plot(range(1, len(r[key]) + 1), r[key], "o-", ms=3, label=r["name"])
        ax.set_title(label, fontsize=9); ax.set_xlabel("epoch")
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    n_cls = runs[0].get("n_classes", 3)
    axes[0, 2].axhline(1 / n_cls, ls=":", c="grey")
    axes[1, 1].axhspan(3e-4, 3e-3, color="green", alpha=0.12)   # healthy band
    axes[0, 0].legend(fontsize=7)
    if title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


def plot_layer_gradients(series, title="why deep plain networks do not train"):
    """Gradient norm per layer at initialisation. series = {label: [norms]}."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for (label, g), style in zip(series.items(), ["o-", "s-", "^-", "v-"]):
        ax.semilogy(range(1, len(g) + 1), g, style, ms=4, label=label)
    ax.set_xlabel("conv layer index  (1 = closest to the input)")
    ax.set_ylabel("gradient norm at initialisation")
    ax.set_title(title, fontsize=10)
    ax.legend(); ax.grid(alpha=0.3, which="both")
    return fig


def plot_lr_finder(lrs, losses, smooth_window=7):
    """The LR range test, with a suggested value."""
    import matplotlib.pyplot as plt
    smooth = np.convolve(losses, np.ones(smooth_window) / smooth_window, mode="same")
    k = smooth_window // 2
    best = int(np.argmin(smooth[k:-k])) + k
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(lrs, losses, alpha=0.3, color="grey", label="raw")
    ax.semilogx(lrs[k:-k], smooth[k:-k], lw=2, label="smoothed")
    ax.axvline(lrs[best], color="red", ls="--", label=f"minimum @ {lrs[best]:.1e}")
    ax.axvline(lrs[best] / 10, color="green", ls="--",
               label=f"suggested lr = {lrs[best] / 10:.1e}")
    ax.set_xlabel("learning rate"); ax.set_ylabel("batch loss")
    ax.set_title("LR range test: one epoch, and you never guess again", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    return fig, lrs[best] / 10


# --------------------------------------------------------------------------
# calibration / uncertainty
# --------------------------------------------------------------------------
def reliability_plot(ax, probs, labels, title, n_bins=15):
    """Accuracy vs confidence, with bin populations annotated."""
    from .metrics import calibration_bins
    bins = calibration_bins(probs, labels, n_bins)
    cf = [b[1] for b in bins]; acc = [b[2] for b in bins]; cnt = [b[3] for b in bins]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.bar(cf, acc, width=0.9 / n_bins, color="#4cc9f0",
           edgecolor="k", lw=0.4, label="accuracy")
    ax.plot(cf, cf, color="#f7b32b", lw=1.5, label="confidence")
    for c, a, n in zip(cf, acc, cnt):
        if n > 5:
            ax.text(c, max(a, c) + 0.02, str(n), fontsize=6, ha="center", rotation=90)
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.12)
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.set_title(title, fontsize=9); ax.legend(fontsize=7, loc="upper left")
    return ax


def reliability_panel(items, n_bins=15):
    """items = [(probs, labels, title), ...] side by side."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(items), figsize=(4.4 * len(items), 4.1))
    axes = np.atleast_1d(axes)
    for ax, (p, y, t) in zip(axes, items):
        reliability_plot(ax, p, y, t, n_bins)
    fig.tight_layout()
    return fig


def plot_pull(pulls, titles):
    """Pull histograms against a unit Gaussian. pulls = list of 1-D arrays."""
    import matplotlib.pyplot as plt
    grid = np.linspace(-5, 5, 200)
    gauss = np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi)
    fig, axes = plt.subplots(1, len(pulls), figsize=(5.2 * len(pulls), 3.9))
    axes = np.atleast_1d(axes)
    for ax, pull, title in zip(axes, pulls, titles):
        pull = np.asarray(pull)
        ax.hist(pull, bins=np.linspace(-5, 5, 60), density=True,
                color="#4cc9f0", edgecolor="k", lw=0.3)
        ax.plot(grid, gauss, "k--", lw=1.5, label="unit Gaussian")
        ax.set_title(f"{title}\nmean {pull.mean():+.2f}, std {pull.std():.2f}",
                     fontsize=9)
        ax.set_xlabel(r"$(y_{\rm true} - \mu)\,/\,\sigma$")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
