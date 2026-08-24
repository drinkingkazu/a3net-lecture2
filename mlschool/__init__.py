"""
mlschool — helpers for ML Summer School lecture 2 (large models).

The notebooks carry the lecture. This package carries the parts of the code
that are *not* the lecture: the detector simulator, plotting, metrics, and
training-loop boilerplate. Anything a student needs to read in order to follow
the argument stays visible in the notebook.

    import mlschool as ms

    events = ms.generate_dataset(4000, seed=0)
    ms.summarize(events)
    ms.plot_class_examples(events)
"""

import pathlib as _pathlib

#: Root of the course repository. With the editable install used by the
#: notebooks' setup cell, this points into the cloned checkout, so assets that
#: ship with the repo (slides/, pdfs/) can be found no matter what the working
#: directory is -- which on Colab is /content, not the clone.
ROOT = _pathlib.Path(__file__).resolve().parent.parent


def asset(*parts):
    """Absolute path to a file shipped with the repository.

        ms.asset("slides", "NB1", "intro")
    """
    return str(ROOT.joinpath(*parts))


# `sparse` imports MinkowskiEngine lazily (inside its functions), so listing
# it here is safe even on Colab where ME is absent.
from . import diagrams, metrics, plotting, points, slides, sparse, training
from .data import (
    CLASS_NAMES,
    KINK_NAMES,
    SEG_COLORS,
    SEG_NAMES,
    SIZE,
    check_shortcuts,
    generate_dataset,
    generate_kink_dataset,
    points_dataset,
    summarize,
    to_points,
)
from .metrics import (
    calibration_bins,
    class_fractions,
    ece,
    entropy,
    pull_report,
    report_calibration,
    seg_metrics,
    show_metrics,
    sigma_ranking_table,
)
from .plotting import (
    plot_class_examples,
    plot_event,
    plot_grid,
    plot_histories,
    plot_layer_gradients,
    plot_lr_finder,
    plot_points,
    plot_pull,
    plot_segmentation_comparison,
    reliability_panel,
    reliability_plot,
)
from .training import (
    accuracy,
    device,
    fit,
    fit_instrumented,
    layer_gradients,
    lr_finder,
    mem_baseline,
    mem_used,
    n_params,
    predict_logits,
    shift_augment,
)

def hello():
    """One-line environment report, printed by every notebook's setup cell."""
    import os
    import torch
    dev = training.device()
    print(f"mlschool {__version__} ({os.path.dirname(__file__)})  |  "
          f"torch {torch.__version__}  |  device: {dev}")
    if dev == "cpu":
        print("No GPU found. Everything still runs, roughly 5x slower.")


__version__ = "1.0.0"
