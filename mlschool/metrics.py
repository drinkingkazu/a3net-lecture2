"""
mlschool.metrics — the numbers you should report instead of bare accuracy.

Segmentation: IoU / recall / precision, because pixel accuracy is meaningless
when 98% of pixels are background.

Calibration: reliability bins, ECE, and the pull statistics a physicist would
actually ask for.
"""

import numpy as np

from .data import SEG_NAMES


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------
def seg_metrics(pred, target, n_classes=3, names=None):
    """Per-class IoU, recall and precision. Accepts torch tensors or arrays."""
    names = names or SEG_NAMES
    pred = np.asarray(pred.cpu() if hasattr(pred, "cpu") else pred)
    target = np.asarray(target.cpu() if hasattr(target, "cpu") else target)
    out = {}
    for c in range(n_classes):
        p, t = (pred == c), (target == c)
        inter = int((p & t).sum())
        out[names[c]] = {
            "IoU": inter / max(int((p | t).sum()), 1),
            "recall": inter / max(int(t.sum()), 1),
            "precision": inter / max(int(p.sum()), 1),
        }
    return out


def show_metrics(metrics, title=""):
    if title:
        print(f"\n{title}")
    print(f"  {'class':<12}{'IoU':>8}{'recall':>10}{'precision':>12}")
    for name, d in metrics.items():
        print(f"  {name:<12}{d['IoU']:>8.3f}{d['recall']:>10.3f}{d['precision']:>12.3f}")


def class_fractions(x, n_classes=3, names=None):
    names = names or SEG_NAMES
    x = np.asarray(x.cpu() if hasattr(x, "cpu") else x)
    frac = np.bincount(x.ravel(), minlength=n_classes) / x.size
    return ", ".join(f"{names[c]}={frac[c]:.4f}" for c in range(n_classes)), frac


# --------------------------------------------------------------------------
# classification calibration
# --------------------------------------------------------------------------
def calibration_bins(probs, labels, n_bins=15):
    """[(bin centre, mean confidence, accuracy, count), ...] for non-empty bins."""
    conf, pred = probs.max(1)
    correct = (pred == labels).float()
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() > 0:
            out.append(((edges[i] + edges[i + 1]) / 2, float(conf[m].mean()),
                        float(correct[m].mean()), int(m.sum())))
    return out


def ece(probs, labels, n_bins=15):
    """Expected calibration error: population-weighted |accuracy - confidence|."""
    n = len(labels)
    return sum(cnt / n * abs(acc - cf)
               for _, cf, acc, cnt in calibration_bins(probs, labels, n_bins))


def report_calibration(tag, probs, labels):
    """One line: accuracy, mean confidence, the gap, ECE, NLL."""
    import torch.nn.functional as F
    conf, pred = probs.max(1)
    acc = float((pred == labels).float().mean())
    nll = float(F.nll_loss(probs.clamp_min(1e-12).log(), labels))
    e = ece(probs, labels)
    print(f"{tag:<34} acc {acc:.3f}   mean conf {float(conf.mean()):.3f}   "
          f"gap {float(conf.mean()) - acc:+.3f}   ECE {e:.4f}   NLL {nll:.3f}")
    return e


def entropy(probs):
    p = probs.clamp_min(1e-12)
    return -(p * p.log()).sum(1)


# --------------------------------------------------------------------------
# regression uncertainty
# --------------------------------------------------------------------------
def pull_report(y_true, mu, sigma, scale=1.0, label="pull"):
    """Pull mean/width plus 1/2/3 sigma coverage against Gaussian expectations."""
    pull = (y_true - mu) / sigma
    print(f"{label}:  mean {float(pull.mean()):+.3f}   std {float(pull.std()):.3f}"
          f"     (want 0.000 and 1.000)")
    for k, expected in zip((1, 2, 3), (0.6827, 0.9545, 0.9973)):
        cov = float((pull.abs() < k).float().mean())
        print(f"  coverage |pull| < {k}:  {cov:.3f}   (Gaussian expects {expected:.3f})")
    return pull


def sigma_ranking_table(y_true, mu, sigma, scale=1.0, n_groups=3):
    """Does the predicted sigma actually track the realised error?"""
    import torch
    order = torch.argsort(sigma)
    per = len(order) // n_groups
    names = ["smallest sigma", "middle", "largest sigma"] if n_groups == 3 else \
            [f"group {i + 1}" for i in range(n_groups)]
    print(f"{'group':<26}{'predicted sigma':>18}{'actual RMSE':>14}")
    print("-" * 58)
    rows = []
    for i, name in enumerate(names):
        sel = order[i * per:(i + 1) * per] if i < n_groups - 1 else order[i * per:]
        pred = float(sigma[sel].mean()) * scale
        rmse = float((y_true - mu)[sel].pow(2).mean().sqrt()) * scale
        rows.append((pred, rmse))
        print(f"{name + ' third':<26}{pred:>18.4f}{rmse:>14.4f}")
    return rows
