"""
mlschool.sparse — MinkowskiEngine helpers for notebook 6.

MinkowskiEngine is imported lazily inside the functions, so importing
`mlschool` on Colab (where it is not installed) still works.
"""

import time

import numpy as np

from .data import to_points


def require_me():
    """Import MinkowskiEngine with a useful message if it is missing."""
    try:
        import MinkowskiEngine as ME
    except ImportError as exc:
        raise ImportError(
            "MinkowskiEngine is not installed. Run this notebook in the "
            "container provided with the lecture, or install spconv and adapt "
            "the API names. See the requirements box at the top of notebook 6."
        ) from exc
    return ME


def to_sparse_list(ds, scale=1.0, with_labels=False):
    """Dense images -> per-event (int coordinates, features[, point labels])."""
    import torch
    out = []
    for i in range(len(ds["image"])):
        if with_labels:
            c, f, l = to_points(ds["image"][i], ds["seg"][i])
            out.append((torch.tensor(c, dtype=torch.int32),
                        torch.tensor(f / scale), torch.tensor(l)))
        else:
            c, f = to_points(ds["image"][i])
            out.append((torch.tensor(c, dtype=torch.int32), torch.tensor(f / scale)))
    return out


def make_batch(items, idx, device="cuda"):
    """Collate events into one SparseTensor (coordinates carry a batch column)."""
    ME = require_me()
    coords, feats = ME.utils.sparse_collate([items[i][0] for i in idx],
                                            [items[i][1] for i in idx])
    return ME.SparseTensor(features=feats.float().to(device),
                           coordinates=coords.to(device))


def make_seg_batch(items, idx, device="cuda"):
    """As `make_batch`, plus the concatenated per-hit labels."""
    import torch
    ME = require_me()
    coords, feats = ME.utils.sparse_collate([items[i][0] for i in idx],
                                            [items[i][1] for i in idx])
    labels = torch.cat([items[i][2] for i in idx])
    return (ME.SparseTensor(features=feats.float().to(device),
                            coordinates=coords.to(device)), labels.to(device))


def active_site_counts(x, layers):
    """Push a SparseTensor through `layers` and record the active-site count."""
    import torch
    n = [x.C.shape[0]]
    h = x
    with torch.no_grad():
        for layer in layers:
            h = layer(h)
            n.append(h.C.shape[0])
    return n


def train_sparse_classifier(model, items, Y, val_items, Yval, epochs=6, bs=64,
                            lr=2e-3, seed=0, device="cuda"):
    import torch
    import torch.nn.functional as F
    from .training import mem_baseline, mem_used
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    base = mem_baseline()
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(items))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs].tolist()
            opt.zero_grad()
            F.cross_entropy(model(make_batch(items, idx, device)),
                            Y[idx].to(device)).backward()
            opt.step()
    secs, mem = time.time() - t0, mem_used(base)
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(val_items), bs):
            idx = list(range(i, min(i + bs, len(val_items))))
            correct += int((model(make_batch(val_items, idx, device)).argmax(1).cpu()
                            == Yval[idx]).sum())
    return correct / len(val_items), secs, mem


def train_dense_classifier(model, X, Y, XV, YV, epochs=6, bs=64, lr=2e-3,
                           seed=0, device="cuda"):
    """The dense reference, measured exactly the same way."""
    import torch
    import torch.nn.functional as F
    from .training import mem_baseline, mem_used
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    base = mem_baseline()
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(model(X[b].to(device)), Y[b].to(device)).backward()
            opt.step()
    secs, mem = time.time() - t0, mem_used(base)
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(XV), 128):
            correct += int((model(XV[i:i + 128].to(device)).argmax(1).cpu()
                            == YV[i:i + 128]).sum())
    return correct / len(XV), secs, mem


def scaling_benchmark(sizes, dense_factory, sparse_factory, generate, batch=32,
                      device="cuda", repeats=5):
    """Time and peak memory of one forward+backward, dense vs sparse, vs size.

    dense_factory() / sparse_factory() build fresh models; generate(size, batch)
    returns a dataset dict.
    """
    import gc
    import torch
    import torch.nn.functional as F
    from .training import mem_baseline, mem_used
    ME = require_me()
    rows = []
    print(f"{'size':>7}{'pixels':>12}{'hits':>9}{'occ%':>7}"
          f"{'dense MB':>11}{'dense ms':>10}{'sparse MB':>11}{'sparse ms':>11}")
    print("-" * 78)
    for size in sizes:
        ds = generate(size, batch)
        y = torch.randint(0, 3, (batch,)).to(device)
        hits = int((ds["image"] > 0).sum())
        scale = float(np.percentile(ds["image"][ds["image"] > 0], 99))

        base = mem_baseline()
        try:
            model = dense_factory().to(device)
            X = torch.tensor(ds["image"])[:, None].to(device) / scale
            for _ in range(3):
                F.cross_entropy(model(X), y).backward()
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(repeats):
                F.cross_entropy(model(X), y).backward()
            torch.cuda.synchronize()
            d_ms, d_mb = (time.time() - t0) / repeats * 1000, mem_used(base)
            del model, X
        except torch.cuda.OutOfMemoryError:
            d_ms = d_mb = float("nan")

        base = mem_baseline()
        items = to_sparse_list(ds, scale=scale)
        model = sparse_factory().to(device)
        for _ in range(3):
            F.cross_entropy(model(make_batch(items, range(batch), device)), y).backward()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(repeats):
            F.cross_entropy(model(make_batch(items, range(batch), device)), y).backward()
        torch.cuda.synchronize()
        s_ms, s_mb = (time.time() - t0) / repeats * 1000, mem_used(base)
        del model
        ME.clear_global_coordinate_manager()
        gc.collect()

        px = batch * size * size
        rows.append((size, px, hits, d_mb, d_ms, s_mb, s_ms))
        print(f"{size:>7}{px:>12,}{hits:>9,}{100 * hits / px:>7.2f}"
              f"{d_mb:>11.0f}{d_ms:>10.1f}{s_mb:>11.0f}{s_ms:>11.1f}")
    return rows


def plot_scaling(rows):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sizes = [r[0] for r in rows]
    for ax, (di, si, lab) in zip(axes, [(3, 5, "peak memory [MB]"),
                                        (4, 6, "time per step [ms]")]):
        ax.loglog(sizes, [r[di] for r in rows], "o-", label="dense")
        ax.loglog(sizes, [r[si] for r in rows], "s-", label="sparse")
        ax.set_xlabel("detector linear size [pixels]"); ax.set_ylabel(lab)
        ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.suptitle("dense cost grows with volume; sparse cost grows with hits", y=1.02)
    fig.tight_layout()
    return fig
