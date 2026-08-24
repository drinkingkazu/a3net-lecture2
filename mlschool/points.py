"""
mlschool.points — point-cloud representation and the models used as baselines.

`to_point_cloud` is the padded "Option A" representation from notebook 3.
`DeepSets` and `PointGNN` live here because notebook 3 builds them as the
lesson, and notebooks 5 reuses them merely as baselines -- no reason to make a
student read the same forty lines twice.
"""

import numpy as np

from .data import to_points


def to_point_cloud(ds, n_max=384, seed=0, centre=True, scale=48.0):
    """Dense images -> (points, mask), padded to n_max.

    points[i] is (n_max, 3) = (x, y, log-charge); mask marks the real entries.
    With `centre`, coordinates are relative to the event's charge centroid, so
    absolute position in the detector carries no information.
    """
    rng = np.random.default_rng(seed)
    import torch
    n = len(ds["image"])
    pts = np.zeros((n, n_max, 3), np.float32)
    mask = np.zeros((n, n_max), np.float32)
    for i in range(n):
        coords, feats = to_points(ds["image"][i])
        if len(coords) > n_max:
            keep = rng.choice(len(coords), n_max, replace=False)
            coords, feats = coords[keep], feats[keep]
        m = len(coords)
        origin = ((coords * feats).sum(0) / feats.sum()) if centre else 0.0
        pts[i, :m, :2] = (coords - origin) / scale
        pts[i, :m, 2] = np.log1p(feats[:, 0]) / 3.0
        mask[i, :m] = 1.0
    return torch.tensor(pts), torch.tensor(mask)


def padding_report(n_hits, n_max):
    """What padding to n_max costs, and how many events lose hits."""
    slots = n_max * len(n_hits)
    real = int(np.minimum(n_hits, n_max).sum())
    print(f"padding/truncating to N = {n_max}")
    print(f"  padded slots        {slots:>12,}")
    print(f"  of which real hits  {real:>12,}  ({100 * real / slots:.0f} %)")
    print(f"  wasted on padding   {slots - real:>12,}  "
          f"({100 * (1 - real / slots):.0f} %)")
    print(f"  events truncated    {100 * float((n_hits > n_max).mean()):>11.0f} %"
          f"  (we throw hits away)")
    print(f"\nconcatenating instead: {int(n_hits.sum()):,} hits stored, "
          f"no waste, no truncation")


def shuffle_points(p, m):
    """Randomly permute the point ordering within each event."""
    import torch
    perm = torch.stack([torch.randperm(p.shape[1], device=p.device)
                        for _ in range(len(p))])
    return (torch.gather(p, 1, perm[..., None].expand(-1, -1, p.shape[-1])),
            torch.gather(m, 1, perm))


def point_accuracy(model, P, Mk, Y, shuffled=False, bs=128, dev=None):
    import torch
    from .training import device
    dev = dev or device()
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(P), bs):
            p, m = P[i:i + bs].to(dev), Mk[i:i + bs].to(dev)
            if shuffled:
                p, m = shuffle_points(p, m)
            correct += int((model(p, m).argmax(1).cpu() == Y[i:i + bs]).sum())
    return correct / len(P)


def train_point_model(name, model, Ptr, Mtr, Ytr, Pva, Mva, Yva, epochs=8,
                      lr=2e-3, bs=64, seed=0, scheduler=None, dev=None,
                      check_shuffled=True):
    """Train a model with signature model(points, mask) and report the numbers
    that matter for notebook 3: parameters, accuracy, accuracy under point
    relabelling, wall-clock, peak memory."""
    import time
    import torch
    import torch.nn.functional as F
    from .training import device, mem_baseline, mem_used, n_params
    dev = dev or device()
    torch.manual_seed(seed)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * int(np.ceil(len(Ptr) / bs))
    sched = (torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                 pct_start=0.2)
             if scheduler else None)
    base = mem_baseline()
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Ptr))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(model(Ptr[b].to(dev), Mtr[b].to(dev)),
                            Ytr[b].to(dev)).backward()
            opt.step()
            if sched:
                sched.step()
    secs, mem = time.time() - t0, mem_used(base)
    res = {"name": name, "params": n_params(model),
           "acc": point_accuracy(model, Pva, Mva, Yva, dev=dev),
           "acc_shuffled": (point_accuracy(model, Pva, Mva, Yva, True, dev=dev)
                            if check_shuffled else float("nan")),
           "seconds": secs, "peak_MB": mem}
    print(f"{name:<22} params {res['params']:>8,}   acc {res['acc']:.3f}   "
          + (f"acc(shuffled) {res['acc_shuffled']:.3f}   " if check_shuffled else "")
          + f"{secs:>4.0f} s   peak {mem:>5.0f} MB")
    return res, model


def comparison_table(rows, reprs=None):
    print(f"{'model':<22}{'repr.':<12}{'params':>9}{'val acc':>10}"
          f"{'shuffled':>10}{'train s':>9}{'peak MB':>10}")
    print("-" * 82)
    for r in rows:
        sh = "n/a" if np.isnan(r["acc_shuffled"]) else f"{r['acc_shuffled']:.3f}"
        rp = (reprs or {}).get(r["name"], "")
        print(f"{r['name']:<22}{rp:<12}{r['params']:>9,}{r['acc']:>10.3f}"
              f"{sh:>10}{r['seconds']:>9.0f}{r['peak_MB']:>10.0f}")


# --------------------------------------------------------------------------
# reference models (built from scratch in notebook 3; reused as baselines later)
# --------------------------------------------------------------------------
def build_deepsets(hidden=64, n_classes=3, in_dim=3):
    """Per-point MLP + symmetric aggregation. Permutation invariant."""
    import torch
    import torch.nn as nn

    class _DeepSets(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden))
            self.rho = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, n_classes))

        def forward(self, p, m):
            f = self.phi(p) * m[..., None]
            return self.rho(torch.cat(
                [f.sum(1) / 10.0,
                 f.masked_fill(m[..., None] == 0, -1e9).amax(1)], dim=-1))

    return _DeepSets()


def knn_graph(p, m, k=12):
    """Indices of the k nearest neighbours of each point, within each event."""
    import torch
    d = torch.cdist(p[..., :2], p[..., :2])
    d = d.masked_fill(m[:, None, :] == 0, 1e9)
    return d.topk(k, dim=-1, largest=False).indices


def build_pointgnn(hidden=48, k=12, depth=2, n_classes=3, in_dim=3):
    """EdgeConv message passing on a k-NN graph."""
    import torch
    import torch.nn as nn

    class _EdgeConv(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(2 * cin, cout), nn.ReLU(),
                                     nn.Linear(cout, cout), nn.ReLU())

        def forward(self, h, idx):
            B, N, C = h.shape
            nbr = torch.gather(h.unsqueeze(1).expand(B, N, N, C), 2,
                               idx[..., None].expand(-1, -1, -1, C))
            centre = h[:, :, None, :].expand_as(nbr)
            return self.mlp(torch.cat([centre, nbr - centre], dim=-1)).amax(2)

    class _PointGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.k = k
            self.convs = nn.ModuleList(
                [_EdgeConv(in_dim if i == 0 else hidden, hidden) for i in range(depth)])
            self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                      nn.Linear(hidden, n_classes))

        def forward(self, p, m):
            idx = knn_graph(p, m, self.k)
            h = p
            for conv in self.convs:
                h = conv(h, idx)
            h = h * m[..., None]
            return self.head(torch.cat(
                [h.sum(1) / 10.0,
                 h.masked_fill(m[..., None] == 0, -1e9).amax(1)], dim=-1))

    return _PointGNN()
