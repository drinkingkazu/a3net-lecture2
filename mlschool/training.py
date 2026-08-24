"""
mlschool.training — training-loop boilerplate.

The *contents* of a training loop are worth reading once (notebook 2 shows the
five quantities you should log and why). Re-typing the loop in six notebooks is
not, so it lives here.

Everything takes its data explicitly. No hidden globals: if you want to know
what a function trained on, it is in the call.
"""

import time

import numpy as np


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------
def predict_logits(model, X, bs=256, train_mode=False, dev=None):
    """Batched forward pass. train_mode=True keeps dropout on (for MC dropout)."""
    import torch
    dev = dev or device()
    model.train() if train_mode else model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + bs].to(dev)).cpu()
                          for i in range(0, len(X), bs)])


def accuracy(model, X, Y, bs=256, dev=None):
    return float((predict_logits(model, X, bs, dev=dev).argmax(1) == Y).float().mean())


# --------------------------------------------------------------------------
# augmentation
# --------------------------------------------------------------------------
def shift_augment(x, max_shift=12):
    """Randomly translate each image. Injects translation invariance through
    the data instead of through the architecture."""
    import torch
    out = torch.empty_like(x)
    dx = torch.randint(-max_shift, max_shift + 1, (len(x),))
    dy = torch.randint(-max_shift, max_shift + 1, (len(x),))
    for i in range(len(x)):
        out[i] = torch.roll(x[i], shifts=(int(dy[i]), int(dx[i])), dims=(1, 2))
    return out


def _make_optimizer(model, name, lr, weight_decay, params=None):
    import torch
    p = list(params) if params is not None else list(model.parameters())
    if name == "adam":
        return torch.optim.Adam(p, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(p, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(p, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")


# --------------------------------------------------------------------------
# plain training
# --------------------------------------------------------------------------
def fit(model, X, Y, epochs=None, steps=None, lr=2e-3, bs=64, optimizer="adam",
        weight_decay=0.0, scheduler=None, augment=False, shuffle=True,
        loss_fn=None, seed=0, params=None, dev=None, verbose=False):
    """Train and return the model.

    Give either `epochs` or `steps`. `steps` fixes the optimisation budget
    regardless of dataset size, which is what you want when comparing runs
    trained on different numbers of examples.
    """
    import torch
    import torch.nn.functional as F
    dev = dev or device()
    torch.manual_seed(seed)
    model = model.to(dev)
    loss_fn = loss_fn or F.cross_entropy
    opt = _make_optimizer(model, optimizer, lr, weight_decay, params)

    n = len(X)
    per_epoch = int(np.ceil(n / bs))
    total = steps if steps is not None else epochs * per_epoch
    sched = None
    if scheduler == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=total, pct_start=0.2)

    order = torch.argsort(Y) if not shuffle else None
    perm, cursor = (order if order is not None else torch.randperm(n)), 0
    model.train()
    for step in range(total):
        if cursor + bs > n:
            perm = order if order is not None else torch.randperm(n)
            cursor = 0
        b = perm[cursor:cursor + bs]
        cursor += bs
        if len(b) < 2:
            continue
        xb = shift_augment(X[b]) if augment else X[b]
        opt.zero_grad()
        loss = loss_fn(model(xb.to(dev)), Y[b].to(dev))
        loss.backward()
        opt.step()
        if sched:
            sched.step()
        if verbose and (step + 1) % per_epoch == 0:
            print(f"  step {step + 1}/{total}  loss {loss.item():.4f}")
    return model


# --------------------------------------------------------------------------
# instrumented training (notebook 2)
# --------------------------------------------------------------------------
def dead_fraction(model, xb, dev=None):
    """Fraction of ReLU outputs that are exactly zero."""
    import torch
    import torch.nn as nn
    dev = dev or device()
    stats = []
    hooks = [m.register_forward_hook(
        lambda _m, _i, o: stats.append(float((o == 0).float().mean())))
        for m in model.modules() if isinstance(m, nn.ReLU)]
    model.eval()
    with torch.no_grad():
        model(xb.to(dev))
    for h in hooks:
        h.remove()
    return float(np.mean(stats)) if stats else 0.0


def fit_instrumented(name, model, X, Y, Xval, Yval, epochs=6, lr=1e-3, bs=64,
                     optimizer="adam", weight_decay=0.0, shuffle=True,
                     augment=False, scheduler=None, clip=None, seed=0,
                     n_classes=3, dev=None, verbose=False, light=False,
                     log_every=1 / 20):
    """Train while logging the quantities from notebook 2 §1.

    Two resolutions, because the two kinds of quantity cost very different
    amounts to measure:

    * **train-side** (`train_loss`, `grad_norm`, `update_ratio`) are computed on
      every step anyway, so recording them `log_every` epochs apart is free.
      Sub-epoch resolution matters: a run whose data was never shuffled shows a
      sawtooth with one tooth per class, and averaging over an epoch hides it.
    * **validation** (`val_loss`, `val_acc`) and `dead` need extra forward
      passes, so they stay at epoch boundaries.

    The two share one x-axis in units of epochs: `step_epoch` for the fine
    series, `epoch` for the coarse ones.

    `model` may be a module or a zero-argument factory. Prefer the factory: the
    network is then built AFTER the seed is set, so the run reproduces whatever
    happened earlier in the notebook.

    `light=True` skips the per-step update-ratio bookkeeping, which copies every
    parameter twice per step, for runs where only the final accuracy matters.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    dev = dev or device()
    torch.manual_seed(seed)
    if not isinstance(model, nn.Module):
        model = model()                       # build after seeding
    model = model.to(dev)
    opt = _make_optimizer(model, optimizer, lr, weight_decay)

    n = len(X)
    per_epoch = int(np.ceil(n / bs))
    steps = epochs * per_epoch
    every = max(1, int(round(log_every * per_epoch)))     # steps between samples
    sched = (torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.15)
        if scheduler in ("onecycle", "cosine") else None)

    order = torch.argsort(Y) if not shuffle else None
    h = {k: [] for k in ("step_epoch", "train_loss", "grad_norm", "update_ratio",
                         "epoch", "val_loss", "val_acc", "dead", "lr")}
    acc = {"loss": 0.0, "g": 0.0, "u": 0.0, "k": 0}       # running bucket
    step = 0
    t0 = time.time()

    for _ in range(epochs):
        model.train()
        perm = order if order is not None else torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            xb = shift_augment(X[b]) if augment else X[b]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb.to(dev)), Y[b].to(dev))
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip if clip else 1e12))
            if light:
                opt.step()
            else:
                before = torch.cat([p.detach().flatten()
                                    for p in model.parameters()])
                opt.step()
                after = torch.cat([p.detach().flatten()
                                   for p in model.parameters()])
                acc["u"] += float((after - before).norm()
                                  / before.norm().clamp(min=1e-12))
            acc["loss"] += loss.item()
            acc["g"] += gnorm
            acc["k"] += 1
            step += 1
            if sched:
                sched.step()

            if acc["k"] >= every:                          # flush a fine sample
                h["step_epoch"].append(step / per_epoch)
                h["train_loss"].append(acc["loss"] / acc["k"])
                h["grad_norm"].append(acc["g"] / acc["k"])
                h["update_ratio"].append(acc["u"] / acc["k"])
                acc = {"loss": 0.0, "g": 0.0, "u": 0.0, "k": 0}

        if acc["k"]:                                       # partial bucket
            h["step_epoch"].append(step / per_epoch)
            h["train_loss"].append(acc["loss"] / acc["k"])
            h["grad_norm"].append(acc["g"] / acc["k"])
            h["update_ratio"].append(acc["u"] / acc["k"])
            acc = {"loss": 0.0, "g": 0.0, "u": 0.0, "k": 0}

        logits = predict_logits(model, Xval, dev=dev)      # once per epoch
        h["epoch"].append(step / per_epoch)
        h["val_loss"].append(float(F.cross_entropy(logits, Yval)))
        h["val_acc"].append(float((logits.argmax(1) == Yval).float().mean()))
        h["dead"].append(0.0 if light else dead_fraction(model, X[:64], dev))
        h["lr"].append(opt.param_groups[0]["lr"])
        if verbose:
            print(f"  ep {len(h['epoch'])}/{epochs}  "
                  f"train {h['train_loss'][-1]:.4f}  val {h['val_loss'][-1]:.4f}  "
                  f"acc {h['val_acc'][-1]:.3f}  |g| {h['grad_norm'][-1]:.2e}  "
                  f"upd {h['update_ratio'][-1]:.1e}")

    h["name"], h["seconds"], h["n_classes"] = name, time.time() - t0, n_classes
    h["model"] = model
    print(f"{name:<34} final val acc {h['val_acc'][-1]:.3f}   ({h['seconds']:.0f} s)")
    return h


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
def lr_finder(model, X, Y, lr_min=1e-7, lr_max=10.0, n_steps=120, bs=64,
              seed=0, dev=None):
    """Exponentially ramp the learning rate over a few hundred steps."""
    import torch
    import torch.nn.functional as F
    dev = dev or device()
    torch.manual_seed(seed)
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr_min)
    lrs = np.logspace(np.log10(lr_min), np.log10(lr_max), n_steps)
    losses = []
    perm = torch.randperm(len(X))
    model.train()
    for k, lr in enumerate(lrs):
        for g in opt.param_groups:
            g["lr"] = float(lr)
        start = (k * bs) % max(len(X) - bs, 1)
        b = perm[start:start + bs]
        opt.zero_grad()
        loss = F.cross_entropy(model(X[b].to(dev)), Y[b].to(dev))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return lrs, np.array(losses)


def layer_gradients(model, X, Y, bs=64, dev=None, ndim=4):
    """Gradient norm of every conv weight from one batch at initialisation."""
    import torch.nn.functional as F
    dev = dev or device()
    model = model.to(dev)
    model.train()
    F.cross_entropy(model(X[:bs].to(dev)), Y[:bs].to(dev)).backward()
    return [float(p.grad.norm()) for p in model.parameters()
            if p.grad is not None and p.dim() == ndim]


# --------------------------------------------------------------------------
# resource accounting
# --------------------------------------------------------------------------
def mem_baseline():
    """Reset CUDA peak-memory tracking and return the current allocation.

    `max_memory_allocated` is a process-wide high-water mark, so anything still
    resident from an earlier cell would be charged to whichever model you
    measure next. Report the increment, not the mark.
    """
    import gc
    import torch
    gc.collect()
    if not torch.cuda.is_available():
        return 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


def mem_used(base):
    import torch
    if not torch.cuda.is_available():
        return float("nan")
    return (torch.cuda.max_memory_allocated() - base) / 1e6
