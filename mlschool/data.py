"""
mlschool.data — synthetic LArTPC-like event images for the ML summer school.

One dataset, used by every notebook in the series, in three representations:
dense image, sparse (coords, features), and point cloud / graph.

Physics cartoon
---------------
A liquid-argon TPC records ionisation charge on a 2D readout plane. Charged
particles leave two visually distinct signatures:

  * tracks  -- thin, continuous, roughly constant dE/dx, with a Bragg peak
               (a sharp rise in deposited charge) where the particle stops.
  * showers -- an electromagnetic cascade: a cone that starts narrow at the
               interaction vertex and widens with depth, diffuse and stochastic.

Plus a sprinkling of uncorrelated noise hits, which are *not* signal.

The images are sparse (a few percent of pixels are non-zero), which is the
whole reason the point-cloud / graph representations are interesting.

Labels provided (three tasks off one shared encoder)
----------------------------------------------------
  label   (N,)      event class: 0 = track only, 1 = shower only, 2 = both
  seg     (N,H,W)   per-pixel semantics: 0 = background, 1 = track, 2 = shower
  energy  (N,)      total true deposited charge (arbitrary units / 1000)

Deliberate design choices (they are teaching material, not accidents)
---------------------------------------------------------------------
  * Pixel charges are left on their RAW scale (tens to hundreds), NOT
    normalised. Forgetting to normalise inputs is one of the bugs in the
    debugging clinic notebook.
  * The semantic classes are heavily imbalanced (~97% background), so a
    plain per-pixel cross-entropy has a trivial degenerate optimum.
  * `jitter` controls how far the interaction vertex can sit from the image
    centre. Training with small jitter and testing with large jitter is the
    cleanest demonstration that an MLP memorises absolute position while a
    CNN does not.
  * `gain` / `noise_rate` / `scatter` let you manufacture a domain shift
    ("simulation" vs "data") for the transfer-learning notebook.
"""

import numpy as np

SIZE = 96
CLASS_NAMES = ["track only", "shower only", "track + shower"]
SEG_NAMES = ["background", "track", "shower"]
SEG_COLORS = ["#101418", "#4cc9f0", "#f7b32b"]


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def _deposit(acc, x, y, q, size):
    """Accumulate charge q at floating-point positions (x, y) onto a grid."""
    xi = np.round(x).astype(np.int64)
    yi = np.round(y).astype(np.int64)
    m = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
    np.add.at(acc, (yi[m], xi[m]), q[m])


def _track(rng, acc, size, x0, y0, theta, length, scatter=0.02, bragg=1.0):
    """A minimum-ionising particle: thin, continuous, Bragg peak at the end."""
    ds = 0.25
    n = max(int(length / ds), 8)
    th = theta + np.cumsum(rng.normal(0.0, scatter, n))
    x = x0 + np.cumsum(np.cos(th)) * ds
    y = y0 + np.cumsum(np.sin(th)) * ds
    s = np.arange(n) / n
    dq = 8.0 * (1.0 + 3.0 * bragg * np.exp(-((1.0 - s) / 0.06)))   # Bragg peak
    dq = dq * rng.lognormal(0.0, 0.35, n)                       # Landau-ish
    _deposit(acc, x, y, dq * ds, size)
    return x[-1], y[-1]


def _shower(rng, acc, size, x0, y0, theta, energy, scatter=0.02):
    """An EM cascade: narrow at the vertex, widening cone, stochastic."""
    n = max(int(600 * energy), 60)
    t = rng.gamma(2.5, 4.5, n)                                  # depth
    width = 0.4 + 0.16 * t + 30.0 * scatter                     # transverse spread
    r = rng.normal(0.0, 1.0, n) * width
    ct, st = np.cos(theta), np.sin(theta)
    x = x0 + t * ct - r * st
    y = y0 + t * st + r * ct
    dq = 3.0 * rng.lognormal(0.0, 0.6, n)
    _deposit(acc, x, y, dq, size)


# --------------------------------------------------------------------------
# one event
# --------------------------------------------------------------------------
def _event(rng, size, jitter, gain, noise_rate, scatter):
    trk = np.zeros((size, size), np.float32)
    shw = np.zeros((size, size), np.float32)

    label = rng.integers(0, 3)
    cx = size / 2 + rng.uniform(-jitter, jitter)
    cy = size / 2 + rng.uniform(-jitter, jitter)

    if label in (0, 2):
        for _ in range(rng.integers(1, 3)):
            _track(rng, trk, size, cx, cy,
                   rng.uniform(0, 2 * np.pi),
                   rng.uniform(0.25, 0.55) * size, scatter)
    if label in (1, 2):
        _shower(rng, shw, size, cx, cy,
                rng.uniform(0, 2 * np.pi),
                rng.uniform(0.6, 1.6), scatter)

    signal = trk + shw
    energy = signal.sum() / 1000.0

    seg = np.zeros((size, size), np.int8)
    seg[(trk > 0) & (trk >= shw)] = 1
    seg[(shw > 0) & (shw > trk)] = 2

    img = signal.copy()
    if noise_rate > 0:                                          # uncorrelated hits
        k = rng.poisson(noise_rate)
        xs = rng.integers(0, size, k)
        ys = rng.integers(0, size, k)
        img[ys, xs] += rng.uniform(1.0, 6.0, k).astype(np.float32)

    return (img * gain).astype(np.float32), seg, np.int64(label), np.float32(energy)


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
def generate_dataset(n, size=SIZE, seed=0, jitter=8.0, gain=1.0,
                     noise_rate=30.0, scatter=0.02, progress=False):
    """Generate `n` events.

    jitter      max displacement of the interaction vertex from the image
                centre, in pixels. Small = every event looks centred.
    gain        multiplies all charges (a detector calibration change).
    noise_rate  mean number of uncorrelated noise hits per event.
    scatter     multiple-scattering strength; also widens showers.

    Returns dict with keys: image (N,H,W) f32, seg (N,H,W) i8,
    label (N,) i64, energy (N,) f32.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((n, size, size), np.float32)
    seg = np.zeros((n, size, size), np.int8)
    lab = np.zeros(n, np.int64)
    ene = np.zeros(n, np.float32)

    it = range(n)
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, desc="generating events")
        except ImportError:
            pass

    for i in it:
        img[i], seg[i], lab[i], ene[i] = _event(
            rng, size, jitter, gain, noise_rate, scatter)

    return {"image": img, "seg": seg, "label": lab, "energy": ene}


# --------------------------------------------------------------------------
# a *controlled* two-class sample, for the MLP-vs-CNN translation experiment
# --------------------------------------------------------------------------
KINK_NAMES = ["straight", "kink"]


def generate_kink_dataset(n, size=SIZE, seed=0, jitter=6.0, length=52.0,
                          scatter=0.008, noise_rate=15.0, gain=1.0):
    """Straight track vs. two tracks meeting at a vertex.

    Why a second sample? In the 3-class dataset a shower deposits far more
    charge over far more pixels than a track, so *total charge* alone almost
    gives the answer away -- and total charge is translation invariant. A
    model could score well on shifted images while having learned nothing
    about shape, and our experiment would measure the wrong thing.

    Here both classes are built with the same total track length, the same
    charge per unit length, and the centre of charge is placed identically.
    Hit count and total charge therefore carry (almost) no information: the
    ONLY usable signal is the shape, which is exactly what we want to test.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((n, size, size), np.float32)
    lab = np.zeros(n, np.int64)

    for i in range(n):
        c = rng.integers(0, 2)
        canvas = np.zeros((size, size), np.float32)

        if c == 0:                                              # one straight track
            arms = [(rng.uniform(0, 2 * np.pi), length)]
        else:                                                   # two arms, one vertex
            a = rng.uniform(0.35, 0.65) * length
            th = rng.uniform(0, 2 * np.pi)
            dth = rng.uniform(np.pi / 4, 3 * np.pi / 4) * rng.choice([-1, 1])
            arms = [(th, a), (th + dth, length - a)]

        # draw into a big scratch canvas centred on itself, then recentre so
        # that the centre of charge lands at the same place for both classes
        big = size * 2
        scratch = np.zeros((big, big), np.float32)
        for th, L in arms:
            # bragg=0: a stopping particle deposits a burst of charge at its
            # end point, so a two-armed event would have two bright spots and
            # a one-armed event one -- a position-free giveaway. Suppressed.
            _track(rng, scratch, big, big / 2, big / 2, th, L, scatter, bragg=0.0)
        ys, xs = np.nonzero(scratch)
        q = scratch[ys, xs]
        cx = (xs * q).sum() / q.sum()
        cy = (ys * q).sum() / q.sum()
        tx = size / 2 + rng.uniform(-jitter, jitter)
        ty = size / 2 + rng.uniform(-jitter, jitter)
        _deposit(canvas, xs - cx + tx, ys - cy + ty, q, size)

        if noise_rate > 0:
            k = rng.poisson(noise_rate)
            canvas[rng.integers(0, size, k), rng.integers(0, size, k)] += \
                rng.uniform(1.0, 6.0, k).astype(np.float32)

        img[i] = canvas * gain
        lab[i] = c

    seg = (img > 0).astype(np.int8)
    return {"image": img, "label": lab, "seg": seg,
            "energy": img.sum((1, 2)) / 1000.0}


def check_shortcuts(ds, names=KINK_NAMES):
    """Can the two classes be told apart by global, position-free statistics?

    If they can, a translation-invariance experiment on this sample is
    meaningless -- the model can score well without learning any shape.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    img, y = ds["image"], ds["label"]
    feats = np.stack([img.sum((1, 2)), (img > 0).sum((1, 2)),
                      img.max((1, 2)), img.std((1, 2))], 1)
    acc = cross_val_score(LogisticRegression(max_iter=2000), feats, y, cv=3).mean()
    print("logistic regression on [total charge, hit count, max, std]")
    print(f"  accuracy = {acc:.3f}   (chance = {1 / len(names):.3f})")
    for k, nm in enumerate(["total charge", "hit count"]):
        print(f"  {nm:12s} " + "  ".join(
            f"{names[c]}: {feats[y == c, k].mean():8.1f}" for c in range(len(names))))
    return acc


def summarize(ds, name="dataset"):
    """Print the numbers you should always look at before training anything."""
    img, seg = ds["image"], ds["seg"]
    n = len(img)
    occ = (img > 0).mean()
    lines = [
        f"{name}: {n} events of {img.shape[1]}x{img.shape[2]}",
        f"  occupancy (non-zero pixels)   {100 * occ:.2f} %",
        f"  charge  min/mean/max          {img.min():.1f} / "
        f"{img[img > 0].mean():.1f} / {img.max():.1f}",
        f"  event class balance           " + ", ".join(
            f"{CLASS_NAMES[c]}={np.mean(ds['label'] == c):.2f}" for c in range(3)),
        f"  pixel class balance           " + ", ".join(
            f"{SEG_NAMES[c]}={np.mean(seg == c):.4f}" for c in range(3)),
        f"  energy  mean/std              {ds['energy'].mean():.2f} / "
        f"{ds['energy'].std():.2f}",
        f"  dense memory                  {img.nbytes / 1e6:.1f} MB",
        f"  non-zero (coord+value) memory {(img > 0).sum() * 12 / 1e6:.1f} MB",
    ]
    print("\n".join(lines))


# --------------------------------------------------------------------------
# alternative representations
# --------------------------------------------------------------------------
def to_points(image, seg=None, max_points=None, rng=None):
    """Dense image -> point cloud.

    Returns coords (M,2) float32 in pixel units and feats (M,1) float32
    charge, plus per-point labels if `seg` is given.
    """
    ys, xs = np.nonzero(image)
    q = image[ys, xs].astype(np.float32)
    if max_points is not None and len(q) > max_points:
        rng = rng or np.random.default_rng(0)
        keep = rng.choice(len(q), max_points, replace=False)
        ys, xs, q = ys[keep], xs[keep], q[keep]
    coords = np.stack([xs, ys], 1).astype(np.float32)
    feats = q[:, None]
    if seg is None:
        return coords, feats
    return coords, feats, seg[ys, xs].astype(np.int64)


def points_dataset(ds, max_points=None, seed=0):
    """Convert a whole dense dataset to a list of variable-length point clouds."""
    rng = np.random.default_rng(seed)
    return [to_points(ds["image"][i], ds["seg"][i], max_points, rng)
            for i in range(len(ds["image"]))]
