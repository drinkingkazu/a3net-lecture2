# ML Summer School — Lecture 2: Large models (CNNs, GNNs, deep learning applications)

Hands-on notebooks for the second lecture of a four-lecture machine-learning
summer school aimed at physics PhD students meeting ML applications for the first
time.

Three hours is not enough time for this material, so these notebooks are built
around **one organising thesis** rather than a tour of architectures:

> Your data has structure. The architecture's job is to encode that structure as
> a prior. The optimiser's job is to survive the resulting depth. Everything else
> is diagnosis.

Every notebook is self-contained and carries the lecture narrative **in place**
as markdown — they are readable as lecture notes and runnable as exercises. All
of them run on a free Colab GPU except NB6, which needs a container with
MinkowskiEngine.

---

## The notebooks

| # | notebook | covers | T4 runtime |
|---|---|---|---|
| 0 | `00_data_and_representations` | Universal approximation vs. optimisability. MLP (2.4 M params) vs. CNN (16 k) under translation. Dense / sparse / point-cloud representations and their cost. | ~5 min |
| 1 | `01_cnn_encoder_and_heads` | Receptive fields (computed *and* measured). Downsampling, resolution, U-Net skips. One encoder, three heads. Class imbalance. Loss landscapes and robust losses. | ~8 min |
| 2 | `02_debugging_clinic` | **The anchor notebook.** Five broken training runs to diagnose from curves alone, then the fixes: LR finder, schedules, optimisers, normalisation, residuals, shuffling, augmentation, weight decay. | ~15 min |
| 3 | `03_sparse_points_graphs` | Batching variable-size data. Permutation invariance. Deep Sets, k-NN graphs, message passing, PyTorch Geometric. Where sparse convolutions fit. | ~8 min |
| 4 | `04_transfer_learning` | The simulation-to-data gap, measured. Zero-label BatchNorm recalibration. Fine-tuning vs. linear probing vs. scratch as a function of label count. | ~13 min |
| 5 | `05_attention` | *Optional add-on.* Attention as message passing on a learned complete graph. Skippable if the lecture is running long. | ~10 min |

### Extra notebooks (not part of the 3 hours)

| # | notebook | covers | runtime |
|---|---|---|---|
| 6 | `06_sparse_convolutions` | **Needs MinkowskiEngine — not Colab.** Sparse tensors, the dilation failure mode and submanifold convolutions, a sparse classifier and U-Net, and how cost scales as the detector grows. | ~5 min |
| 7 | `07_calibration_uncertainty` | Stand-alone, Colab-friendly. Reliability diagrams and ECE, temperature scaling, why calibration dies under domain shift, MC dropout and deep ensembles, and regression uncertainty checked with a pull distribution. | ~10 min |

### Suggested 180-minute lecture mapping

| block | min | notebook |
|---|---|---|
| Framing: architecture as prior | 10 | NB0 §3–5 |
| CNNs, receptive fields, task heads | 45 | NB1 |
| Making deep models trainable | 50 | NB2 |
| Irregular data: sets and graphs | 35 | NB3 |
| Scale, transfer, practice | 25 | NB4 |
| *Attention (droppable)* | +15 | NB5 |

---

## The dataset

All eight notebooks use **one simulated dataset**, so that "representation is a
modelling choice" is something students *measure* rather than something they are
told. It is a cartoon of a liquid-argon TPC (`mlschool_data.py`):

- 96×96 charge images, ~2.5 % occupancy
- **tracks** — thin, continuous, constant dE/dx, with a Bragg peak at the end
- **showers** — widening stochastic cones
- uncorrelated noise hits
- three labels per event: event class (3-way), **per-pixel** semantics, and
  deposited energy — so one encoder can drive classification, segmentation and
  regression

Deliberate design choices that are themselves teaching material: charges are left
**unnormalised**, semantic classes are **98 % background**, and the detector
parameters (`gain`, `noise_rate`, `scatter`, `jitter`) are exposed so that domain
shifts and controlled experiments can be manufactured on demand.

The data is generated in seconds inside each notebook. Nothing to download.

> **Note on realism.** This is a synthetic simulator, not real detector data. It
> was chosen so the notebooks are self-contained on Colab and so that controlled
> experiments (matched control samples, tunable domain shift) are possible at all.
> Swapping in real LArTPC data means replacing `generate_dataset` and adjusting
> `SIZE`; everything downstream is agnostic.

---

## Running them

**On Colab (what students do).** Open the notebook, set *Runtime → Change runtime
type → GPU*, and run all cells. The first cell clones this repository and
installs it:

```python
REPO_URL = "https://github.com/kterao/mlschool-lecture2.git"
...
import mlschool as ms
```

It is idempotent — safe to re-run, and a no-op if `mlschool` is already
importable (e.g. a local checkout). After it, everything shared lives behind
`ms.`: the simulator, plotting, metrics, training boilerplate, schematic
diagrams and the slide viewer.

**Set `REPO_URL` before the school.** It is defined in exactly one place,
`build_notebooks.py`, and substituted into every notebook's setup cell at build
time. Change it there and rebuild.

Only NB3 installs anything further (`torch_geometric`, guarded by a
`try/except`). **NB6 is the exception** — it needs MinkowskiEngine and will not
run on Colab; give students the container image for that one.

**Locally, in the project container.**

```bash
apptainer exec --nv /home/kazu/sw/rtx4090/images/test.sif \
    jupyter nbconvert --to notebook --execute --inplace notebooks/00_*.ipynb
```

---

## Editing the notebooks

Do **not** edit `notebooks/*.ipynb` directly — they are build artefacts.

Sources live in `src/*.nbsrc`, a plain-text format with markers at column 0:

```
#%% md      markdown cell, raw markdown until the next marker
#%% code    code cell
#%% setup   the standard clone-and-install cell, generated from REPO_URL
#%% inject X   legacy: inline a module as a %%writefile cell (no longer used)
```

Solution notebooks live in `src/solutions/` and build to `notebooks/solutions/`.
After editing:

```bash
apptainer exec --nv /home/kazu/sw/rtx4090/images/test.sif python build_notebooks.py       # all
apptainer exec --nv /home/kazu/sw/rtx4090/images/test.sif python build_notebooks.py 02    # just NB2
```

Shipped notebooks are committed **without outputs**, so students execute
everything themselves.

```
mlschool/             the installed package -- everything the notebooks share
  data.py               detector simulator + representation converters
  plotting.py           event, training-curve and calibration plots
  metrics.py            IoU / recall / precision, ECE, pull statistics
  training.py           training-loop boilerplate and diagnostics
  points.py             point-cloud conversion and reference models
  sparse.py             MinkowskiEngine helpers (lazy import)
  diagrams.py           schematic figures for the lecture text
  slides.py             put PDF slides inside a notebook
pyproject.toml        makes `pip install -e .` work
build_notebooks.py    .nbsrc -> .ipynb   (REPO_URL is set here)
src/*.nbsrc           notebook sources (edit these)
src/solutions/*.nbsrc solution notebook sources
notebooks/**/*.ipynb  build artefacts (distribute these)
slides/, pdfs/        lecture slide assets
```

---

## Notes for the lecturer

**Every quantitative claim in the prose was measured, and several were rewritten
when the measurement disagreed.** Those disagreements are now teaching points in
their own right, and they are the parts most worth reading before you teach:

- **NB1** — inverse-frequency class weighting makes segmentation *worse* here.
  Imbalance is only fatal when the classes are confusable, which the notebook
  demonstrates by contrasting a 1×1-convolution model (track recall exactly
  0.000) against the U-Net (track IoU 0.90, unweighted). The lesson becomes "ask
  whether your model can see the difference before you reach for a reweighting
  scheme".
- **NB2 §3, patient E** — the curve that looks exactly like overfitting (train
  loss down, validation loss up) is **not** overfitting. Evaluated with batch
  statistics instead of BatchNorm's running averages, the same model is already
  at 0.62 accuracy in epoch 1 and improves monotonically. The notebook uses this
  to teach the general habit: before diagnosing overfitting, list everything that
  differs between your train and eval code paths. A genuine overfitting curve is
  then shown separately, using an MLP.
- **NB2 §5** — the classic "you forgot to normalise your inputs" failure will not
  reproduce, because Adam is scale-invariant per parameter and BatchNorm
  renormalises immediately. And the one configuration that *does* fail is not a
  mis-tuned learning rate — no LR from $10^{-3}$ to $1$ rescues it. Normalising
  shrinks the first-layer gradient 370×, stretching the spread across depth to
  25 000× (measured), and one global step size cannot serve both ends.
- **NB5** — the transformer beats the k-NN GNN, and the gap survives a
  parameter-matched control. The explanation is receptive field, not attention
  magic.
- **NB7** — the textbook deep-ensemble recipe ("train 5 with different seeds")
  changes *nothing* here; giving each member a different data subset cuts ECE
  fourfold. And the heteroscedastic regressor ends up correctly calibrated on
  average while its per-event $\sigma$ does **not** track which events are
  actually hard — two different failures, only one of which post-hoc calibration
  can fix.

**Deliberate omissions**, flagged so you can mention them:

- **Sparse/submanifold convolutions** are discussed in NB3 §7 but not run there,
  because MinkowskiEngine and spconv need compiled CUDA extensions that are
  fragile on Colab. **NB6 runs them properly** and requires the provided
  container.
- **Evaluation, calibration and uncertainty** are not covered anywhere in the
  four-lecture series. **NB7 fills this gap** as a stand-alone notebook; NB4 §5
  flags the consequences for physics analyses.
- Exact equivariance is left to Lecture 4. NB0 ends by measuring the aliasing
  that makes CNN translation invariance only approximate, which is the handoff.
