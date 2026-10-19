# The label-authority study: structure and how to run it

This document describes what the study measures, how the repository is
organised, how to reproduce every result, and which output backs each table and
figure of the paper.

## 1. What the study measures

Emozionalmente contains 6,902 acted Italian clips from 431 speakers in seven
classes (anger, disgust, fear, joy, neutrality, sadness, surprise). Every clip
has an **expressed** label (the emotion the speaker was asked to act) and
exactly five **listener judgements** (34,510 valid votes from 829 listeners),
each with an audio-quality flag.

The same clips therefore support three operational ground truths:

| Authority | Construction | Training loss | Validation selection | Hard reference |
|---|---|---|---|---|
| `expressed` | one-hot elicitation label | cross-entropy | best accuracy | expressed |
| `recognized_plurality` | most frequent vote; a tie goes to the lowest class index | cross-entropy | best accuracy | plurality |
| `recognized_distribution` | vote proportions, p_c = n_c / 5, no smoothing | soft cross-entropy | lowest soft CE | plurality |

The tie rule makes the plurality reproducible; it does not make the chosen class
a stronger consensus. Class order (`config/study.yaml`) is disgust, joy, fear,
anger, surprise, sadness, neutrality.

**Representations.** XLS-R 300M, HuBERT Large and WavLM Large are frozen. Each
exposes 25 hidden states of width 1,024: state 0 is the convolutional front end,
states 1-24 are Transformer layers. Temporal means always exclude padded frames.

**Probes.**

| Probe | Input | Architecture |
|---|---|---|
| `linear` | one state, mean-pooled | Linear(1024,128) → ReLU → Linear(128,7) |
| `dense` | one state, full sequence | two frame-wise 1×1 convolutions (256 channels, ReLU) → masked mean → Linear(256,7) |
| `agg_dense` | all 25 states | learned softmax weight per state, weighted sum per frame → `dense` |

Single-layer probes read ten depths, 0/20/33/43/53/63/73/87/90/100 % of the
stack → states 0, 5, 8, 10, 13, 15, 18, 21, 22, 24. Per target: 3 backbones ×
(10 layers × 2 probes + 1 `agg_dense`) = 63 configurations; 189 in total, each
trained five times (945 runs).

**Protocol.** One speaker-independent split (4,142 / 1,380 / 1,380 clips) is
shared by all authorities. Adam, learning rate 1e-3, 30 epochs; batch size 256
(`linear`, `dense`) or 16 (`agg_dense`). Run *r* seeds PyTorch and NumPy with *r*,
so the five runs differ in initialisation and minibatch order.

**Metrics.** Hard decisions: unweighted average recall (UAR), the mean of the
class recalls over the classes present in the reference. Agreement strata use
plain accuracy. Probability vectors q are compared with the observed vote
distribution p by KL(p‖q) and the Brier score Σ_c (q_c − p_c)²; cross-entropy
and Jensen-Shannon are supporting diagnostics. Nominal Krippendorff's α
describes listener reliability.

**Selection and aggregation.** Checkpoints are chosen on validation, but the
reported layer of each (target, backbone, probe) is the one with the best
five-run mean **test** UAR against the target's own hard reference. Winner-based
numbers are therefore descriptive and optimistic. Target-level scores average
the nine selected configurations within each run and report mean ± sample SD
(ddof = 1) over the five runs. Paired clip bootstraps (1,000 replicates)
resample test clips and keep all five runs of a clip together; their intervals
are conditional on the trained and selected models.

**Agreement strata** (all-vote share of the winning class, mutually exclusive):
unanimous (1.00) · 0.80 · 0.60 · tied plurality · unique plurality below 0.60.
In the test split they hold 381 / 388 / 384 / 145 / 82 clips.

## 2. Repository layout

```
config/study.yaml            the only configuration file: paths, classes, backbones,
                             layers, probes, targets, split, training, analysis
label_authority/
  config.py                  loads the YAML; shared constants and output paths
  labels.py                  step 1: samples.csv + evaluations.csv -> results/labels.csv
  features.py                step 2: speaker-independent split + frozen hidden states -> features/<backbone>.h5
  data.py                    datasets over the HDF5 states (single layer, or all 25 streamed)
  probes.py                  linear, dense and agg_dense probes
  metrics.py                 UAR / accuracy / macro F1; KL, Brier, CE, JS
  train.py                   step 3: probe grid -> results/probe_results.csv + predictions
  analysis.py                step 4: every table and in-text number -> results/analysis/
  figures.py                 step 5: Figures 1-5 -> results/analysis/figures/
docs/STUDY.md                this file
```

Generated, git-ignored:

```
data/emozionalmente/         the corpus (obtain it from its authors)
features/<backbone>.h5       25 frozen states per clip, float16
results/
  labels.csv                 one row per clip: expressed label, vote histograms, pluralities, ties
  probe_results.csv          one row per configuration: five-run mean/SD of accuracy, UAR, macro F1,
                             agg_dense layer weights, path to its predictions
  predictions/<target>/<backbone>/<probe>_layer_XX.csv   per-run test probabilities
  checkpoints/<target>/<backbone>/<probe>_layer_XX/run_R.pt
  analysis/tables/*.csv, analysis/report.md, analysis/figures/*.png
```

## 3. Running the study

### Environment

Python 3.11. The versions in `requirements.txt` are those used for the paper.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

Paths come from `config/study.yaml` and can be overridden with
`LABEL_AUTHORITY_DATASET`, `LABEL_AUTHORITY_FEATURES`, `LABEL_AUTHORITY_RESULTS`
or a whole alternative config file via `LABEL_AUTHORITY_CONFIG`.

### Step 1 — label authorities (CPU, seconds)

```bash
python -m label_authority.labels
```

Reads the corpus metadata and writes `results/labels.csv`.

### Step 2 — frozen features (GPU)

```bash
python -m label_authority.features                       # all three backbones
python -m label_authority.features --backbone wavlm_large
```

Assigns the speaker-independent split and stores all 25 states of every clip in
`features/<backbone>.h5`. The split is deterministic and identical for the three
backbones. All states are held in memory before writing, so plan for a
high-RAM node; the files are large (tens of GB each).

### Step 3 — probe grid (GPU)

```bash
python -m label_authority.train --list      # 189 pending configurations
python -m label_authority.train             # run them all
python -m label_authority.train --target recognized_distribution --probe agg_dense
```

`--target`, `--backbone` and `--probe` can be repeated to select a slice;
`--layers 10,13` restricts single-layer probes; `--force` retrains existing
rows. Each finished configuration is written to `results/probe_results.csv`
straight away, so re-running the same command resumes an interrupted grid at
the next configuration. `agg_dense` streams every state from HDF5 in batches of
adjacent clips and is by far the slowest probe.

### Step 4 — tables and in-text numbers (CPU, about a minute)

```bash
python -m label_authority.analysis
```

Reads only `labels.csv`, `probe_results.csv` and the prediction files, so it
runs without the corpus audio, the features or a GPU. It writes one CSV per
analysis to `results/analysis/tables/` and a readable summary of every
paper number to `results/analysis/report.md`.

### Step 5 — figures (CPU, seconds)

```bash
python -m label_authority.figures
```

## 4. Where each paper result comes from

All tables are in `results/analysis/tables/`; figures in `results/analysis/figures/`.

| Paper | Output | Function |
|---|---|---|
| Fig. 1, 79.1 % clip agreement, per-class match rates | `human_confusion_clip_plurality.csv`, `fig1_human_confusion.png` | `analysis.human_tables`, `figures.fig1_human_confusion` |
| Leading mismatch pairs (217, 202, 163) | `human_confusion_pairs.csv` (`clip_count`) | `analysis.human_tables` |
| Table III(a): ratings, evaluator/clip accuracy, α | `human_reliability.csv` | `analysis.human_tables` |
| Table III(b): agreement strata | `human_agreement_strata.csv` | `analysis.human_tables` |
| Selected layer of every (target, backbone, probe) | `selected_configurations.csv` | `analysis.select` |
| Table IV: best own-reference UAR | `best_by_target.csv` | `analysis.best_by_target` |
| 63 matched recognized − expressed deltas | `own_reference_deltas.csv` | `analysis.own_reference_deltas` |
| Fig. 2(a) and Table V: common-reference UAR, KL, Brier | `common_reference.csv`, `fig2a_*.png` | `analysis.common_reference` |
| Fig. 2(b): accuracy by agreement stratum | `by_stratum.csv`, `fig2b_*.png` | `analysis.by_stratum` |
| Fig. 3: KL and Brier per target | `common_reference.csv`, `fig3a_kl.png`, `fig3b_brier.png` | `figures.fig3_distribution_alignment` |
| "Lowest Brier in all nine matched comparisons" | `brier_matched_comparisons.csv` | `analysis.brier_matched_comparisons` |
| Fig. 4: distributional error by stratum | `by_stratum.csv`, `fig4_*.png` | `analysis.by_stratum` |
| Table VI: model–human top-10 pair overlap | `confusion_overlap.csv`, `confusion_overlap_by_run.csv` | `analysis.confusion_overlap` |
| Table VII: tied and below-0.60 subsets | `ambiguous_subsets.csv` | `analysis.ambiguous_subsets` |
| Configuration-matched soft − hard sensitivity | `matched_soft_hard.csv` | `analysis.matched_soft_hard` |
| Good-quality re-scoring | `good_quality.csv` | `analysis.good_quality` |
| Table VIII: best single layer vs. `agg_dense` | `aggregation_vs_single.csv` | `analysis.aggregation_vs_single` |
| Fig. 5: `agg_dense` layer weights | `agg_layer_weights.csv`, `fig5_*.png` | `figures.fig5_layer_weights` |
| Table IX and subset intervals: paired bootstrap | `bootstrap.csv` | `analysis.bootstrap` |

## 5. Reproducibility notes

* Probe training on a GPU is not bit-exact across hardware and library
  versions; expect retrained numbers to differ from the paper in the third
  decimal. Steps 4-5 are deterministic given the same predictions.
* All standard deviations are sample SDs (ddof = 1) over the five runs. The
  `agg_dense` weight SDs in `probe_results.csv` follow the same convention.
* The bootstrap seed and replicate count are in `config/study.yaml`
  (`analysis.bootstrap_seed`, `analysis.bootstrap_iterations`).
* Layers and winners are selected on the test split, as in the paper. A
  confirmatory study should select on validation data and keep an untouched
  test set.
