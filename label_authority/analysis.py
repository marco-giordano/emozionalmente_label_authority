"""Recompute every table and in-text statistic of the paper from saved predictions.

Inputs:  results/labels.csv, results/probe_results.csv, results/predictions/
Outputs: results/analysis/tables/*.csv and results/analysis/report.md

Conventions used throughout (they match the paper):
  * Selection. For every (target, backbone, probe) the reported layer is the one
    with the highest five-run mean test UAR against the target's own hard
    reference; this gives nine selected configurations per target. The best
    overall configuration of a target is the best of its nine. Selection is
    post hoc on the test split, so winner-based numbers are descriptive.
  * Own hard reference. expressed -> expressed label; both recognized targets ->
    the all-vote plurality (lowest class index wins ties).
  * Equal-configuration aggregation. A target-level score averages the nine
    selected configurations within each run; the table reports the mean and the
    sample SD (ddof=1) of those five run-level values.
  * Strata. Clips are split by the all-vote share of the winning class:
    unanimous (1.00), 0.80, 0.60, tied plurality, unique plurality below 0.60.

Run:  python -m label_authority.analysis
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from label_authority.config import (BACKBONES, CFG, CLASS_DISPLAY, CLASSES, DISTRIBUTION, EXPRESSED,
                                    N_CLASSES, N_RUNS, PLURALITY, PROBES, TARGET_DISPLAY, TARGETS,
                                    analysis_dir, own_reference, probe_results_path, results_dir)
from label_authority.labels import STRATA, read_labels, stratum
from label_authority.metrics import distribution_rows, hard_metrics

DIST_METRICS = ("kl", "brier", "ce", "js")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class TestClips:
    """Clip-level references for the test split, sorted by sample id."""
    ids: np.ndarray
    refs: dict[str, np.ndarray]           # hard references: expressed, recognized_plurality, plurality_good
    human: dict[str, np.ndarray]          # vote distributions: all, good  (n, 7)
    stratum: np.ndarray
    is_tie: np.ndarray
    top_share: np.ndarray
    n_good: np.ndarray


@dataclass
class Config:
    """One trained configuration: five runs of test predictions aligned to TestClips."""
    target: str
    backbone: str
    probe: str
    layer: int
    pred: np.ndarray                      # (runs, n)
    probs: np.ndarray                     # (runs, n, 7)
    row: dict

    def uar_runs(self, ref: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        m = slice(None) if mask is None else mask
        return np.array([hard_metrics(ref[m], p[m])["uar"] for p in self.pred])

    def accuracy_runs(self, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.array([np.mean(p[mask] == ref[mask]) for p in self.pred])

    def dist_runs(self, human: np.ndarray, metric: str, mask: np.ndarray | None = None) -> np.ndarray:
        m = slice(None) if mask is None else mask
        return np.array([distribution_rows(human[m], q[m])[metric].mean() for q in self.probs])


def load_test_clips(labels: pd.DataFrame, ids: np.ndarray) -> TestClips:
    t = labels.set_index("sample_id").loc[ids]
    votes = {s: t[[f"votes_{s}_{i}" for i in range(N_CLASSES)]].to_numpy(float) for s in ("all", "good")}
    human = {s: np.divide(v, v.sum(axis=1, keepdims=True), out=np.zeros_like(v),
                          where=v.sum(axis=1, keepdims=True) > 0) for s, v in votes.items()}
    share = t["top_share_all"].to_numpy(float)
    tie = t["is_tie_all"].to_numpy(int).astype(bool)
    good_plurality = pd.to_numeric(t["plurality_good_idx"], errors="coerce").fillna(-1).to_numpy(int)
    return TestClips(
        ids=ids,
        refs={EXPRESSED: t["expressed_idx"].to_numpy(int), PLURALITY: t["plurality_all_idx"].to_numpy(int),
              "plurality_good": good_plurality},
        human=human,
        stratum=np.array([stratum(s, b) for s, b in zip(share, tie)]),
        is_tie=tie, top_share=share, n_good=t["n_votes_good"].to_numpy(int),
    )


def load_configs(results: pd.DataFrame, labels: pd.DataFrame) -> tuple[TestClips, dict]:
    configs, clips = {}, None
    cols = ["sample_id", "run", "true_label", "pred_label"] + [f"prob_{c}" for c in range(N_CLASSES)]
    for row in results.to_dict("records"):
        path = Path(row["predictions"])
        df = pd.read_csv(path if path.is_absolute() else results_dir() / path, usecols=cols,
                         dtype={"sample_id": str})
        df = df.sort_values(["run", "sample_id"])
        runs = sorted(df["run"].unique())
        if runs != list(range(N_RUNS)):
            raise ValueError(f"{path}: expected runs 0..{N_RUNS - 1}, found {runs}")
        ids = df.loc[df["run"] == 0, "sample_id"].to_numpy()
        if clips is None:
            clips = load_test_clips(labels, ids)
        n = len(clips.ids)
        if len(df) != N_RUNS * n or not (df["sample_id"].to_numpy().reshape(N_RUNS, n) == clips.ids).all():
            raise ValueError(f"{path}: test clips differ from the other configurations")
        key = (row["target"], row["backbone"], row["probe"], int(row["layer"]))
        cfg = Config(*key, pred=df["pred_label"].to_numpy(int).reshape(N_RUNS, n),
                     probs=df[cols[4:]].to_numpy(float).reshape(N_RUNS, n, N_CLASSES), row=row)
        cfg.probs /= cfg.probs.sum(axis=2, keepdims=True)
        own = clips.refs[own_reference(key[0])]
        if not (df["true_label"].to_numpy(int).reshape(N_RUNS, n) == own).all():
            raise ValueError(f"{path}: stored true labels differ from the {own_reference(key[0])} reference")
        configs[key] = cfg
    return clips, configs


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def mean_sd(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def fmt(mean: float, sd: float, digits: int = 3, sign: bool = False) -> str:
    return f"{mean:{'+' if sign else ''}.{digits}f} ± {sd:.{digits}f}"


def config_label(cfg: Config) -> str:
    probe = {"linear": "Linear", "dense": "Dense", "agg_dense": "Agg-Dense"}[cfg.probe]
    layer = "all states" if cfg.layer < 0 else f"L{cfg.layer}"
    return f"{CFG['backbones'][cfg.backbone]['display_name']} {probe} {layer}"


def pair(a: int, b: int) -> tuple[int, int]:
    return (min(a, b), max(a, b))


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select(clips: TestClips, configs: dict) -> tuple[dict, dict, dict]:
    """Own-reference mean UAR per config, selected layer per (target, backbone, probe), best per target."""
    own_uar = {k: c.uar_runs(clips.refs[own_reference(k[0])]) for k, c in configs.items()}
    selected = {}
    for target in TARGETS:
        for backbone in BACKBONES:
            for probe in PROBES:
                keys = sorted(k for k in configs if k[:3] == (target, backbone, probe))
                if not keys:
                    raise ValueError(f"No results for {target}/{backbone}/{probe}")
                selected[(target, backbone, probe)] = max(keys, key=lambda k: (own_uar[k].mean(), -k[3]))
    best = {t: max((selected[(t, b, p)] for b in BACKBONES for p in PROBES),
                   key=lambda k: own_uar[k].mean()) for t in TARGETS}
    return own_uar, selected, best


def nine(selected: dict, target: str) -> list[tuple]:
    return [selected[(target, b, p)] for b in BACKBONES for p in PROBES]


def equal_config(configs: dict, keys: list[tuple], fn) -> np.ndarray:
    """Per-run mean over configurations: fn(config) returns one value per run."""
    return np.mean([fn(configs[k]) for k in keys], axis=0)


# ---------------------------------------------------------------------------
# Human labels (Section V-A, Table III, Fig. 1)
# ---------------------------------------------------------------------------

def krippendorff_alpha_nominal(counts: np.ndarray) -> float:
    """Nominal Krippendorff's alpha from per-item category counts (variable votes per item)."""
    counts = counts[counts.sum(axis=1) >= 2].astype(float)
    m = counts.sum(axis=1)
    n = m.sum()
    observed = 1.0 - np.sum(counts * (counts - 1) / (m - 1)[:, None]) / n
    totals = counts.sum(axis=0)
    expected = (n * n - totals @ totals) / (n * (n - 1))
    return float(1.0 - observed / expected)


def human_tables(labels: pd.DataFrame) -> dict[str, pd.DataFrame]:
    expressed = labels["expressed_idx"].to_numpy(int)
    reliability = []
    for source in ("all", "good"):
        votes = labels[[f"votes_{source}_{i}" for i in range(N_CLASSES)]].to_numpy(int)
        used = votes.sum(axis=1) >= 2
        plurality = pd.to_numeric(labels[f"plurality_{source}_idx"], errors="coerce").to_numpy()
        reliability.append({
            "votes": source, "ratings": int(votes.sum()),
            "evaluator_accuracy": votes[np.arange(len(votes)), expressed].sum() / votes.sum(),
            "clip_plurality_accuracy": float(np.mean(plurality[used] == expressed[used])),
            "krippendorff_alpha": krippendorff_alpha_nominal(votes),
        })

    strata = [stratum(s, bool(t)) for s, t in zip(labels["top_share_all"], labels["is_tie_all"])]
    match = labels["plurality_all_idx"].to_numpy(int) == expressed
    agreement = pd.DataFrame({"stratum": strata, "match": match}).groupby("stratum")["match"].agg(["size", "mean"])
    agreement = agreement.reindex(STRATA).reset_index().rename(
        columns={"size": "clips", "mean": "plurality_equals_expressed"})

    plurality = labels["plurality_all_idx"].to_numpy(int)
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    np.add.at(confusion, (expressed, plurality), 1)
    confusion_df = pd.DataFrame(confusion, index=CLASS_DISPLAY, columns=CLASS_DISPLAY)

    votes = labels[[f"votes_all_{i}" for i in range(N_CLASSES)]].to_numpy(int)
    clip_pairs, eval_pairs = Counter(), Counter()
    for e, p, v in zip(expressed, plurality, votes):
        if e != p:
            clip_pairs[pair(e, p)] += 1
        for c in range(N_CLASSES):
            if c != e and v[c]:
                eval_pairs[pair(e, c)] += int(v[c])
    pairs = pd.DataFrame([{"class_a": CLASSES[a], "class_b": CLASSES[b], "a": a, "b": b,
                           "evaluator_count": eval_pairs[(a, b)], "clip_count": clip_pairs[(a, b)]}
                          for a in range(N_CLASSES) for b in range(a + 1, N_CLASSES)])
    # Same ranking rule as the paper's human top-10: count, then class names.
    pairs = pairs.sort_values(["evaluator_count", "class_a", "class_b"], ascending=[False, True, True])
    pairs["evaluator_rank"] = np.arange(1, len(pairs) + 1)
    return {"human_reliability": pd.DataFrame(reliability), "human_agreement_strata": agreement,
            "human_confusion_clip_plurality": confusion_df.reset_index(names="expressed"),
            "human_confusion_pairs": pairs.reset_index(drop=True)}


# ---------------------------------------------------------------------------
# Model analyses
# ---------------------------------------------------------------------------

def best_by_target(configs, own_uar, best) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        cfg = configs[best[target]]
        m, s = mean_sd(own_uar[best[target]])
        rows.append({"target": target, "configuration": config_label(cfg), "backbone": cfg.backbone,
                     "probe": cfg.probe, "layer": cfg.layer, "uar_mean": m, "uar_sd": s})
    return pd.DataFrame(rows)


def own_reference_deltas(own_uar) -> pd.DataFrame:
    rows = []
    for (target, backbone, probe, layer), uar in own_uar.items():
        other = own_uar.get((PLURALITY, backbone, probe, layer))
        if target == EXPRESSED and other is not None:
            rows.append({"backbone": backbone, "probe": probe, "layer": layer,
                         "expressed_uar": uar.mean(), "plurality_uar": other.mean(),
                         "plurality_minus_expressed": other.mean() - uar.mean()})
    return pd.DataFrame(rows)


def common_reference(clips, configs, selected) -> pd.DataFrame:
    """Fig. 2(a) and Table V: every target scored against the same references."""
    rows = []
    for target in TARGETS:
        keys = nine(selected, target)
        out = {"target": target}
        for ref in (EXPRESSED, PLURALITY):
            out[f"uar_vs_{ref}_mean"], out[f"uar_vs_{ref}_sd"] = mean_sd(
                equal_config(configs, keys, lambda c: c.uar_runs(clips.refs[ref])))
        for metric in DIST_METRICS:
            out[f"{metric}_mean"], out[f"{metric}_sd"] = mean_sd(
                equal_config(configs, keys, lambda c: c.dist_runs(clips.human["all"], metric)))
        rows.append(out)
    return pd.DataFrame(rows)


def by_stratum(clips, configs, selected) -> pd.DataFrame:
    """Fig. 2(b) (own-reference accuracy) and Fig. 4 (distributional error) per agreement stratum."""
    rows = []
    for target in TARGETS:
        keys = nine(selected, target)
        ref = clips.refs[own_reference(target)]
        for name in STRATA:
            mask = clips.stratum == name
            out = {"target": target, "stratum": name, "n_clips": int(mask.sum())}
            out["accuracy_mean"], out["accuracy_sd"] = mean_sd(
                equal_config(configs, keys, lambda c: c.accuracy_runs(ref, mask)))
            for metric in DIST_METRICS:
                out[f"{metric}_mean"], out[f"{metric}_sd"] = mean_sd(
                    equal_config(configs, keys, lambda c: c.dist_runs(clips.human["all"], metric, mask)))
            rows.append(out)
    return pd.DataFrame(rows)


def brier_matched_comparisons(clips, configs, selected) -> pd.DataFrame:
    """Five-run mean Brier of each target's selected layer, per backbone/probe pair."""
    rows = []
    for backbone in BACKBONES:
        for probe in PROBES:
            out = {"backbone": backbone, "probe": probe}
            for target in TARGETS:
                out[f"brier_{target}"] = configs[selected[(target, backbone, probe)]].dist_runs(
                    clips.human["all"], "brier").mean()
            out["soft_lowest"] = out[f"brier_{DISTRIBUTION}"] == min(out[f"brier_{t}"] for t in TARGETS)
            rows.append(out)
    return pd.DataFrame(rows)


def confusion_overlap(clips, configs, selected, human_pairs: pd.DataFrame, top_k: int = 10):
    """Table VI: overlap of each model's top-10 undirected error pairs with the human evaluator top-10."""
    human_top = {pair(a, b) for a, b in human_pairs.head(top_k)[["a", "b"]].to_numpy()}
    by_run, summary = [], []
    for target in TARGETS:
        for key in nine(selected, target):
            cfg = configs[key]
            ref = clips.refs[own_reference(target)]
            for run, pred in enumerate(cfg.pred):
                counts = Counter(pair(t, p) for t, p in zip(ref, pred) if t != p)
                top = [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][:top_k]
                by_run.append({"target": target, "configuration": config_label(cfg), "run": run,
                               "overlap": len(set(top) & human_top) / top_k})
    by_run = pd.DataFrame(by_run)
    for target, part in by_run.groupby("target", sort=False):
        m, s = mean_sd(part.groupby("run")["overlap"].mean())
        config_means = part.groupby("configuration")["overlap"].mean()
        summary.append({"target": target, "overlap_mean": m, "overlap_sd": s,
                        "config_min": config_means.min(), "config_max": config_means.max(),
                        "min_pairs_any_run": int(round(part["overlap"].min() * top_k))})
    return by_run, pd.DataFrame(summary)


def ambiguous_subsets(clips, configs, selected) -> pd.DataFrame:
    """Table VII: hard and soft recognized training on tied and below-0.60 clips."""
    rows = []
    subsets = {"tied": clips.is_tie, "below_0.6": (~clips.is_tie) & (clips.top_share < 0.6)}
    for name, mask in subsets.items():
        for target in (PLURALITY, DISTRIBUTION):
            keys = nine(selected, target)
            out = {"subset": name, "n_clips": int(mask.sum()), "target": target}
            out["uar_mean"], out["uar_sd"] = mean_sd(
                equal_config(configs, keys, lambda c: c.uar_runs(clips.refs[PLURALITY], mask)))
            for metric in ("kl", "brier"):
                out[f"{metric}_mean"], out[f"{metric}_sd"] = mean_sd(
                    equal_config(configs, keys, lambda c: c.dist_runs(clips.human["all"], metric, mask)))
            rows.append(out)
    return pd.DataFrame(rows)


def matched_soft_hard(clips, configs, selected) -> pd.DataFrame:
    """Soft minus hard at the hard target's selected layer, same backbone/probe/run."""
    deltas = {"uar": [], "kl": [], "brier": []}
    for backbone in BACKBONES:
        for probe in PROBES:
            hard_key = selected[(PLURALITY, backbone, probe)]
            hard, soft = configs[hard_key], configs[(DISTRIBUTION, *hard_key[1:])]
            deltas["uar"].append(soft.uar_runs(clips.refs[PLURALITY]) - hard.uar_runs(clips.refs[PLURALITY]))
            for metric in ("kl", "brier"):
                deltas[metric].append(soft.dist_runs(clips.human["all"], metric)
                                      - hard.dist_runs(clips.human["all"], metric))
    rows = []
    for metric, values in deltas.items():
        m, s = mean_sd(np.mean(values, axis=0))
        rows.append({"metric": metric, "soft_minus_hard_mean": m, "soft_minus_hard_sd": s})
    return pd.DataFrame(rows)


def good_quality(clips, configs, selected) -> pd.DataFrame:
    """Re-score against good-quality votes only (clips with enough good votes), no retraining."""
    mask = clips.n_good >= int(CFG["analysis"]["min_good_votes"])
    rows = []
    for target in TARGETS:
        keys = nine(selected, target)
        out = {"target": target, "n_clips": int(mask.sum())}
        out["uar_mean"], out["uar_sd"] = mean_sd(
            equal_config(configs, keys, lambda c: c.uar_runs(clips.refs["plurality_good"], mask)))
        for metric in ("kl", "brier"):
            out[f"{metric}_mean"], out[f"{metric}_sd"] = mean_sd(
                equal_config(configs, keys, lambda c: c.dist_runs(clips.human["good"], metric, mask)))
        rows.append(out)
    return pd.DataFrame(rows)


def aggregation_vs_single(configs, own_uar) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Table VIII and Fig. 5: agg_dense against the best single-layer probe, plus its layer weights."""
    rows, weights = [], []
    for target in TARGETS:
        for backbone in BACKBONES:
            singles = sorted(k for k in configs if k[:2] == (target, backbone) and k[2] != "agg_dense")
            single = max(singles, key=lambda k: own_uar[k].mean())
            agg = (target, backbone, "agg_dense", -1)
            out = {"target": target, "backbone": backbone, "single": config_label(configs[single])}
            out["single_uar_mean"], out["single_uar_sd"] = mean_sd(own_uar[single])
            out["agg_uar_mean"], out["agg_uar_sd"] = mean_sd(own_uar[agg])
            out["delta_mean"], out["delta_sd"] = mean_sd(own_uar[agg] - own_uar[single])
            rows.append(out)
            row = configs[agg].row
            for layer, (m, s) in enumerate(zip(json.loads(row["agg_weights_mean"]),
                                               json.loads(row["agg_weights_sd"]))):
                weights.append({"target": target, "backbone": backbone, "layer": layer,
                                "weight_mean": m, "weight_sd": s})
    return pd.DataFrame(rows), pd.DataFrame(weights)


# ---------------------------------------------------------------------------
# Paired clip bootstrap (Table IX)
# ---------------------------------------------------------------------------

class _ClipScores:
    """Per-clip quantities of one configuration for fast resampling of whole clips."""

    def __init__(self, cfg: Config, ref: np.ndarray, human: np.ndarray, mask: np.ndarray) -> None:
        self.ref = ref[mask]
        self.correct = (cfg.pred[:, mask] == self.ref).sum(axis=0)       # correct runs per clip
        dist = [distribution_rows(human[mask], q[mask]) for q in cfg.probs]
        self.per_clip = {m: np.mean([d[m] for d in dist], axis=0) for m in ("kl", "brier")}
        self.runs = cfg.pred.shape[0]

    def value(self, metric: str, draw: np.ndarray) -> float:
        if metric == "uar":
            support = np.bincount(self.ref[draw], minlength=N_CLASSES)
            hits = np.bincount(self.ref[draw], weights=self.correct[draw], minlength=N_CLASSES)
            present = support > 0
            return float(np.mean(hits[present] / (support[present] * self.runs)))
        return float(self.per_clip[metric][draw].mean())


BOOTSTRAP_SPECS = [
    # (name, left target, right target, metric, subset, left ref, right ref)
    ("expressed_minus_plurality_own_uar", EXPRESSED, PLURALITY, "uar", "all", EXPRESSED, PLURALITY),
    ("soft_minus_hard_uar", DISTRIBUTION, PLURALITY, "uar", "all", PLURALITY, PLURALITY),
    ("soft_minus_hard_kl", DISTRIBUTION, PLURALITY, "kl", "all", None, None),
    ("soft_minus_hard_brier", DISTRIBUTION, PLURALITY, "brier", "all", None, None),
    ("soft_minus_hard_kl_tied", DISTRIBUTION, PLURALITY, "kl", "tied", None, None),
    ("soft_minus_hard_brier_tied", DISTRIBUTION, PLURALITY, "brier", "tied", None, None),
    ("soft_minus_hard_kl_below_0.6", DISTRIBUTION, PLURALITY, "kl", "below_0.6", None, None),
    ("soft_minus_hard_brier_below_0.6", DISTRIBUTION, PLURALITY, "brier", "below_0.6", None, None),
]


def bootstrap(clips, configs, best) -> pd.DataFrame:
    """Best overall configurations; resample test clips with all five runs of a clip kept together."""
    iterations = int(CFG["analysis"]["bootstrap_iterations"])
    seed = int(CFG["analysis"]["bootstrap_seed"])
    masks = {"all": np.ones(len(clips.ids), bool), "tied": clips.is_tie,
             "below_0.6": (~clips.is_tie) & (clips.top_share < 0.6)}
    rows = []
    for i, (name, left_t, right_t, metric, subset, left_ref, right_ref) in enumerate(BOOTSTRAP_SPECS):
        mask = masks[subset]
        left = _ClipScores(configs[best[left_t]], clips.refs[left_ref or PLURALITY], clips.human["all"], mask)
        right = _ClipScores(configs[best[right_t]], clips.refs[right_ref or PLURALITY], clips.human["all"], mask)
        n = int(mask.sum())
        everything = np.arange(n)
        rng = np.random.default_rng(seed + i)  # clips are indexed in sorted sample-id order
        draws = np.array([left.value(metric, d) - right.value(metric, d)
                          for d in (rng.integers(0, n, n) for _ in range(iterations))])

        def per_run(cfg, ref):
            if metric == "uar":
                return cfg.uar_runs(clips.refs[ref], mask)
            return cfg.dist_runs(clips.human["all"], metric, mask)

        run_delta = (per_run(configs[best[left_t]], left_ref or PLURALITY)
                     - per_run(configs[best[right_t]], right_ref or PLURALITY))
        m, s = mean_sd(run_delta)
        rows.append({"comparison": name, "subset": subset, "n_clips": n,
                     "left": config_label(configs[best[left_t]]), "right": config_label(configs[best[right_t]]),
                     "run_delta_mean": m, "run_delta_sd": s,
                     "estimate": left.value(metric, everything) - right.value(metric, everything),
                     "ci95_low": float(np.quantile(draws, 0.025)),
                     "ci95_high": float(np.quantile(draws, 0.975)),
                     "n_bootstrap": iterations})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def write_report(t: dict[str, pd.DataFrame], path: Path) -> None:
    d = TARGET_DISPLAY
    parts = ["# Label authority: paper numbers\n",
             "Recomputed from `labels.csv` and the saved test predictions by "
             "`python -m label_authority.analysis`. Mean ± sample SD (ddof=1) over five runs.\n"]

    rel = t["human_reliability"]
    parts.append("## Table III(a) — analysis level and reliability\n")
    parts.append(_md_table(["Votes", "Ratings", "Eval. acc.", "Clip acc.", "α"],
                           [[r.votes, f"{r.ratings:,}", f"{r.evaluator_accuracy:.4f}",
                             f"{r.clip_plurality_accuracy:.4f}", f"{r.krippendorff_alpha:.3f}"]
                            for r in rel.itertuples()]))
    parts.append("\n## Table III(b) — agreement strata (all 6,902 clips)\n")
    parts.append(_md_table(["Stratum", "Clips", "Plurality = expressed"],
                           [[r.stratum, f"{r.clips:,}", f"{r.plurality_equals_expressed:.3f}"]
                            for r in t["human_agreement_strata"].itertuples()]))
    top3 = t["human_confusion_pairs"].sort_values("clip_count", ascending=False).head(3)
    parts.append("\nLeading clip-level mismatch pairs: " + ", ".join(
        f"{r.class_a}/{r.class_b} {r.clip_count}" for r in top3.itertuples()))

    parts.append("\n\n## Table IV — best own-reference hard UAR\n")
    parts.append(_md_table(["Training target", "Best configuration", "UAR"],
                           [[d[r.target], r.configuration, fmt(r.uar_mean, r.uar_sd)]
                            for r in t["best_by_target"].itertuples()]))
    deltas = t["own_reference_deltas"]["plurality_minus_expressed"]
    parts.append(f"\nMatched recognized-minus-expressed own-reference deltas: {(deltas < 0).sum()}/"
                 f"{len(deltas)} negative, mean {deltas.mean():+.3f}, range {deltas.min():+.3f} to {deltas.max():+.3f}")

    cr = t["common_reference"]
    parts.append("\n\n## Fig. 2(a) and Table V — common-reference evaluation\n")
    parts.append(_md_table(["Training target", "UAR vs expressed", "UAR vs plurality", "KL", "Brier"],
                           [[d[r.target], fmt(r.uar_vs_expressed_mean, r.uar_vs_expressed_sd),
                             fmt(r.uar_vs_recognized_plurality_mean, r.uar_vs_recognized_plurality_sd),
                             fmt(r.kl_mean, r.kl_sd), fmt(r.brier_mean, r.brier_sd)] for r in cr.itertuples()]))
    bm = t["brier_matched_comparisons"]
    parts.append(f"\nSoft training has the lowest mean Brier in {int(bm['soft_lowest'].sum())}/{len(bm)} "
                 "backbone/probe comparisons.")

    st = t["by_stratum"]
    parts.append("\n\n## Fig. 2(b) — own-reference hard accuracy by agreement stratum\n")
    parts.append(_md_table(["Training target", "Stratum", "n", "Accuracy"],
                           [[d[r.target], r.stratum, r.n_clips, fmt(r.accuracy_mean, r.accuracy_sd)]
                            for r in st.itertuples()]))

    ov = t["confusion_overlap"]
    parts.append("\n\n## Table VI — top-10 model–human confusion-pair overlap\n")
    parts.append(_md_table(["Training target", "Mean ± SD", "Config. range", "Min. pairs"],
                           [[d[r.target], fmt(r.overlap_mean, r.overlap_sd),
                             f"[{r.config_min:.3f}, {r.config_max:.3f}]", r.min_pairs_any_run]
                            for r in ov.itertuples()]))

    parts.append("\n\n## Table VII — ambiguous test subsets\n")
    parts.append(_md_table(["Subset", "n", "Training", "Hard UAR", "KL", "Brier"],
                           [[r.subset, r.n_clips, "Hard" if r.target == PLURALITY else "Soft",
                             fmt(r.uar_mean, r.uar_sd), fmt(r.kl_mean, r.kl_sd), fmt(r.brier_mean, r.brier_sd)]
                            for r in t["ambiguous_subsets"].itertuples()]))
    parts.append("\n\nConfiguration-matched sensitivity (soft − hard at the hard target's layer): " + ", ".join(
        f"{r.metric} {fmt(r.soft_minus_hard_mean, r.soft_minus_hard_sd, sign=True)}"
        for r in t["matched_soft_hard"].itertuples()))
    parts.append("\n\nGood-quality re-scoring:\n\n" + _md_table(
        ["Training target", "n", "UAR vs good plurality", "KL", "Brier"],
        [[d[r.target], r.n_clips, fmt(r.uar_mean, r.uar_sd), fmt(r.kl_mean, r.kl_sd),
          fmt(r.brier_mean, r.brier_sd)] for r in t["good_quality"].itertuples()]))

    parts.append("\n\n## Table VIII — best single layer versus agg_dense\n")
    parts.append(_md_table(["Target", "Backbone", "Best single layer", "Single UAR", "Aggregate UAR", "Paired Δ"],
                           [[d[r.target], CFG["backbones"][r.backbone]["display_name"], r.single,
                             fmt(r.single_uar_mean, r.single_uar_sd), fmt(r.agg_uar_mean, r.agg_uar_sd),
                             fmt(r.delta_mean, r.delta_sd, sign=True)]
                            for r in t["aggregation_vs_single"].itertuples()]))

    parts.append("\n\n## Table IX — paired contrasts on the best individual models\n")
    parts.append(_md_table(["Comparison", "Subset", "Paired-run Δ", "Conditional 95% CI"],
                           [[r.comparison, r.subset, fmt(r.run_delta_mean, r.run_delta_sd, sign=True),
                             f"[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]"]
                            for r in t["bootstrap"].itertuples()]))
    path.write_text("\n".join(parts) + "\n")


def run_analysis() -> dict[str, pd.DataFrame]:
    labels = read_labels()
    results = pd.read_csv(probe_results_path())
    clips, configs = load_configs(results, labels)
    own_uar, selected, best = select(clips, configs)

    tables = human_tables(labels)
    tables["selected_configurations"] = pd.DataFrame(
        [{"target": k[0], "backbone": k[1], "probe": k[2], "layer": k[3],
          "uar_mean": own_uar[k].mean(), "uar_sd": own_uar[k].std(ddof=1)} for k in selected.values()])
    tables["best_by_target"] = best_by_target(configs, own_uar, best)
    tables["own_reference_deltas"] = own_reference_deltas(own_uar)
    tables["common_reference"] = common_reference(clips, configs, selected)
    tables["by_stratum"] = by_stratum(clips, configs, selected)
    tables["brier_matched_comparisons"] = brier_matched_comparisons(clips, configs, selected)
    tables["confusion_overlap_by_run"], tables["confusion_overlap"] = confusion_overlap(
        clips, configs, selected, tables["human_confusion_pairs"])
    tables["ambiguous_subsets"] = ambiguous_subsets(clips, configs, selected)
    tables["matched_soft_hard"] = matched_soft_hard(clips, configs, selected)
    tables["good_quality"] = good_quality(clips, configs, selected)
    tables["aggregation_vs_single"], tables["agg_layer_weights"] = aggregation_vs_single(configs, own_uar)
    tables["bootstrap"] = bootstrap(clips, configs, best)
    return tables


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    tables = run_analysis()
    out = analysis_dir() / "tables"
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False, float_format="%.10g")
    write_report(tables, analysis_dir() / "report.md")
    print(f"Saved {len(tables)} tables to {out} and {analysis_dir() / 'report.md'}")


if __name__ == "__main__":
    main()
