"""Draw the paper figures from the tables written by label_authority.analysis.

    fig1_human_confusion.png            Fig. 1   expressed vs. all-vote plurality
    fig2a_hard_uar_cross_evaluation.png Fig. 2a  training target x hard evaluation reference
    fig2b_accuracy_by_agreement.png     Fig. 2b  own-reference accuracy by agreement stratum
    fig3a_kl.png, fig3b_brier.png       Fig. 3   alignment with the five-vote distribution
    fig4_distribution_by_agreement.png  Fig. 4   distributional error by agreement stratum
    fig5_agg_dense_layer_weights.png    Fig. 5   learned agg_dense mixture weights

Run:  python -m label_authority.figures
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "label_authority_mpl"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from label_authority.config import (BACKBONES, CFG, N_CLASSES, PLURALITY, EXPRESSED, TARGET_DISPLAY,  # noqa: E402
                                    TARGETS, analysis_dir)
from label_authority.labels import STRATA, STRATUM_DISPLAY  # noqa: E402

STYLE = "seaborn-v0_8-whitegrid"
DPI = int(CFG["analysis"]["figure_dpi"])
TARGET_COLORS = {"expressed": "#0072B2", "recognized_plurality": "#D55E00",
                 "recognized_distribution": "#009E73"}
SHORT = {"expressed": "Expressed", "recognized_plurality": "Rec. plurality",
         "recognized_distribution": "Rec. distribution"}
plt.rcParams.update({"font.size": 16, "axes.titlesize": 17, "axes.labelsize": 16,
                     "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
                     "figure.titlesize": 18})


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def fig1_human_confusion(tables: Path, out: Path) -> None:
    counts = pd.read_csv(tables / "human_confusion_clip_plurality.csv", index_col=0).astype(float)
    rates = counts.div(counts.sum(axis=1), axis=0)
    diagonal = np.eye(N_CLASSES, dtype=bool)
    off = rates.mask(diagonal)
    annot = counts.astype(int).astype(str) + "\n" + rates.map(lambda v: f"{v:.2f}")
    annot_off = annot.mask(diagonal | (counts == 0), "")
    disagreement = (1.0 - pd.Series(np.diag(rates), index=rates.index)).sort_values()

    with plt.style.context(STYLE):
        fig = plt.figure(figsize=(18, 6.4))
        grid = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.78], wspace=0.35)
        ax_full, ax_off, ax_bar = (fig.add_subplot(grid[0, i]) for i in range(3))
        sns.heatmap(rates, ax=ax_full, annot=annot, fmt="", cmap="YlGnBu", vmin=0, vmax=1,
                    linewidths=0.5, linecolor="white", cbar_kws={"label": "Row-normalized rate"})
        sns.heatmap(off, ax=ax_off, annot=annot_off, fmt="", cmap="rocket_r", vmin=0,
                    vmax=max(float(np.nanmax(off.to_numpy())), 0.01), mask=diagonal,
                    linewidths=0.5, linecolor="white", cbar_kws={"label": "Off-diagonal rate"})
        for ax, title in ((ax_full, "Full matrix"), (ax_off, "Original label disagreements")):
            ax.set_title(title)
            ax.set_xlabel("Recognized label")
            ax.tick_params(axis="x", rotation=35)
            ax.tick_params(axis="y", rotation=0)
        ax_full.set_ylabel("Expressed label")
        ax_off.set_ylabel("")
        ax_bar.barh(disagreement.index, disagreement.values,
                    color=sns.color_palette("flare", len(disagreement)))
        ax_bar.set_xlim(0.0, max(0.55, float(disagreement.max()) + 0.05))
        ax_bar.set_xlabel("Disagreement rate")
        ax_bar.set_title("1 - diagonal")
        for i, value in enumerate(disagreement.values):
            ax_bar.text(value + 0.01, i, f"{value:.2f}", va="center", fontsize=13)
        fig.suptitle("Expressed vs recognized human labels (All, Clip Plurality)", fontweight="bold")
        _save(fig, out / "fig1_human_confusion.png")


def fig2a_hard_matrix(tables: Path, out: Path) -> None:
    data = pd.read_csv(tables / "common_reference.csv").set_index("target").reindex(TARGETS)
    refs = [EXPRESSED, PLURALITY]
    means = pd.DataFrame({r: data[f"uar_vs_{r}_mean"] for r in refs})
    sds = pd.DataFrame({r: data[f"uar_vs_{r}_sd"] for r in refs})
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(6.0, 3.65))
        hm = sns.heatmap(means, ax=ax, cmap="cividis", vmin=0.70, vmax=0.83, linewidths=1.2,
                         linecolor="white", cbar_kws={"label": "Five-run mean UAR", "shrink": 0.87})
        for i, target in enumerate(TARGETS):
            for j, ref in enumerate(refs):
                value = means.loc[target, ref]
                shade = plt.get_cmap("cividis")((value - 0.70) / 0.13)
                luminance = 0.2126 * shade[0] + 0.7152 * shade[1] + 0.0722 * shade[2]
                ax.text(j + 0.5, i + 0.5, f"{value:.3f}\n± {sds.loc[target, ref]:.3f}", ha="center",
                        va="center", fontsize=12.2, color="black" if luminance > 0.52 else "white")
        ax.set_xticklabels([SHORT[r] for r in refs], rotation=0, fontsize=11.2)
        ax.set_yticklabels([SHORT[t] for t in TARGETS], rotation=0, fontsize=11.2)
        ax.set_xlabel("Hard evaluation target", fontsize=11.5)
        ax.set_ylabel("Training target", fontsize=11.5)
        ax.set_title("Hard UAR cross-evaluation", fontsize=12.5, pad=8)
        cbar = hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10.6)
        cbar.set_label("Five-run mean UAR", fontsize=10.6)
        _save(fig, out / "fig2a_hard_uar_cross_evaluation.png")


def _by_stratum_lines(ax, data: pd.DataFrame, column: str) -> None:
    x = np.arange(len(STRATA))
    for target in TARGETS:
        part = data[data["target"] == target].set_index("stratum").reindex(STRATA)
        ax.errorbar(x, part[f"{column}_mean"], yerr=part[f"{column}_sd"], marker="o", linewidth=2.0,
                    markersize=5, capsize=4, capthick=1.2, elinewidth=1.2, ecolor="0.20",
                    color=TARGET_COLORS[target], label=TARGET_DISPLAY[target])


def fig2b_accuracy_by_agreement(tables: Path, out: Path) -> None:
    data = pd.read_csv(tables / "by_stratum.csv")
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 5.1))
        _by_stratum_lines(ax, data, "accuracy")
        ax.set_xticks(np.arange(len(STRATA)))
        ax.set_xticklabels([STRATUM_DISPLAY[s] for s in STRATA], rotation=25, ha="right", fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylim(0.0, 1.0)
        ax.axhline(1.0 / N_CLASSES, color="black", linewidth=0.9, linestyle=":", label="Chance (1/7)")
        ax.set_xlabel("Human agreement stratum")
        ax.set_ylabel("Equal-config mean hard accuracy")
        ax.legend(frameon=True, ncol=2, loc="lower left", fontsize=12, facecolor="white", edgecolor="white")
        ax.set_title("Accuracy by human-recognition uncertainty", fontweight="bold")
        _save(fig, out / "fig2b_accuracy_by_agreement.png")


def fig3_distribution_alignment(tables: Path, out: Path) -> None:
    data = pd.read_csv(tables / "common_reference.csv").set_index("target").reindex(TARGETS)
    panels = [("kl", "KL divergence", "nats (lower better)", 2.00, 0.10, "fig3a_kl.png"),
              ("brier", "Brier score", "score (lower better)", 0.30, 0.018, "fig3b_brier.png")]
    for metric, title, xlabel, floor, pad, name in panels:
        values, errors = data[f"{metric}_mean"].to_numpy(), data[f"{metric}_sd"].to_numpy()
        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(3.3, 2.65))
            y = np.arange(len(TARGETS))
            ax.barh(y, values, xerr=errors, height=0.56, color=[TARGET_COLORS[t] for t in TARGETS],
                    edgecolor="0.2", linewidth=0.35, capsize=3.0,
                    error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "0.15"})
            ax.set_yticks(y)
            ax.set_yticklabels([SHORT[t] for t in TARGETS], fontsize=12.4)
            ax.invert_yaxis()
            ax.set_xlim(0.0, max(floor, float(np.max(values + errors)) + pad))
            ax.set_xlabel(xlabel, fontsize=12.8)
            ax.set_title(title, fontsize=13.8, fontweight="bold", pad=7)
            ax.tick_params(axis="x", labelsize=12.4)
            ax.grid(axis="x", color="0.86", linewidth=0.7)
            ax.grid(axis="y", visible=False)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            fig.subplots_adjust(left=0.43, right=0.98, top=0.86, bottom=0.20)
            _save(fig, out / name)


def fig4_distribution_by_agreement(tables: Path, out: Path) -> None:
    data = pd.read_csv(tables / "by_stratum.csv")
    panels = [("ce", "Cross-entropy", "nats"), ("kl", "KL human to model", "nats"),
              ("js", "Jensen-Shannon", "nats"), ("brier", "Brier to votes", "score")]
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), sharex=True)
        for ax, (metric, title, ylabel) in zip(axes.flat, panels):
            _by_stratum_lines(ax, data, metric)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.24)
        for ax in axes[-1]:
            ax.set_xticks(np.arange(len(STRATA)))
            ax.set_xticklabels([STRATUM_DISPLAY[s] for s in STRATA], rotation=25, ha="right")
            ax.set_xlabel("Human agreement stratum")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(TARGETS), frameon=False)
        fig.suptitle("Distributional error by human-recognition uncertainty", fontweight="bold")
        fig.subplots_adjust(bottom=0.16, top=0.90, hspace=0.34, wspace=0.24)
        _save(fig, out / "fig4_distribution_by_agreement.png")


def fig5_layer_weights(tables: Path, out: Path) -> None:
    weights = pd.read_csv(tables / "agg_layer_weights.csv")
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2), sharex=True, sharey=True)
        for ax, target in zip(axes, TARGETS):
            for backbone in BACKBONES:
                part = weights[(weights["target"] == target) & (weights["backbone"] == backbone)].sort_values("layer")
                x, mean, sd = part["layer"], part["weight_mean"].to_numpy(), part["weight_sd"].to_numpy()
                line, = ax.plot(x, mean, marker="o", markersize=3.5, linewidth=2.0,
                                label=CFG["backbones"][backbone]["display_name"])
                ax.fill_between(x, np.maximum(0.0, mean - sd), mean + sd, color=line.get_color(),
                                alpha=0.18, linewidth=0)
            ax.set_title(TARGET_DISPLAY[target])
            ax.set_xlabel("Layer")
            ax.set_xlim(0, 24)
        axes[0].set_ylim(bottom=0.0)
        axes[0].set_ylabel("Mean aggregation weight")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(BACKBONES), frameon=False)
        fig.suptitle("Agg-Dense learned layer weighting", fontweight="bold")
        fig.subplots_adjust(bottom=0.22, top=0.86, wspace=0.12)
        _save(fig, out / "fig5_agg_dense_layer_weights.png")


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    tables = analysis_dir() / "tables"
    out = analysis_dir() / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for draw in (fig1_human_confusion, fig2a_hard_matrix, fig2b_accuracy_by_agreement,
                 fig3_distribution_alignment, fig4_distribution_by_agreement, fig5_layer_weights):
        draw(tables, out)


if __name__ == "__main__":
    main()
