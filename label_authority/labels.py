"""Build the per-clip label table that defines the three label authorities.

Emozionalmente gives one elicited (expressed) label per clip in samples.csv and
five listener judgements per clip in evaluations.csv, each with an audio-quality
flag. This module turns them into results/labels.csv, one row per clip:

    expressed_idx                 the elicitation condition
    votes_{all,good}_{0..6}       listener vote histogram (all votes / good-quality votes)
    n_votes_{all,good}
    plurality_{all,good}_idx      most-voted class; ties go to the lowest class index
    top_share_{all,good}          share of the winning class
    is_tie_{all,good}             1 when two or more classes share the top count
    tie_indices_{all,good}        the tied classes, "|"-separated

Run:  python -m label_authority.labels
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from label_authority.config import CLASSES, N_CLASSES, dataset_root, labels_path

ALIASES = {"neutral": "neutrality"}
SOURCES = ("all", "good")
STRATA = ["unanimous", "share_0.8", "share_0.6", "tied", "below_0.6"]
STRATUM_DISPLAY = {
    "unanimous": "1.00",
    "share_0.8": "0.80-0.99",
    "share_0.6": "0.60-0.79",
    "tied": "Tie",
    "below_0.6": "<0.60",
}


def class_index(raw: str) -> int | None:
    label = str(raw).strip().lower()
    label = ALIASES.get(label, label)
    return CLASSES.index(label) if label in CLASSES else None


def sample_id(file_name: str) -> str:
    return f"emozionalmente::{Path(file_name).stem}"


def stratum(top_share: float, is_tie: bool) -> str:
    """Mutually exclusive listener-agreement strata used throughout the paper."""
    if top_share >= 1.0:
        return "unanimous"
    if top_share >= 0.8:
        return "share_0.8"
    if top_share >= 0.6:
        return "share_0.6"
    return "tied" if is_tie else "below_0.6"


def _plurality_fields(votes: np.ndarray, source: str) -> dict:
    total = int(votes.sum())
    if total == 0:
        return {f"n_votes_{source}": 0, f"plurality_{source}_idx": "", f"top_share_{source}": "",
                f"is_tie_{source}": "", f"tie_indices_{source}": ""}
    winners = np.flatnonzero(votes == votes.max())
    return {
        f"n_votes_{source}": total,
        f"plurality_{source}_idx": int(winners[0]),
        f"top_share_{source}": float(votes.max()) / total,
        f"is_tie_{source}": int(len(winners) > 1),
        f"tie_indices_{source}": "|".join(str(i) for i in winners),
    }


def build_labels(samples_csv: Path, evaluations_csv: Path) -> pd.DataFrame:
    samples = pd.read_csv(samples_csv, dtype=str)
    evaluations = pd.read_csv(evaluations_csv, dtype=str)
    votes: dict[str, dict[str, Counter]] = {s: defaultdict(Counter) for s in SOURCES}
    for row in evaluations.itertuples(index=False):
        idx = class_index(row.emotion_recognized)
        if idx is None:
            continue
        votes["all"][row.file_name][idx] += 1
        if str(row.audio_quality).strip().lower() == "good":
            votes["good"][row.file_name][idx] += 1

    rows = []
    for row in samples.itertuples(index=False):
        expressed = class_index(row.emotion_expressed)
        if expressed is None:
            raise ValueError(f"Unknown expressed label {row.emotion_expressed!r} for {row.file_name}")
        out = {"sample_id": sample_id(row.file_name), "file_name": row.file_name,
               "speaker_id": row.actor, "expressed_idx": expressed}
        for source in SOURCES:
            hist = np.array([votes[source][row.file_name][i] for i in range(N_CLASSES)])
            out.update({f"votes_{source}_{i}": int(hist[i]) for i in range(N_CLASSES)})
            out.update(_plurality_fields(hist, source))
        rows.append(out)
    return pd.DataFrame(rows)


def read_labels(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or labels_path(), dtype={"sample_id": str, "tie_indices_all": str,
                                                     "tie_indices_good": str})


def main() -> None:
    meta = dataset_root() / "metadata"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples-csv", type=Path, default=meta / "samples.csv")
    parser.add_argument("--evaluations-csv", type=Path, default=meta / "evaluations.csv")
    parser.add_argument("--out", type=Path, default=labels_path())
    args = parser.parse_args()
    labels = build_labels(args.samples_csv, args.evaluations_csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.out, index=False)
    print(f"Saved {args.out} ({len(labels)} clips, {int(labels['n_votes_all'].sum())} valid votes)")


if __name__ == "__main__":
    main()
