"""Study configuration: one YAML file, a few derived constants, and paths."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("LABEL_AUTHORITY_CONFIG", ROOT / "config" / "study.yaml"))

with open(CONFIG_PATH) as f:
    CFG: dict = yaml.safe_load(f)

CLASSES: list[str] = list(CFG["classes"])
N_CLASSES = len(CLASSES)
CLASS_DISPLAY = [c.capitalize() if c != "neutrality" else "Neutral" for c in CLASSES]
BACKBONES: list[str] = list(CFG["backbones"])
PROBES: list[str] = list(CFG["probes"])
LAYERS: list[int] = [int(layer) for layer in CFG["layers"]]
N_STATES = int(CFG["n_states"])
N_RUNS = int(CFG["training"]["n_runs"])

EXPRESSED, PLURALITY, DISTRIBUTION = "expressed", "recognized_plurality", "recognized_distribution"
TARGETS: list[str] = list(CFG["label_targets"])
TARGET_DISPLAY = {
    EXPRESSED: "Expressed",
    PLURALITY: "Recognized plurality",
    DISTRIBUTION: "Recognized distribution",
}
SOFT_TARGETS = {DISTRIBUTION}


def own_reference(target: str) -> str:
    """Hard reference a target is scored against: its own label, plurality for the soft target."""
    return EXPRESSED if target == EXPRESSED else PLURALITY


def _path(key: str, env: str) -> Path:
    override = os.environ.get(env)
    path = Path(override) if override else Path(CFG["paths"][key])
    return path if path.is_absolute() else ROOT / path


def dataset_root() -> Path:
    return _path("dataset_root", "LABEL_AUTHORITY_DATASET")


def features_dir() -> Path:
    return _path("features_dir", "LABEL_AUTHORITY_FEATURES")


def results_dir() -> Path:
    return _path("results_dir", "LABEL_AUTHORITY_RESULTS")


def feature_path(backbone: str) -> Path:
    return features_dir() / f"{backbone}.h5"


def labels_path() -> Path:
    return results_dir() / "labels.csv"


def probe_results_path() -> Path:
    return results_dir() / "probe_results.csv"


def predictions_path(target: str, backbone: str, probe: str, layer: int) -> Path:
    name = "all_states" if layer < 0 else f"layer_{layer:02d}"
    return results_dir() / "predictions" / target / backbone / f"{probe}_{name}.csv"


def analysis_dir() -> Path:
    return results_dir() / "analysis"
