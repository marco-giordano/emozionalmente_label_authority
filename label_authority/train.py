"""Train the probe grid: 3 label targets x 3 backbones x 21 probe configurations.

Each configuration (target, backbone, probe, layer) is trained N_RUNS times
(default five) from different seeds. A run trains for a fixed number of epochs
and keeps the checkpoint that is best on validation: highest accuracy for the
hard targets, lowest soft cross-entropy for the soft target. The kept
checkpoint is then scored once on the test split.

Outputs (under results/):
    probe_results.csv                       one row per configuration
    predictions/<target>/<backbone>/*.csv   per-run test probabilities
    checkpoints/<target>/<backbone>/...     best-validation state of every run

Configurations already in probe_results.csv are skipped, so an interrupted
grid resumes by re-running the same command.

Run:  python -m label_authority.train                       # full grid (189 configs)
      python -m label_authority.train --list                # show what is pending
      python -m label_authority.train --target expressed --backbone wavlm_large --probe dense
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from label_authority.config import (BACKBONES, CFG, LAYERS, N_CLASSES, N_RUNS, PROBES, TARGETS,
                                    SOFT_TARGETS, feature_path, predictions_path,
                                    probe_results_path, results_dir)
from label_authority.data import LayerDataset, LayerStackDataset, SortishBatchSampler
from label_authority.labels import read_labels
from label_authority.metrics import hard_metrics
from label_authority.probes import build_probe

log = logging.getLogger(__name__)
KEY = ["target", "backbone", "probe", "layer"]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def build_loaders(target: str, backbone: str, probe: str, layer: int, labels: pd.DataFrame):
    h5 = feature_path(backbone)
    if not h5.exists():
        raise FileNotFoundError(f"{h5} not found; run python -m label_authority.features first")
    train_cfg = CFG["training"]
    if probe != "agg_dense":
        bs = int(train_cfg["batch_size"])
        sets = [LayerDataset(h5, layer, s, labels, target, pooled=probe == "linear")
                for s in ("train", "val", "test")]
        return [DataLoader(ds, batch_size=bs, shuffle=s == "train", num_workers=0)
                for ds, s in zip(sets, ("train", "val", "test"))]

    bs = int(train_cfg["agg_batch_size"])
    workers = int(train_cfg["agg_num_workers"])
    extra = {"persistent_workers": True, "prefetch_factor": 2} if workers > 0 else {}
    loaders = []
    for split in ("train", "val", "test"):
        ds = LayerStackDataset(h5, split, labels, target)
        sampler = SortishBatchSampler(ds, bs, shuffle=split == "train")
        loaders.append(DataLoader(ds, batch_sampler=sampler, num_workers=workers,
                                  pin_memory=torch.cuda.is_available(), **extra))
    return loaders


def _unpack(batch, device):
    feats, lengths, hard = batch[0].to(device), batch[1].to(device), batch[2].to(device)
    soft = batch[3].to(device, torch.float32) if len(batch) == 4 else None
    return feats, lengths, hard, soft


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    probs, hard, soft_ce, n = [], [], 0.0, 0
    for batch in loader:
        feats, lengths, labels, soft = _unpack(batch, device)
        logits = model(feats, lengths)
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        hard.append(labels.cpu().numpy())
        if soft is not None:
            soft_ce += float(-(soft * F.log_softmax(logits, dim=-1)).sum(dim=-1).sum())
            n += len(soft)
    probs, hard = np.concatenate(probs), np.concatenate(hard)
    out = {"probs": probs, "true": hard, **hard_metrics(hard, probs.argmax(axis=1))}
    if n:
        out["soft_ce"] = soft_ce / n
    return out


def train_run(run: int, probe: str, loaders, n_states: int, hidden_size: int, soft: bool,
              device, checkpoint: Path) -> tuple[dict, list[float] | None]:
    torch.manual_seed(run)
    np.random.seed(run)
    model = build_probe(probe, N_CLASSES, hidden_size, n_states).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(CFG["training"]["lr"]))
    train_loader, val_loader, test_loader = loaders
    best_score, best_state = float("-inf"), None
    for epoch in range(int(CFG["training"]["epochs"])):
        model.train()
        for batch in train_loader:
            feats, lengths, labels, soft_labels = _unpack(batch, device)
            logits = model(feats, lengths)
            if soft:
                loss = -(soft_labels * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            else:
                loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val = evaluate(model, val_loader, device)
        score = -val["soft_ce"] if soft else val["accuracy"]
        if score > best_score:
            best_score, best_state = score, copy.deepcopy(model.state_dict())
        log.info("  run %d epoch %2d  val_acc=%.4f%s", run, epoch + 1, val["accuracy"],
                 f"  val_soft_ce={val['soft_ce']:.4f}" if soft else "")

    model.load_state_dict(best_state)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in best_state.items()}, checkpoint)
    weights = model.layer_weights().detach().cpu().tolist() if hasattr(model, "layer_weights") else None
    return evaluate(model, test_loader, device), weights


# ---------------------------------------------------------------------------
# One configuration
# ---------------------------------------------------------------------------

def run_configuration(target: str, backbone: str, probe: str, layer: int, labels: pd.DataFrame,
                      n_runs: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()
    loaders = build_loaders(target, backbone, probe, layer, labels)
    ds = loaders[2].dataset
    test_ids = ds.sample_ids
    name = "all_states" if layer < 0 else f"layer_{layer:02d}"
    ckpt_dir = results_dir() / "checkpoints" / target / backbone / f"{probe}_{name}"

    runs, weights, frames = [], [], []
    for run in range(n_runs):
        test, w = train_run(run, probe, loaders, ds.n_states, ds.hidden_size, target in SOFT_TARGETS,
                            device, ckpt_dir / f"run_{run}.pt")
        runs.append(test)
        if w is not None:
            weights.append(w)
        frame = pd.DataFrame({"sample_id": test_ids, "run": run, "true_label": test["true"],
                              "pred_label": test["probs"].argmax(axis=1)})
        for c in range(N_CLASSES):
            frame[f"prob_{c}"] = test["probs"][:, c]
        frames.append(frame)
        log.info("[%s/%s/%s/%s] run %d test UAR=%.4f", target, backbone, probe, layer, run, test["uar"])

    pred_path = predictions_path(target, backbone, probe, layer)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(pred_path, index=False, float_format="%.8g")

    def mean_sd(key):
        values = np.array([r[key] for r in runs])
        return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0

    row = {"target": target, "backbone": backbone, "probe": probe, "layer": layer, "n_runs": n_runs}
    for key in ("accuracy", "uar", "macro_f1"):
        row[f"{key}_mean"], row[f"{key}_sd"] = mean_sd(key)
    if weights:
        w = np.asarray(weights)
        row["agg_weights_mean"] = json.dumps(w.mean(axis=0).tolist())
        row["agg_weights_sd"] = json.dumps(w.std(axis=0, ddof=1).tolist() if len(w) > 1 else [0.0] * w.shape[1])
    row["predictions"] = str(pred_path.relative_to(results_dir()))
    row["wall_time_s"] = round(time.time() - start, 1)
    return row


def save_row(row: dict) -> None:
    """Insert or replace one configuration row, writing the CSV atomically."""
    path = probe_results_path()
    table = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=list(row))
    if len(table):
        same = (table[KEY].astype(str) == pd.Series({k: str(row[k]) for k in KEY})).all(axis=1)
        table = table[~same]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    tmp = path.with_suffix(".tmp")
    table.to_csv(tmp, index=False)
    tmp.replace(path)


def work_list(args) -> list[tuple[str, str, str, int]]:
    layers = [int(x) for x in args.layers.split(",")] if args.layers else LAYERS
    work = []
    for target in args.target or TARGETS:
        for backbone in args.backbone or BACKBONES:
            for probe in args.probe or PROBES:
                for layer in ([-1] if probe == "agg_dense" else layers):
                    work.append((target, backbone, probe, layer))
    return work


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=TARGETS, action="append")
    parser.add_argument("--backbone", choices=BACKBONES, action="append")
    parser.add_argument("--probe", choices=PROBES, action="append")
    parser.add_argument("--layers", help="Comma-separated layer indices (default: the ten in the config).")
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--list", action="store_true", help="Print pending configurations and exit.")
    parser.add_argument("--force", action="store_true", help="Retrain configurations already present.")
    args = parser.parse_args()

    done = set()
    if probe_results_path().exists() and not args.force:
        table = pd.read_csv(probe_results_path())
        done = {(r.target, r.backbone, r.probe, int(r.layer)) for r in table.itertuples()
                if (results_dir() / r.predictions).exists()}
    pending = [item for item in work_list(args) if item not in done]
    if args.list:
        for item in pending:
            print("\t".join(map(str, item)))
        return
    log.info("%d configuration(s) pending", len(pending))
    labels = read_labels()
    for i, (target, backbone, probe, layer) in enumerate(pending, start=1):
        log.info("[%d/%d] %s / %s / %s / layer %s", i, len(pending), target, backbone, probe, layer)
        save_row(run_configuration(target, backbone, probe, layer, labels, args.n_runs))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
