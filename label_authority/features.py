"""Extract all frozen hidden states of one backbone for every Emozionalmente clip.

The speaker-independent train/val/test split is assigned here, once, and stored
in the HDF5 file so every label authority, backbone and probe sees the same
clips. The assignment is greedy and deterministic (seed in config/study.yaml):
speakers are visited from largest to smallest and each goes to the split that
brings sample counts and per-class counts closest to the 60/20/20 targets.
On Emozionalmente it yields 4,142 / 1,380 / 1,380 clips.

HDF5 layout (features/<backbone>.h5):
    split        (N,)              "train" | "val" | "test"
    labels       (N,)   int8       expressed class index
    speaker_ids  (N,)   str
    sample_ids   (N,)   str        "emozionalmente::<file stem>"
    lengths      (N,)   int32      valid frames before padding
    layer_{i}    (N, T_max, H)     float16, i = 0..24

Extraction keeps every clip's hidden states in host memory before writing, so
it needs a high-RAM machine (order of 100 GB for Emozionalmente).

Run:  python -m label_authority.features --backbone xlsr_300m
"""

from __future__ import annotations

import argparse
import logging

import h5py
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel

from label_authority.config import BACKBONES, CFG, N_CLASSES, dataset_root, feature_path
from label_authority.labels import class_index, sample_id

log = logging.getLogger(__name__)
SPLITS = ("train", "val", "test")
SAMPLE_RATE = 16_000
BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# Clips and speaker-independent split
# ---------------------------------------------------------------------------

def load_clips() -> list[dict]:
    root = dataset_root()
    samples = pd.read_csv(root / "metadata" / "samples.csv", dtype=str)
    samples.columns = samples.columns.str.strip()
    clips = []
    for row in samples.itertuples(index=False):
        path = root / "audio" / row.file_name
        label = class_index(row.emotion_expressed)
        if label is None or not path.exists():
            continue
        clips.append({"path": path, "label": label, "speaker_id": str(row.actor),
                      "sample_id": sample_id(row.file_name)})
    log.info("Loaded %d clips from %s", len(clips), root)
    return clips


def _split_score(samples, labels, speakers, target_samples, target_labels) -> float:
    sample_err = np.sum(((samples - target_samples) / np.maximum(target_samples, 1.0)) ** 2)
    label_err = np.mean(((labels - target_labels) / np.maximum(target_labels, 1.0)) ** 2)
    return float(sample_err + label_err + 100.0 * np.sum(speakers == 0))


def assign_splits(clips: list[dict]) -> None:
    """Greedy speaker-level assignment; writes clip["split"] in place."""
    fractions = np.asarray(CFG["split"]["fractions"], dtype=float)
    rng = np.random.default_rng(int(CFG["split"]["seed"]))

    speaker_clips: dict[str, list[int]] = {}
    for i, clip in enumerate(clips):
        speaker_clips.setdefault(clip["speaker_id"], []).append(i)
    speaker_hist = {s: np.bincount([clips[i]["label"] for i in idx], minlength=N_CLASSES)
                    for s, idx in speaker_clips.items()}
    totals = sum(speaker_hist.values())
    target_samples = fractions * totals.sum()
    target_labels = fractions[:, None] * totals[None, :]

    speakers = sorted(speaker_clips)
    shuffled = [str(s) for s in rng.permutation(np.asarray(speakers, dtype=object)).tolist()]
    rank = {s: r for r, s in enumerate(shuffled)}
    ordered = sorted(speakers, key=lambda s: (-int(speaker_hist[s].sum()), rank[s]))

    samples = np.zeros(3)
    labels = np.zeros((3, N_CLASSES))
    n_speakers = np.zeros(3)
    for speaker in ordered:
        hist = speaker_hist[speaker].astype(float)
        scores = []
        for k in range(3):
            s, l, n = samples.copy(), labels.copy(), n_speakers.copy()
            s[k] += hist.sum()
            l[k] += hist
            n[k] += 1
            scores.append(_split_score(s, l, n, target_samples, target_labels))
        best = int(np.argmin(scores))
        samples[best] += hist.sum()
        labels[best] += hist
        n_speakers[best] += 1
        for i in speaker_clips[speaker]:
            clips[i]["split"] = SPLITS[best]


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class Backbone:
    def __init__(self, backbone: str, device: torch.device) -> None:
        name = CFG["backbones"][backbone]["name"]
        self.dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.device = device
        self.processor = AutoFeatureExtractor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, dtype=self.dtype).eval().to(device)

    def load(self, path) -> torch.Tensor:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wave = torch.from_numpy(data.mean(axis=1))
        if sr != SAMPLE_RATE:
            wave = torchaudio.functional.resample(wave, sr, SAMPLE_RATE)
        inputs = self.processor(wave.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        return inputs.input_values.squeeze(0)

    @torch.no_grad()
    def hidden_states(self, waves: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (B, n_states, T, H) float16 hidden states and valid frame lengths."""
        raw = torch.tensor([w.shape[0] for w in waves])
        padded = torch.zeros(len(waves), int(raw.max()))
        mask = torch.zeros(len(waves), int(raw.max()), dtype=torch.long)
        for i, w in enumerate(waves):
            padded[i, : w.shape[0]] = w
            mask[i, : w.shape[0]] = 1
        out = self.model(padded.to(self.device, self.dtype), attention_mask=mask.to(self.device),
                         output_hidden_states=True)
        hidden = torch.stack([h.cpu().to(torch.float16) for h in out.hidden_states], dim=1)
        lengths = self.model._get_feat_extract_output_lengths(raw).long()
        return hidden, lengths.clamp(min=1, max=hidden.shape[2])


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract(backbone: str) -> None:
    out_path = feature_path(backbone)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clips = load_clips()
    assign_splits(clips)
    for split in SPLITS:
        part = [c for c in clips if c["split"] == split]
        log.info("%-5s %5d clips, %3d speakers", split, len(part), len({c["speaker_id"] for c in part}))

    model = Backbone(backbone, device)
    feats: list[torch.Tensor] = [None] * len(clips)  # type: ignore[list-item]
    lengths = np.zeros(len(clips), dtype=np.int32)
    for start in tqdm(range(0, len(clips), BATCH_SIZE), desc=f"{backbone} inference"):
        batch = clips[start : start + BATCH_SIZE]
        hidden, valid = model.hidden_states([model.load(c["path"]) for c in batch])
        for j in range(len(batch)):
            lengths[start + j] = int(valid[j])
            feats[start + j] = hidden[j, :, : int(valid[j])].clone()
    del model

    t_max = int(lengths.max())
    n_states, hidden_size = int(feats[0].shape[0]), int(feats[0].shape[-1])
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    with h5py.File(tmp_path, "w") as hf:
        hf.attrs.update({"backbone": backbone, "n_states": n_states,
                         "hidden_size": hidden_size, "T_max": t_max})
        text = h5py.string_dtype()
        hf.create_dataset("split", data=np.array([c["split"] for c in clips], dtype=text))
        hf.create_dataset("labels", data=np.array([c["label"] for c in clips], dtype=np.int8))
        hf.create_dataset("speaker_ids", data=np.array([c["speaker_id"] for c in clips], dtype=text))
        hf.create_dataset("sample_ids", data=np.array([c["sample_id"] for c in clips], dtype=text))
        hf.create_dataset("lengths", data=lengths)
        for layer in tqdm(range(n_states), desc="writing states"):
            data = np.zeros((len(clips), t_max, hidden_size), dtype=np.float16)
            for i, f in enumerate(feats):
                data[i, : lengths[i]] = f[layer].numpy()
            hf.create_dataset(f"layer_{layer}", data=data, compression="gzip",
                              chunks=(1, t_max, hidden_size))
    tmp_path.replace(out_path)
    log.info("Saved %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=BACKBONES, action="append",
                        help="Backbone(s) to extract; default: all three.")
    args = parser.parse_args()
    for backbone in args.backbone or BACKBONES:
        extract(backbone)


if __name__ == "__main__":
    main()
