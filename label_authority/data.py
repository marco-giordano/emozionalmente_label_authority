"""PyTorch datasets over the extracted HDF5 hidden states.

Every item carries the clip's hard training label and, for the soft target, the
five-vote probability vector. Labels come from results/labels.csv and are looked
up by sample id, so the same features serve all three label authorities.

LayerDataset       one state per clip; mean-pooled (linear) or full sequence (dense)
LayerStackDataset  all 25 states per clip (agg_dense); read from HDF5 batch by batch
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from label_authority.config import EXPRESSED, N_CLASSES, SOFT_TARGETS

POOL_CHUNK = 128


def _decode(values) -> np.ndarray:
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def target_arrays(labels: pd.DataFrame, sample_ids: np.ndarray, target: str):
    """Hard labels (and soft vote distributions for the soft target) aligned to sample_ids."""
    table = labels.set_index("sample_id").loc[sample_ids]
    hard_col = "expressed_idx" if target == EXPRESSED else "plurality_all_idx"
    hard = table[hard_col].to_numpy(dtype=np.int64)
    soft = None
    if target in SOFT_TARGETS:
        votes = table[[f"votes_all_{i}" for i in range(N_CLASSES)]].to_numpy(dtype=np.float32)
        soft = votes / votes.sum(axis=1, keepdims=True)
    return hard, soft


class _SplitBase(Dataset):
    def __init__(self, h5_path, split: str, labels: pd.DataFrame, target: str) -> None:
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as hf:
            self.indices = np.flatnonzero(_decode(hf["split"][:]) == split)
            self.sample_ids = _decode(hf["sample_ids"][:])[self.indices]
            self.t_max = int(hf.attrs["T_max"])
            self.lengths = np.clip(hf["lengths"][:][self.indices], 1, self.t_max).astype(np.int64)
            self.hidden_size = int(hf.attrs["hidden_size"])
            self.n_states = int(hf.attrs["n_states"])
        self.hard, self.soft = target_arrays(labels, self.sample_ids, target)

    def __len__(self) -> int:
        return len(self.indices)

    def _item(self, features: torch.Tensor, i: int) -> tuple:
        out = (features, int(self.lengths[i]), int(self.hard[i]))
        return out + (torch.from_numpy(self.soft[i]),) if self.soft is not None else out


class LayerDataset(_SplitBase):
    """One hidden state, loaded into memory. pooled=True gives (H,) vectors."""

    def __init__(self, h5_path, layer: int, split: str, labels: pd.DataFrame, target: str,
                 pooled: bool) -> None:
        super().__init__(h5_path, split, labels, target)
        with h5py.File(h5_path, "r") as hf:
            ds = hf[f"layer_{layer}"]
            if pooled:
                feats = np.empty((len(self.indices), self.hidden_size), dtype=np.float32)
                for start in range(0, len(self.indices), POOL_CHUNK):
                    chunk = ds[self.indices[start : start + POOL_CHUNK]].astype(np.float32)
                    for j, n in enumerate(self.lengths[start : start + POOL_CHUNK]):
                        feats[start + j] = chunk[j, :n].mean(axis=0)
            else:
                feats = ds[self.indices].astype(np.float32)
        self.features = torch.from_numpy(feats)

    def __getitem__(self, i: int) -> tuple:
        return self._item(self.features[i], i)


class LayerStackDataset(_SplitBase):
    """All hidden states of a clip, (n_states, T, H) float16, streamed from HDF5.

    Batches are read with one HDF5 call per state and cropped to the longest
    clip in the batch. Each DataLoader worker opens its own file handle.
    """

    def __init__(self, h5_path, split: str, labels: pd.DataFrame, target: str) -> None:
        super().__init__(h5_path, split, labels, target)
        self._hf: h5py.File | None = None

    def __getitems__(self, local: list[int]) -> list[tuple]:
        if self._hf is None:
            self._hf = h5py.File(self.h5_path, "r")
        local = np.asarray(local, dtype=np.int64)
        length = int(self.lengths[local].max()) if len(local) > 1 else self.t_max
        rows = self.indices[local]
        order = np.argsort(rows)
        sorted_rows = rows[order]
        stack = np.empty((len(local), self.n_states, length, self.hidden_size), dtype=np.float16)
        for layer in range(self.n_states):
            ds = self._hf[f"layer_{layer}"]
            out = np.empty((len(local), length, self.hidden_size), dtype=np.float16)
            start = 0
            while start < len(sorted_rows):  # read contiguous runs of rows in one call
                stop = start + 1
                while stop < len(sorted_rows) and sorted_rows[stop] == sorted_rows[stop - 1] + 1:
                    stop += 1
                out[start:stop] = ds[sorted_rows[start] : sorted_rows[stop - 1] + 1, :length]
                start = stop
            stack[order, layer] = out
        feats = torch.from_numpy(stack)
        return [self._item(feats[k], int(i)) for k, i in enumerate(local)]

    def __getitem__(self, i: int) -> tuple:
        return self.__getitems__([i])[0]


class SortishBatchSampler(Sampler[list[int]]):
    """Fixed batches of clips adjacent in the HDF5 file; batch order reshuffled every epoch.

    Adjacent rows make streamed reads mostly contiguous. The epoch counter keeps
    advancing across the runs of one configuration, so each run sees a different
    batch order while the whole configuration stays deterministic.
    """

    def __init__(self, dataset: LayerStackDataset, batch_size: int, shuffle: bool, seed: int = 0) -> None:
        self.order = np.argsort(dataset.indices).astype(np.int64)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        batches = [self.order[s : s + self.batch_size].tolist()
                   for s in range(0, len(self.order), self.batch_size)]
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(batches)
            self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return -(-len(self.order) // self.batch_size)
