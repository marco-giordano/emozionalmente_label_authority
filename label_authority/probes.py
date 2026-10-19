"""The three shallow probe families trained on frozen hidden states.

linear     mean-pooled state -> Linear(H,128) -> ReLU -> Linear(128,7)
dense      frame-wise Conv1d(H,256,1) -> ReLU -> Conv1d(256,256,1) -> ReLU
           -> masked mean pool -> Linear(256,7)
agg_dense  softmax-weighted sum of all 25 states at every frame -> dense
"""

from __future__ import annotations

import torch
import torch.nn as nn


def masked_mean(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean over the last (time) axis of (B, C, T), ignoring padded frames."""
    lengths = lengths.to(x.device).long().clamp(min=1, max=x.shape[-1])
    mask = torch.arange(x.shape[-1], device=x.device)[None, :] < lengths[:, None]
    return (x * mask[:, None]).sum(dim=-1) / lengths[:, None].to(x.dtype)


class LinearProbe(nn.Module):
    def __init__(self, n_classes: int, hidden_size: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, n_classes))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(x)  # x: (B, H), already mean-pooled


class DenseProbe(nn.Module):
    def __init__(self, n_classes: int, hidden_size: int, width: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, width, kernel_size=1), nn.ReLU(),
            nn.Conv1d(width, width, kernel_size=1), nn.ReLU(),
        )
        self.classifier = nn.Linear(width, n_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.classifier(masked_mean(self.conv(x.transpose(1, 2)), lengths))  # x: (B, T, H)


class AggDenseProbe(nn.Module):
    def __init__(self, n_classes: int, hidden_size: int, n_states: int) -> None:
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_states))
        self.dense_probe = DenseProbe(n_classes, hidden_size)

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=0)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        weights = self.layer_weights()  # x: (B, L, T, H)
        mixed = None
        for i in range(x.shape[1]):
            term = x[:, i].to(weights.dtype) * weights[i]
            mixed = term if mixed is None else mixed + term
        return self.dense_probe(mixed, lengths)


def build_probe(probe: str, n_classes: int, hidden_size: int, n_states: int) -> nn.Module:
    if probe == "linear":
        return LinearProbe(n_classes, hidden_size)
    if probe == "dense":
        return DenseProbe(n_classes, hidden_size)
    if probe == "agg_dense":
        return AggDenseProbe(n_classes, hidden_size, n_states)
    raise ValueError(f"Unknown probe: {probe}")
