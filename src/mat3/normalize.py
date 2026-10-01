# Copyright (c) 2026 OMRON SINIC X Corporation

"""Feature-wise normalisation.

The normalisation follows the ``Normalize`` module of LeRobot (v1.x), Copyright 2024 The
HuggingFace Inc. team, Apache License 2.0 (see THIRD_PARTY_NOTICES.md); it is re-implemented here
so that models trained with LeRobot v1.6 and with recent LeRobot versions see identical inputs.
"""

from __future__ import annotations

import torch
from torch import nn

# Only observations are normalised; actions are fed to the action token in their raw scale.
DEFAULT_MODES = {
    "observation.eef.position": "mean_std",
    "observation.eef.rotation_ortho6": "min_max",
    "observation.contactile": "mean_std",
    "observation.vive_tracker_pose": "mean_std",
    "observation.ft": "mean_std",
}
_STATS_FOR_MODE = {"mean_std": ("mean", "std"), "min_max": ("min", "max")}


class Normalizer(nn.Module):
    """``mean_std``: ``(x - mean) / (std + 1e-8)``; ``min_max``: map ``[min, max]`` to ``[-1, 1]``."""

    def __init__(
        self, stats: dict[str, dict[str, torch.Tensor]], modes: dict[str, str] | None = None
    ):
        super().__init__()
        self.modes = dict(modes or DEFAULT_MODES)
        for key, mode in self.modes.items():
            if mode not in _STATS_FOR_MODE:
                raise ValueError(f"unknown normalisation mode {mode!r} for {key}")
            for s in _STATS_FOR_MODE[mode]:
                value = torch.as_tensor(stats[key][s], dtype=torch.float32).clone()
                self.register_buffer(self._name(key, s), value)

    @staticmethod
    def _name(key: str, stat: str) -> str:
        return f"{key.replace('.', '_')}__{stat}"

    def stats(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            key: {s: getattr(self, self._name(key, s)).cpu() for s in _STATS_FOR_MODE[mode]}
            for key, mode in self.modes.items()
        }

    @torch.no_grad()
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = dict(batch)
        for key, mode in self.modes.items():
            if key not in batch:
                continue
            if mode == "mean_std":
                mean, std = (
                    getattr(self, self._name(key, "mean")),
                    getattr(self, self._name(key, "std")),
                )
                batch[key] = (batch[key] - mean) / (std + 1e-8)
            else:
                lo, hi = (
                    getattr(self, self._name(key, "min")),
                    getattr(self, self._name(key, "max")),
                )
                batch[key] = (batch[key] - lo) / (hi - lo + 1e-8)
                batch[key] = batch[key] * 2 - 1
        return batch
