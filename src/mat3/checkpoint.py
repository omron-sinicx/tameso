# Copyright (c) 2026 OMRON SINIC X Corporation

"""Checkpoint I/O for MAT^3 (and loading checkpoints of the original research code)."""

from __future__ import annotations

from pathlib import Path

import torch

from mat3.model import MaskedTactileTransformer, MAT3Config
from mat3.normalize import Normalizer

FORMAT = "mat3/v1"


def save_checkpoint(
    path: str | Path, model: MaskedTactileTransformer, normalizer: Normalizer, **extra
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FORMAT,
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "normalizer": {"modes": normalizer.modes, "stats": normalizer.stats()},
            **extra,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Returns ``(model, normalizer, checkpoint_dict)``; the model is in eval mode."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} checkpoint (use `load_legacy_checkpoint`)")
    model = MaskedTactileTransformer(MAT3Config.from_dict(ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state_dict"])
    normalizer = Normalizer(ckpt["normalizer"]["stats"], ckpt["normalizer"]["modes"])
    return model.to(device).eval(), normalizer.to(device), ckpt


def load_legacy_checkpoint(
    run_dir: str | Path, device: str | torch.device = "cpu", name: str = "best_model.pth"
):
    """Load a run of the original code (``<run>/.hydra/config.yaml`` + ``<run>/models/*.pth``).

    Returns ``(model, normalizer, omegaconf_config)``.
    """
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    # the original code always used causal attention
    model_cfg = {**OmegaConf.to_container(cfg.model, resolve=True), "causal": True}
    model = MaskedTactileTransformer(MAT3Config.from_dict(model_cfg))
    ckpt = torch.load(run_dir / "models" / name, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    stats: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in load_file(run_dir / "models" / "stats.safetensors").items():
        key, stat = k.rsplit("/", 1)
        stats.setdefault(key, {})[stat] = v
    modes = OmegaConf.to_container(cfg.dataset.training_data_info.input_normalization_modes)
    normalizer = Normalizer(stats, modes)
    return model.to(device).eval(), normalizer.to(device), cfg
