# Copyright (c) 2026 OMRON SINIC X Corporation

"""Masked-reconstruction pre-training of the MAT^3 encoder, logged with trackio.

pixi run train                                  # default config (src/mat3/conf/config.yaml)
pixi run train train.epochs=5 data.batch_size=64
pixi run debug                                  # a few steps, for debugging the pipeline
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from mat3.checkpoint import save_checkpoint
from mat3.masking import NUM_TAXELS, MaskStrategy
from mat3.model import NUM_CONTACTILE_PILLARS, MaskedTactileTransformer
from mat3.normalize import Normalizer

logger = logging.getLogger(__name__)

LogFn = Callable[[dict], None]


def reconstruction_loss(
    model: MaskedTactileTransformer,
    batch: dict[str, torch.Tensor],
    mask_strategy: MaskStrategy,
    max_mask_ratio: float,
    loss_type: str = "reconstruction",
) -> tuple[torch.Tensor, torch.Tensor]:
    """MSE on (normalised) taxel readings and actions. Returns ``(loss_contactile, loss_action)``."""
    pred_contactile, pred_action, _ = model(
        batch, mask_strategy=mask_strategy, max_mask_ratio=max_mask_ratio
    )
    bs, T, _ = batch["observation.contactile"].shape
    target_action = torch.cat([batch["action.position_cmd"], batch["action.rotation_cmd"]], dim=-1)
    target_contactile = batch["observation.contactile"].view(bs, T, NUM_CONTACTILE_PILLARS, 3)[
        :, :, :NUM_TAXELS, :
    ]
    if loss_type == "reconstruction":
        return F.mse_loss(pred_contactile, target_contactile), F.mse_loss(
            pred_action, target_action
        )
    if loss_type == "next_prediction":
        return (
            F.mse_loss(pred_contactile[:, :-1], target_contactile[:, 1:]),
            F.mse_loss(pred_action[:, :-1], target_action[:, 1:]),
        )
    raise ValueError(f"unknown loss_type {loss_type!r}")


class Trainer:
    def __init__(
        self,
        model: MaskedTactileTransformer,
        normalizer: Normalizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        epochs: int = 30,
        lr: float = 1e-4,
        mask_strategy: MaskStrategy | str = MaskStrategy.RANDOM_ALL,
        max_mask_ratio: float = 0.6,
        loss_type: str = "reconstruction",
        amp: bool = True,
        device: str | torch.device | None = None,
        output_dir: str | Path | None = None,
        max_steps_per_epoch: int | None = None,
        log_fn: LogFn | None = None,
        run_config: dict | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.normalizer = normalizer.to(self.device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.epochs, self.loss_type = epochs, loss_type
        self.mask_strategy, self.max_mask_ratio = MaskStrategy(mask_strategy), max_mask_ratio
        self.amp = amp
        self.output_dir = Path(output_dir) if output_dir else None
        self.max_steps = max_steps_per_epoch
        self.log = log_fn or (lambda d: None)
        self.run_config = run_config or {}
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scaler = GradScaler(self.device.type, enabled=amp and self.device.type == "cuda")
        self.best_val_loss = math.inf

    def _prepare(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.normalizer({k: v.to(self.device) for k, v in batch.items()})

    def _losses(self, batch):
        with autocast(device_type=self.device.type, enabled=self.amp):
            lc, la = reconstruction_loss(
                self.model, batch, self.mask_strategy, self.max_mask_ratio, self.loss_type
            )
            return lc + la, lc, la

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total, n = 0.0, 0
        for step, batch in enumerate(
            tqdm(self.train_loader, desc=f"epoch {epoch + 1}/{self.epochs} [train]")
        ):
            if self.max_steps is not None and step >= self.max_steps:
                break
            batch = self._prepare(batch)
            self.optimizer.zero_grad()
            loss, lc, la = self._losses(batch)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += loss.item()
            n += 1
            self.log(
                {
                    "train/loss": loss.item(),
                    "train/loss_contactile": lc.item(),
                    "train/loss_action": la.item(),
                }
            )
        return total / max(n, 1)

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.model.eval()
        sums, n = {"loss": 0.0, "loss_contactile": 0.0, "loss_action": 0.0}, 0
        for step, batch in enumerate(tqdm(self.val_loader, desc="[val]")):
            if self.max_steps is not None and step >= self.max_steps:
                break
            loss, lc, la = self._losses(self._prepare(batch))
            sums["loss"] += loss.item()
            sums["loss_contactile"] += lc.item()
            sums["loss_action"] += la.item()
            n += 1
        return {k: v / max(n, 1) for k, v in sums.items()}

    def _save(self, name: str, epoch: int, loss: float) -> None:
        if self.output_dir is None:
            return
        save_checkpoint(
            self.output_dir / name,
            self.model,
            self.normalizer,
            optimizer_state_dict=self.optimizer.state_dict(),
            epoch=epoch,
            loss=loss,
            config=self.run_config,
        )

    def fit(self) -> dict[str, float]:
        val: dict[str, float] = {}
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            val = self.validate()
            self.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    **{f"val/{k}": v for k, v in val.items()},
                }
            )
            logger.info(
                "epoch %d/%d  train %.6f  val %.6f", epoch + 1, self.epochs, train_loss, val["loss"]
            )
            if val["loss"] < self.best_val_loss:
                self.best_val_loss = val["loss"]
                self._save("best_model.pth", epoch, val["loss"])
        self._save("final_model.pth", self.epochs, val.get("loss", math.nan))
        return val


def set_deterministic() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def run(cfg) -> dict[str, float]:
    """Train from a (hydra / omegaconf) config; see ``src/mat3/conf/config.yaml``."""
    from omegaconf import OmegaConf

    from mat3.data import ContactileData, make_loader
    from mat3.model import MAT3Config

    if cfg.train.deterministic:
        set_deterministic()
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    model = MaskedTactileTransformer(
        MAT3Config.from_dict(OmegaConf.to_container(cfg.model, resolve=True))
    )
    data = ContactileData.load(cfg.data.repo_id, cfg.data.root, cfg.data.revision)
    train_ds, val_ds = data.splits(cfg.data.seq_len, cfg.data.train_ratio)
    normalizer = Normalizer(data.stats, OmegaConf.to_container(cfg.data.normalization))
    train_loader = make_loader(
        train_ds, cfg.data.batch_size, train=True, num_workers=cfg.data.num_workers
    )
    val_loader = make_loader(
        val_ds, cfg.data.batch_size, train=False, num_workers=cfg.data.num_workers
    )

    run_config = OmegaConf.to_container(cfg, resolve=True)
    log_fn = None
    if cfg.logging.trackio:
        import trackio

        trackio.init(
            project=cfg.logging.project,
            name=cfg.logging.run_name,
            config=run_config,
            space_id=cfg.logging.space_id,
        )
        log_fn = trackio.log

    trainer = Trainer(
        model,
        normalizer,
        train_loader,
        val_loader,
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        mask_strategy=cfg.train.mask_strategy,
        max_mask_ratio=cfg.train.max_mask_ratio,
        loss_type=cfg.train.loss_type,
        amp=cfg.train.amp,
        output_dir=cfg.train.output_dir,
        max_steps_per_epoch=cfg.train.max_steps_per_epoch,
        log_fn=log_fn,
        run_config=run_config,
    )
    try:
        return trainer.fit()
    finally:
        if cfg.logging.trackio:
            trackio.finish()


def main() -> None:
    import hydra

    # absolute path: a relative `config_path` is resolved as the module `mat3.conf` in (non-editable)
    # wheel installs, which is not a package
    conf_dir = str(Path(__file__).resolve().parent / "conf")
    hydra.main(config_path=conf_dir, config_name="config", version_base="1.3")(run)()


if __name__ == "__main__":
    main()
