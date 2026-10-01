# Copyright (c) 2026 OMRON SINIC X Corporation

"""Offline evaluation of retrieval: query every validation window and compare the retrieved action
with the ground truth.

    pixi run eval-db --checkpoint outputs/<date>/<time>/models/best_model.pth \
                     --db outputs/<date>/<time>/db/global_average_layer-1

``--target window_mean`` (default) compares against the mean action of the query window, which is
the protocol used for the numbers reported with the original code; ``--target last`` compares
against the action of the last step of the window.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from mat3.checkpoint import load_checkpoint
from mat3.data import ContactileData, make_loader
from mat3.metrics import action_metrics
from mat3.retrieval import TactileDB, encode_dataset

logger = logging.getLogger(__name__)


def evaluate(model, normalizer, db: TactileDB, loader, target: str = "window_mean") -> dict:
    keys, last_actions, window_actions, _ = encode_dataset(
        model,
        normalizer,
        loader,
        db.meta.get("layer", -1),
        db.meta.get("pooling", "global_average"),
        db.meta.get("mask_current_action", True),
    )
    pred = db.query(keys, k=1)[0][:, 0]
    true = window_actions.mean(axis=1) if target == "window_mean" else last_actions
    return {"target": target, **action_metrics(true, pred)}


def main(argv=None) -> dict:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--target", default="window_mean", choices=["window_mean", "last"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--data-root", type=Path, default=None)
    a = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, normalizer, ckpt = load_checkpoint(a.checkpoint, device)
    dcfg = ckpt["config"]["data"]
    data = ContactileData.from_config(dcfg, a.data_root)
    _, val_ds = data.splits(dcfg["seq_len"], dcfg["train_ratio"])
    results = evaluate(
        model,
        normalizer,
        TactileDB.load(a.db),
        make_loader(val_ds, a.batch_size, train=False, shuffle=False),
        a.target,
    )

    out = a.db / f"evaluation_{a.target}.json"
    out.write_text(json.dumps(results, indent=2))
    pos, rot = results["position"], results["rotation"]
    logger.info(
        "N=%d  position RMSE %.3f mm / MAE %.3f mm  rotation RMSE %.3f deg / MAE %.3f deg  -> %s",
        results["num_samples"],
        pos["rmse"],
        pos["mae"],
        rot["rmse_deg"],
        rot["mae_deg"],
        out,
    )
    return results


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    main()
