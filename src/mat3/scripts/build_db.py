# Copyright (c) 2026 OMRON SINIC X Corporation

"""Encode the training split with a trained MAT^3 encoder and build a vicinity database.

pixi run build-db --checkpoint outputs/<date>/<time>/models/best_model.pth
# -> outputs/<date>/<time>/db/<pooling>_layer<layer>/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from mat3.checkpoint import load_checkpoint
from mat3.data import ContactileData, make_loader
from mat3.retrieval import DEFAULT_ANN, Pooling, TactileDB, encode_dataset

logger = logging.getLogger(__name__)


def default_db_path(checkpoint: Path, pooling: str, layer: int) -> Path:
    return checkpoint.parent.parent / "db" / f"{pooling}_layer{layer}"


def main(argv=None) -> TactileDB:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--layer", type=int, default=-1, help="encoder layer to tap (-1: last)")
    p.add_argument(
        "--pooling", default=Pooling.GLOBAL_AVERAGE.value, choices=[x.value for x in Pooling]
    )
    p.add_argument("--backend", default="hnsw", choices=["hnsw", "voyager", "basic"])
    p.add_argument("--m", type=int, default=DEFAULT_ANN["m"], help="HNSW graph degree")
    p.add_argument("--ef-construction", type=int, default=DEFAULT_ANN["ef_construction"])
    p.add_argument("--ef-search", type=int, default=DEFAULT_ANN["ef_search"])
    p.add_argument("--metric", default="euclidean")
    p.add_argument(
        "--no-mask-current-action",
        action="store_true",
        help="do not mask the current action token when computing keys (behaviour of the original code)",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="local dataset directory (default: from the checkpoint config / HF cache)",
    )
    a = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, normalizer, ckpt = load_checkpoint(a.checkpoint, device)
    dcfg = ckpt["config"]["data"]
    data = ContactileData.from_config(dcfg, a.data_root)
    train_ds, _ = data.splits(dcfg["seq_len"], dcfg["train_ratio"])
    loader = make_loader(train_ds, a.batch_size, train=False, shuffle=False)

    mask = not a.no_mask_current_action
    keys, actions, _, indices = encode_dataset(model, normalizer, loader, a.layer, a.pooling, mask)
    meta = {
        "checkpoint": str(a.checkpoint),
        "layer": a.layer,
        "pooling": a.pooling,
        "mask_current_action": mask,
        "seq_len": dcfg["seq_len"],
        "dataset": dcfg.get("repo_id", dcfg.get("name")),
        "frame_index": indices.tolist(),
    }
    db = TactileDB.build(
        keys, actions, a.backend, a.metric, a.m, a.ef_construction, a.ef_search, meta=meta
    )
    out = a.out or default_db_path(a.checkpoint, a.pooling, a.layer)
    db.save(out)
    logger.info(
        "built a database with %d entries (%d-dim keys) -> %s", len(db), keys.shape[-1], out
    )
    return db


if __name__ == "__main__":
    main()
