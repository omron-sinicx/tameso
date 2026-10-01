# Copyright (c) 2026 OMRON SINIC X Corporation

"""Convert the original (LeRobot v1.6) Contactile dataset on the Hugging Face Hub to LeRobot v3.0.

The dataset on the Hub (e.g. ``omron-sinicx/contactile_200_lota``) was recorded with LeRobot
codebase v1.6 (``meta_data/`` + an arrow ``train/`` split), which recent LeRobot versions cannot
read. :meth:`mat3.data.ContactileData.load` calls this automatically when the Hub repository has
no v3.0 files; it

1. downloads ``meta_data/`` and ``train/`` (videos are not needed by MAT^3),
2. rewrites every contiguous run of frames as one LeRobot v3.0 episode (frame order is preserved,
   episodes are renumbered in order of appearance),
3. copies the *original* normalisation statistics into ``meta/stats.json`` so that models trained
   on either version see identical inputs.

The command line writes the v3.0 copy to a directory, e.g. to upload it to the Hub::

    python -m mat3.data.convert --repo-id omron-sinicx/contactile_200_lota --revision v1.6 --out v3/contactile_200_lota
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
from huggingface_hub import snapshot_download
from safetensors.numpy import load_file

logger = logging.getLogger(__name__)

TASK = "peg-in-hole insertion with a Contactile tactile sensor"
SKIP_COLUMNS = {"timestamp", "frame_index", "episode_index", "index", "next.done"}


def download_raw(repo_id: str, local_dir: Path | None = None, revision: str | None = None) -> Path:
    """Download the v1.6 files (without videos; to the HF cache by default) and return their path."""
    path = snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
        allow_patterns=["meta_data/*", "train/data-*.arrow", "train/*.json"],
    )
    return Path(path)


def read_arrow_split(split_dir: Path) -> pa.Table:
    """Read a `datasets.save_to_disk` split without `datasets` (its v1.6 features are unknown today)."""
    state = json.loads((split_dir / "state.json").read_text())
    tables = []
    for f in state["_data_files"]:
        with pa.memory_map(str(split_dir / f["filename"])) as src:
            tables.append(pa.ipc.open_stream(src).read_all())
    return pa.concat_tables(tables)


def column_to_numpy(table: pa.Table, name: str) -> np.ndarray:
    col = table.column(name).combine_chunks()
    if pa.types.is_list(col.type) or pa.types.is_fixed_size_list(col.type):
        return np.asarray(col.flatten().to_numpy(zero_copy_only=False)).reshape(len(col), -1)
    return col.to_numpy(zero_copy_only=False)


def contiguous_runs(episode_index: np.ndarray) -> list[tuple[int, int]]:
    """[(start, stop), ...] of maximal runs with a constant episode index."""
    change = np.flatnonzero(np.diff(episode_index) != 0) + 1
    bounds = np.concatenate([[0], change, [len(episode_index)]])
    return list(zip(bounds[:-1].tolist(), bounds[1:].tolist()))


def convert(raw_dir: Path, out_dir: Path, repo_id: str, overwrite: bool = False) -> Path:
    from lerobot.datasets.io_utils import load_stats, write_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info = json.loads((raw_dir / "meta_data" / "info.json").read_text())
    table = read_arrow_split(raw_dir / "train")
    columns = {
        name: column_to_numpy(table, name)
        for name in table.column_names
        if name not in SKIP_COLUMNS and not name.startswith("observation.image")
    }
    episode_index = column_to_numpy(table, "episode_index")
    runs = contiguous_runs(episode_index)

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists (use --overwrite)")
        shutil.rmtree(out_dir)

    features = {
        name: {"dtype": "float32", "shape": (arr.shape[1],), "names": None}
        for name, arr in columns.items()
    }
    ds = LeRobotDataset.create(
        repo_id=repo_id, fps=info["fps"], features=features, root=out_dir, use_videos=False
    )
    for start, stop in runs:
        for i in range(start, stop):
            ds.add_frame({**{k: v[i].astype(np.float32) for k, v in columns.items()}, "task": TASK})
        ds.save_episode()
    ds.finalize()

    # keep the original statistics (they define the normalisation used by the paper models)
    raw_stats: dict[str, dict[str, np.ndarray]] = {}
    for k, v in load_file(raw_dir / "meta_data" / "stats.safetensors").items():
        key, stat = k.rsplit("/", 1)
        raw_stats.setdefault(key, {})[stat] = v
    stats = load_stats(out_dir) or {}
    for key in columns:
        if key in raw_stats:
            stats.setdefault(key, {}).update(
                {s: raw_stats[key][s] for s in ("mean", "std", "min", "max")}
            )
    write_stats(stats, out_dir)

    (out_dir / "meta" / "source.json").write_text(
        json.dumps(
            {
                "source_repo_id": repo_id,
                "source_codebase_version": info.get("codebase_version"),
                "source_episode_index": [int(episode_index[s]) for s, _ in runs],
            },
            indent=2,
        )
    )
    logger.info("converted %d frames / %d episodes -> %s", len(episode_index), len(runs), out_dir)
    return out_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo-id", default="omron-sinicx/contactile_200_lota")
    p.add_argument("--revision", default="v1.6", help="revision holding the LeRobot v1.6 files")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    convert(
        download_raw(a.repo_id, revision=a.revision),
        a.out,
        repo_id=a.repo_id,
        overwrite=a.overwrite,
    )


if __name__ == "__main__":
    main()
