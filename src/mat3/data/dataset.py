# Copyright (c) 2026 OMRON SINIC X Corporation

"""Sliding-window datasets on top of a LeRobot v3.0 dataset.

A sample at frame ``i`` is the window of the ``seq_len`` most recent frames ``[i - seq_len + 1, i]``
for every key in :data:`SEQUENCE_KEYS`. Frames before the start of an episode are filled with the
episode's first frame (LeRobot's ``delta_timestamps`` padding).

Train / validation splits are *contiguous frame ranges* (``[0, N * train_ratio)`` and the rest),
as in the original research code (``split="train[:N]"`` in LeRobot v1.6). Inside a split, an
episode is a maximal run of frames with the same ``episode_index``; in particular the episode cut
by the split boundary starts anew in the validation split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

SEQUENCE_KEYS = (
    "observation.contactile",
    "action.position_cmd",
    "action.rotation_cmd",
    "observation.ft",
    "observation.eef.position",
    "observation.eef.rotation_ortho6",
    "observation.vive_tracker_pose",
)


class WindowDataset(Dataset):
    """Past-only windows over the frame range ``[start, stop)`` of in-memory columns."""

    def __init__(
        self,
        columns: dict[str, torch.Tensor],
        episode_index: torch.Tensor,
        start: int,
        stop: int,
        seq_len: int,
    ):
        self.columns = columns
        self.start, self.stop, self.seq_len = start, stop, seq_len
        ep = episode_index[start:stop]
        # start (relative to `start`) of the run each frame belongs to
        new_run = torch.ones(len(ep), dtype=torch.bool)
        new_run[1:] = ep[1:] != ep[:-1]
        run_id = torch.cumsum(new_run.long(), 0) - 1
        run_start = torch.nonzero(new_run).squeeze(1)
        self.frame_run_start = run_start[run_id]
        self.episode_index = run_id  # renumbered per split, like LeRobot v1.6
        self.episode_data_index = {
            "from": run_start,
            "to": torch.cat([run_start[1:], torch.tensor([len(ep)])]),
        }
        self._offsets = torch.arange(-(seq_len - 1), 1)

    def __len__(self) -> int:
        return self.stop - self.start

    def window(self, i: int) -> torch.Tensor:
        """Absolute frame indices of the window ending at split-relative frame ``i``."""
        return self.start + torch.clamp(i + self._offsets, min=int(self.frame_run_start[i]))

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = self.window(i)
        item = {k: self.columns[k][idx] for k in SEQUENCE_KEYS}
        item["index"] = torch.tensor(self.start + i)
        item["episode_index"] = self.episode_index[i]
        return item


@dataclass
class ContactileData:
    """Frames of a LeRobot v3.0 dataset held in memory (the whole dataset is ~20 MB)."""

    columns: dict[str, torch.Tensor]
    episode_index: torch.Tensor
    stats: dict[str, dict[str, torch.Tensor]]
    fps: int

    root: Path | None = None

    @classmethod
    def load(
        cls,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
    ) -> ContactileData:
        """Load a dataset by its Hugging Face Hub id; files are downloaded on first use.

        The dataset is cached as LeRobot v3.0 under ``root`` (default:
        ``$HF_LEROBOT_HOME/<repo_id>``). If the Hub repository only provides the original LeRobot
        v1.6 files, they are downloaded and converted into the cache (see :mod:`mat3.data.convert`).
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils.constants import HF_LEROBOT_HOME

        cache = root is None
        root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        if not (root / "meta" / "info.json").exists() and not hub_has_v3(repo_id, revision):
            from mat3.data.convert import convert, download_raw

            if not cache and root.exists() and any(root.iterdir()):
                raise FileExistsError(
                    f"{root} is not a LeRobot v3.0 dataset and the Hub repository {repo_id} has no "
                    "v3.0 files; use an empty / new directory (or no `root`) to convert it there"
                )
            logger.info("%s has no LeRobot v3.0 files; converting the v1.6 files", repo_id)
            convert(download_raw(repo_id, revision=revision), root, repo_id=repo_id, overwrite=True)
        # an explicit revision skips LeRobot's lookup of a `v3.x` version tag on the Hub
        ds = LeRobotDataset(repo_id, root=root, revision=revision or "main", download_videos=False)
        hf = ds.hf_dataset.with_format("torch", columns=[*SEQUENCE_KEYS, "episode_index"])[:]
        columns = {k: hf[k].to(torch.float32) for k in SEQUENCE_KEYS}
        stats = {
            k: {s: torch.as_tensor(np.asarray(v), dtype=torch.float32) for s, v in st.items()}
            for k, st in ds.meta.stats.items()
            if isinstance(st, dict)
        }
        logger.info(
            "loaded %s (%s): %d frames, %d episodes",
            repo_id,
            root,
            len(hf["episode_index"]),
            ds.meta.total_episodes,
        )
        return cls(
            columns=columns,
            episode_index=hf["episode_index"].long(),
            stats=stats,
            fps=ds.meta.fps,
            root=root,
        )

    @classmethod
    def from_config(cls, data_cfg: dict, root: str | Path | None = None) -> ContactileData:
        """``data`` section of a training config (as stored in checkpoints)."""
        if "repo_id" not in data_cfg:  # configs written before `data.repo_id` existed
            return cls.load(data_cfg["name"], root or Path(data_cfg["root"]) / data_cfg["name"])
        return cls.load(data_cfg["repo_id"], root or data_cfg.get("root"), data_cfg.get("revision"))

    def __len__(self) -> int:
        return len(self.episode_index)

    def splits(self, seq_len: int, train_ratio: float = 0.8) -> tuple[WindowDataset, WindowDataset]:
        n_train = int(len(self) * train_ratio)
        return (
            WindowDataset(self.columns, self.episode_index, 0, n_train, seq_len),
            WindowDataset(self.columns, self.episode_index, n_train, len(self), seq_len),
        )


def hub_has_v3(repo_id: str, revision: str | None = None) -> bool:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import (
        HfHubHTTPError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    api = HfApi()
    try:
        api.repo_info(repo_id, repo_type="dataset", revision=revision)
    except (RepositoryNotFoundError, RevisionNotFoundError) as e:
        raise FileNotFoundError(
            f"dataset {repo_id!r} (revision {revision or 'main'}) not found on the Hugging Face Hub; "
            "check `data.repo_id` / `data.revision`, or log in (`hf auth login`) for private datasets"
        ) from e
    except (HfHubHTTPError, OSError):  # offline / transient errors: let LeRobot report them
        return True
    try:
        return api.file_exists(repo_id, "meta/info.json", repo_type="dataset", revision=revision)
    except (HfHubHTTPError, OSError):  # offline / transient errors: let LeRobot report them
        return True


def make_loader(
    ds: Dataset, batch_size: int, train: bool, num_workers: int = 0, shuffle: bool | None = None
):
    """Loader settings of the original code: both splits are shuffled, only train drops the last batch."""
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True if shuffle is None else shuffle,
        drop_last=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
