# Copyright (c) 2026 OMRON SINIC X Corporation

"""Vicinity retrieval: encode windows with MAT^3, pool the token embeddings, look up the nearest
demonstration and return its action.

A database stores, for every training window, the pooled embedding (key) and the *last-step*
action of the window (value): ``[position_cmd (3), rotation_cmd (3)]`` in raw units.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from vicinity import Backend, Metric, Vicinity

from mat3.data import SEQUENCE_KEYS
from mat3.model import MaskedTactileTransformer
from mat3.normalize import Normalizer

ACTION_KEYS = ("action.position_cmd", "action.rotation_cmd")


class Pooling(str, Enum):
    GLOBAL_AVERAGE = "global_average"  # mean over time steps and tokens -> [N, D]   (default)
    AVERAGE = "average"  # mean over tokens -> [N, T, D]
    ACTION_TOKEN = "action_token"  # action token -> [N, T, D]


def pool(embeddings: np.ndarray, pooling: Pooling | str = Pooling.GLOBAL_AVERAGE) -> np.ndarray:
    """``embeddings``: ``[N, T, tokens, D]``."""
    pooling = Pooling(pooling)
    if pooling == Pooling.GLOBAL_AVERAGE:
        return np.mean(embeddings, axis=(1, 2))
    if pooling == Pooling.AVERAGE:
        return np.mean(embeddings, axis=2)
    return embeddings[:, :, -1, :]


def window_actions(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Raw actions of a window batch, ``[B, T, 6]``."""
    return torch.cat([batch[k] for k in ACTION_KEYS], dim=-1)


@torch.no_grad()
def embed(
    model: MaskedTactileTransformer,
    normalizer: Normalizer,
    batch: dict[str, torch.Tensor],
    layer: int | None = -1,
    pooling: Pooling | str = Pooling.GLOBAL_AVERAGE,
    mask_current_action: bool = True,
) -> np.ndarray:
    """Retrieval keys ``z`` of a batch of raw (un-normalised) windows: ``[B, D]`` for
    ``global_average`` (default), ``[B, T * D]`` for the per-step poolings."""
    device = next(model.parameters()).device
    x = normalizer({k: batch[k].to(device) for k in SEQUENCE_KEYS})
    tokens = model.encode(x, layer=layer, mask_current_action=mask_current_action)
    keys = pool(tokens.cpu().numpy(), pooling)
    return keys.reshape(len(keys), -1)  # per-step poolings: concatenate the T step vectors


@torch.no_grad()
def encode_dataset(
    model,
    normalizer,
    loader: DataLoader,
    layer: int | None = -1,
    pooling=Pooling.GLOBAL_AVERAGE,
    mask_current_action: bool = True,
):
    """Returns ``(keys [N, D], last-step actions [N, 6], window actions [N, T, 6], frame indices [N])``."""
    keys, actions, indices = [], [], []
    for batch in tqdm(loader, desc="encoding"):
        keys.append(embed(model, normalizer, batch, layer, pooling, mask_current_action))
        actions.append(window_actions(batch).numpy())
        indices.append(batch["index"].numpy())
    actions_np = np.concatenate(actions)
    return np.concatenate(keys), actions_np[:, -1], actions_np, np.concatenate(indices)


# HNSW parameters tuned on contactile_200_lota (15,910 keys, 256-dim; validation windows as
# queries): recall@1 = 0.98 at ~0.1 ms/query on one CPU core. The library defaults
# (M=16, ef_construction=200, ef=10), which the original code used, give recall@1 = 0.54.
ANN_BACKENDS = ("hnsw", "voyager")
DEFAULT_ANN = {"m": 32, "ef_construction": 400, "ef_search": 400}


class TactileDB:
    """Nearest-neighbour store of ``embedding -> action`` backed by `vicinity`.

    The default backend is HNSW (``hnswlib``) for real-time queries; ``voyager`` (Spotify's HNSW)
    takes the same parameters. ``basic`` performs exact search (for analysis only). ``ef_search``
    trades recall for latency and is re-applied when a database is loaded. HNSW indices are built
    with multi-threaded insertion and are therefore not bit-reproducible across builds.
    """

    def __init__(self, vicinity: Vicinity, actions: np.ndarray, meta: dict | None = None):
        self.vicinity = vicinity
        self.actions = np.asarray(actions, dtype=np.float32)
        self.meta = meta or {}
        if self.meta.get("ef_search") is not None:
            self.set_ef_search(self.meta["ef_search"])

    @classmethod
    def build(
        cls,
        keys: np.ndarray,
        actions: np.ndarray,
        backend: str = "hnsw",
        metric: str = "euclidean",
        m: int = DEFAULT_ANN["m"],
        ef_construction: int = DEFAULT_ANN["ef_construction"],
        ef_search: int | None = DEFAULT_ANN["ef_search"],
        meta: dict | None = None,
    ) -> TactileDB:
        ann = backend in ANN_BACKENDS
        vicinity = Vicinity.from_vectors_and_items(
            vectors=np.asarray(keys),
            items=list(range(len(keys))),
            backend_type=Backend(backend),
            metric=Metric(metric),
            store_vectors=True,
            **({"m": m, "ef_construction": ef_construction} if ann else {}),
        )
        meta = {"backend": backend, "metric": metric, **(meta or {})}
        if ann:
            meta.update(m=m, ef_construction=ef_construction, ef_search=ef_search)
        return cls(vicinity, actions, meta)

    def set_ef_search(self, ef: int) -> None:
        """Size of the dynamic candidate list at query time (HNSW backends only)."""
        index = getattr(self.vicinity.backend, "index", None)
        if hasattr(index, "set_ef"):  # hnswlib
            index.set_ef(int(ef))
        elif hasattr(index, "ef"):  # voyager
            index.ef = int(ef)
        self.meta["ef_search"] = int(ef)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.vicinity.save(path / "index", overwrite=True)
        np.save(path / "actions.npy", self.actions)
        (path / "meta.json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> TactileDB:
        path = Path(path)
        return cls(
            Vicinity.load(path / "index"),
            np.load(path / "actions.npy"),
            json.loads((path / "meta.json").read_text()),
        )

    def __len__(self) -> int:
        return len(self.actions)

    def query(self, keys: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``keys [B, D]`` -> ``(actions [B, k, 6], distances [B, k], db indices [B, k])``."""
        results = self.vicinity.query(np.atleast_2d(keys), k=k)
        idx = np.array([[item for item, _ in r] for r in results], dtype=np.int64)
        dist = np.array([[score for _, score in r] for r in results], dtype=np.float32)
        return self.actions[idx], dist, idx


class RetrievalPolicy:
    """Non-parametric policy of TaMeSo: encode the observation history, retrieve the ``k`` nearest
    entries of the tactile memory (L2) and execute the action of one of them, sampled uniformly."""

    def __init__(
        self,
        model: MaskedTactileTransformer,
        normalizer: Normalizer,
        db: TactileDB,
        k: int = 1,
        seed: int | None = None,
    ):
        self.model, self.normalizer, self.db = model.eval(), normalizer, db
        self.layer = db.meta.get("layer", -1)
        self.pooling = db.meta.get("pooling", Pooling.GLOBAL_AVERAGE)
        self.mask_current_action = db.meta.get("mask_current_action", True)
        self.k = k
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_pretrained(
        cls, checkpoint: str | Path, db_path: str | Path, device: str = "cpu", **kwargs
    ) -> RetrievalPolicy:
        from mat3.checkpoint import load_checkpoint

        model, normalizer, _ = load_checkpoint(checkpoint, device)
        return cls(model, normalizer, TactileDB.load(db_path), **kwargs)

    def predict(
        self, observation: dict[str, np.ndarray | torch.Tensor], k: int | None = None
    ) -> np.ndarray:
        """``observation[key]``: ``[T, D]`` window (or ``[B, T, D]``) of the observation history;
        the current action (last step of the action keys) is masked and may hold any value.
        Returns ``[6]`` (or ``[B, 6]``)."""
        k = k or self.k
        batch = {
            key: torch.as_tensor(np.asarray(observation[key]), dtype=torch.float32)
            for key in SEQUENCE_KEYS
        }
        single = batch["observation.contactile"].dim() == 2
        if single:
            batch = {key: v[None] for key, v in batch.items()}
        z_q = embed(
            self.model, self.normalizer, batch, self.layer, self.pooling, self.mask_current_action
        )
        actions, _, _ = self.db.query(z_q, k=k)
        choice = (
            self.rng.integers(0, k, size=len(actions)) if k > 1 else np.zeros(len(actions), int)
        )
        action = actions[np.arange(len(actions)), choice]
        return action[0] if single else action
