# Copyright (c) 2026 OMRON SINIC X Corporation

"""Token masking strategies for masked tactile modelling.

Every time step is tokenised into ``TOKENS_PER_STEP = 10`` tokens: 9 taxel tokens followed by one
action token. A *keep mask* of shape ``[batch, seq_len, 10]`` marks the tokens that are fed to the
encoder (``True``) and the ones that are replaced by the (zero) mask token (``False``).

``RANDOM_ALL`` (default) is the masked token prediction of the paper: for every window a masking
ratio is drawn from ``U(0, max_mask_ratio)`` (``max_mask_ratio = 0.6``) and each taxel / action token
is masked with that probability. ``LAST_ACTION`` masks the action of the current step, which is how
retrieval keys are computed at inference.

The attention pattern (bidirectional by default, causal for the original code) is a property of the
model (``MAT3Config.causal``), not of the masking strategy: ``CAUSAL`` / ``CAUSAL_LAST_ACTION`` are
aliases of ``NONE`` / ``LAST_ACTION`` kept for config compatibility with the original code.
"""

from __future__ import annotations

from enum import Enum

import torch

NUM_TAXELS = 9
TOKENS_PER_STEP = NUM_TAXELS + 1  # 9 taxels + 1 action token
ACTION_TOKEN = NUM_TAXELS


class MaskStrategy(str, Enum):
    NONE = "none"
    LAST_ACTION = "last_action"  # hide the action of the last step (action "retrieval" query)
    RANDOM_ACTION = "random_action"  # hide action tokens with ratio ~ U(0, max_mask_ratio)
    RANDOM_TAXEL = "random_taxel"  # hide taxel tokens with ratio ~ U(0, max_mask_ratio)
    RANDOM_ALL = "random_all"  # hide any token with ratio ~ U(0, max_mask_ratio)  (default)
    CAUSAL = "causal"
    CAUSAL_LAST_ACTION = "causal_last_action"


def token_keep_mask(
    batch_size: int,
    seq_len: int,
    strategy: MaskStrategy | str = MaskStrategy.NONE,
    max_mask_ratio: float = 0.6,
    device: torch.device | str | None = None,
) -> torch.Tensor | None:
    """Return a boolean keep mask ``[batch_size, seq_len, TOKENS_PER_STEP]`` or ``None`` (keep all).

    For the random strategies the mask ratio is sampled per sample from ``U(0, max_mask_ratio)``.
    The order of the random draws is part of the contract (training reproducibility).
    """
    strategy = MaskStrategy(strategy)
    if strategy in (MaskStrategy.NONE, MaskStrategy.CAUSAL):
        return None

    keep = torch.ones(batch_size, seq_len, TOKENS_PER_STEP, dtype=torch.bool, device=device)

    if strategy in (MaskStrategy.LAST_ACTION, MaskStrategy.CAUSAL_LAST_ACTION):
        keep[:, -1, ACTION_TOKEN] = False

    elif strategy == MaskStrategy.RANDOM_ALL:
        ratio = torch.rand(batch_size, 1, 1, device=device) * max_mask_ratio
        keep &= ~(torch.rand(batch_size, seq_len, TOKENS_PER_STEP, device=device) < ratio)

    elif strategy == MaskStrategy.RANDOM_ACTION:
        ratio = torch.rand(batch_size, 1, device=device) * max_mask_ratio
        keep[..., ACTION_TOKEN] = ~(torch.rand(batch_size, seq_len, device=device) < ratio)

    elif strategy == MaskStrategy.RANDOM_TAXEL:
        ratio = torch.rand(batch_size, 1, device=device) * max_mask_ratio
        for t in range(NUM_TAXELS):
            keep[..., t] = ~(torch.rand(batch_size, seq_len, device=device) < ratio)

    return keep
