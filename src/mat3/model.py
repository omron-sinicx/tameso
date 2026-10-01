# Copyright (c) 2026 OMRON SINIC X Corporation

"""MAT^3: MAsked Tactile Trajectory Transformer encoder.

Each time step ``t`` of a window of ``seq_len`` frames is turned into 10 tokens::

    token_i = [ W_i  * taxel_i(t)           | spatial(x_i, y_i) ]   i = 0..8   (3x3 taxel grid)
    token_9 = [ W_a  * action(t)            | spatial(0, 0)     ]              (action token)

Proprioceptive / F/T context is fused into every token of the step ("soft concat")::

    ctx(t)   = w_ft * W_ft ft(t) + w_wp * W_wp eef_pose(t) + w_vp * W_vp tracker_pose(t)
    x_i(t)   = w_content * (w_base * token_i(t) + w_ctx * [ctx(t) | 0]) + w_pos * [pe(t) | 0]

The 10 * seq_len tokens are encoded by a bidirectional Transformer encoder and decoded back to
taxel readings and actions (masked reconstruction, the masking ratio ~ U(0, 0.6) per window). Token embeddings of an intermediate / the last layer
are used as retrieval keys (see :mod:`mat3.retrieval`).

Parameter names are kept identical to the original research code so that its checkpoints load
with ``strict=True``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from mat3.masking import NUM_TAXELS, TOKENS_PER_STEP, MaskStrategy, token_keep_mask

# (x, y) position of each taxel on the 3x3 sensor grid; the action token uses the centre.
TAXEL_GRID = [[-1, 1], [0, 1], [1, 1], [-1, 0], [0, 0], [1, 0], [-1, -1], [0, -1], [1, -1]]
CENTER_TAXEL = 4
NUM_CONTACTILE_PILLARS = 11  # the sensor reports 11 pillars x (x, y, z); the first 9 form the grid

INPUT_KEYS = (
    "observation.contactile",  # [.., 33]
    "action.position_cmd",  # [.., 3]
    "action.rotation_cmd",  # [.., 3]
    "observation.ft",  # [.., 6]
    "observation.eef.position",  # [.., 3]
    "observation.eef.rotation_ortho6",  # [.., 6]
    "observation.vive_tracker_pose",  # [.., 7]
)


@dataclass
class ModalityWeights:
    content_weight: float = 0.9
    position_weight: float = 0.1
    base_weight: float = 0.4
    context_weight: float = 0.6
    ft_weight: float = 0.2
    wp_weight: float = 0.4
    vp_weight: float = 0.4


@dataclass
class MAT3Config:
    ninp: int = 256  # token dim (= content dim + spatial dim)
    nhead: int = 8
    nhid: int = 512  # feed-forward dim
    nlayers: int = 4
    dropout: float = 0.1
    spatial_embed_dim: int = 8
    seq_len: int = 15
    modality_weights: ModalityWeights = field(default_factory=ModalityWeights)
    # The paper uses a *bidirectional* encoder. The original research code always applied a causal
    # mask; set `causal=True` to reproduce it (e.g. for checkpoints of the original code).
    causal: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> MAT3Config:
        d = dict(d)
        d.pop("_target_", None)
        mw = d.pop("modality_weights", None) or {}
        return cls(**d, modality_weights=ModalityWeights(**mw))

    def to_dict(self) -> dict:
        return asdict(self)


# `sinusoidal_embedding_1d` and `causal_attention_mask` are adapted from the PyTorch examples
# (word_language_model), Copyright (c) 2017 PyTorch contributors, BSD 3-Clause License.
# See THIRD_PARTY_NOTICES.md.
def sinusoidal_embedding_1d(max_len: int, dim: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding, ``[1, max_len, dim]``."""
    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


def sinusoidal_embedding_2d(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Fixed 2D sinusoidal encoding of integer grid positions, ``[len(positions), dim]``."""
    enc = torch.zeros(len(positions), dim)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    for i, pos in enumerate(positions):
        enc[i, 0::4] = torch.sin(pos[0] * div_term[::2])
        enc[i, 1::4] = torch.cos(pos[0] * div_term[::2])
        enc[i, 2::4] = torch.sin(pos[1] * div_term[::2])
        enc[i, 3::4] = torch.cos(pos[1] * div_term[::2])
    return enc


def causal_attention_mask(num_tokens: int, device=None) -> torch.Tensor:
    """Additive float mask: token ``j`` is visible from token ``i`` iff ``j <= i``."""
    mask = (torch.triu(torch.ones(num_tokens, num_tokens)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
    return mask.to(device)


class MaskedTactileTransformer(nn.Module):
    def __init__(self, config: MAT3Config | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = MAT3Config.from_dict(kwargs)
        elif kwargs:
            raise TypeError("pass either `config` or keyword arguments")
        self.config = config
        ninp, sdim = config.ninp, config.spatial_embed_dim
        self.ninp = ninp
        self.spatial_embed_dim = sdim
        self.content_embed_dim = cdim = ninp - sdim
        self.seq_len = config.seq_len
        self.modality_weights = asdict(config.modality_weights)

        # NOTE: the creation order of the sub-modules below determines the RNG stream used for
        # their initialisation and must not be changed (bit-exact reproducibility).
        self.register_buffer("pos_embed", sinusoidal_embedding_1d(5000, cdim), persistent=False)
        self.spatial_encodings = nn.Parameter(
            sinusoidal_embedding_2d(torch.tensor(TAXEL_GRID), sdim), requires_grad=False
        )
        self.input_linears_contactile = nn.ModuleList(
            [nn.Linear(3, cdim) for _ in range(NUM_TAXELS)]
        )
        self.ft_embedding = nn.Linear(6, cdim)
        self.wp_embedding = nn.Linear(9, cdim)  # eef position (3) + rotation ortho6 (6)
        self.vp_embedding = nn.Linear(7, cdim)  # tracker pose (xyz + quaternion)
        self.action_encoder = nn.Linear(6, cdim)  # position cmd (3) + rotation cmd (3)
        self.action_decoder = nn.Linear(ninp, 6)
        layer = TransformerEncoderLayer(
            ninp, config.nhead, config.nhid, config.dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(layer, config.nlayers)
        self.decoder_contactile = nn.ModuleList([nn.Linear(ninp, 3) for _ in range(NUM_TAXELS)])
        self._init_weights()

    def _init_weights(self, initrange: float = 0.1) -> None:
        linears = [
            *self.input_linears_contactile,
            *self.decoder_contactile,
            self.ft_embedding,
            self.wp_embedding,
            self.vp_embedding,
            self.action_encoder,
            self.action_decoder,
        ]
        for linear in linears:
            linear.weight.data.uniform_(-initrange, initrange)
            linear.bias.data.zero_()

    # ------------------------------------------------------------------ tokens ---
    def tokenize(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Normalised observation/action window -> input tokens ``[bs, T, 10, ninp]``."""
        w = self.modality_weights
        contactile = batch["observation.contactile"]
        device = contactile.device
        bs, T = contactile.shape[:2]
        taxels = contactile.view(bs, T, NUM_CONTACTILE_PILLARS, 3)[:, :, :NUM_TAXELS, :]
        action = torch.cat([batch["action.position_cmd"], batch["action.rotation_cmd"]], dim=-1)
        ft = batch["observation.ft"]
        wp = torch.cat(
            [batch["observation.eef.position"], batch["observation.eef.rotation_ortho6"]], dim=-1
        )
        vp = batch["observation.vive_tracker_pose"]

        tokens = [
            torch.cat(
                [
                    self.input_linears_contactile[i](taxels[..., i, :]),
                    self.spatial_encodings[i].expand(bs, T, -1),
                ],
                dim=-1,
            )
            for i in range(NUM_TAXELS)
        ]
        tokens.append(
            torch.cat(
                [
                    self.action_encoder(action),
                    self.spatial_encodings[CENTER_TAXEL].expand(bs, T, -1),
                ],
                dim=-1,
            )
        )
        base = torch.stack(tokens, dim=2)  # [bs, T, 10, ninp]

        context = (
            w["ft_weight"] * self.ft_embedding(ft)
            + w["wp_weight"] * self.wp_embedding(wp)
            + w["vp_weight"] * self.vp_embedding(vp)
        )
        context = torch.cat(
            [context, torch.zeros(bs, T, self.spatial_embed_dim, device=device)], dim=-1
        )
        content = w["base_weight"] * base + w["context_weight"] * context.unsqueeze(2).expand(
            -1, -1, TOKENS_PER_STEP, -1
        )

        pos = self.pos_embed[:, :T, :].unsqueeze(2).expand(-1, -1, TOKENS_PER_STEP, -1)
        pos = torch.cat(
            [pos, torch.zeros(1, T, TOKENS_PER_STEP, self.spatial_embed_dim, device=device)], dim=-1
        )
        return w["content_weight"] * content + w["position_weight"] * pos

    def decode(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[bs, T, 10, ninp]`` -> taxel ``[bs, T, 9, 3]`` and action ``[bs, T, 6]`` predictions."""
        contactile = torch.stack(
            [self.decoder_contactile[i](hidden[:, :, i, :]) for i in range(NUM_TAXELS)], dim=2
        )
        return contactile, self.action_decoder(hidden[:, :, NUM_TAXELS, :])

    # ----------------------------------------------------------------- forward ---
    def forward(
        self,
        batch: dict[str, torch.Tensor],
        mask_strategy: MaskStrategy | str = MaskStrategy.NONE,
        max_mask_ratio: float = 0.6,
        embedding_layer: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Masked reconstruction.

        Returns ``(contactile_pred [bs,T,9,3], action_pred [bs,T,6], embeddings [bs,T,10,ninp])``.
        ``embeddings`` are the last-layer outputs, or those of ``embedding_layer`` (eval mode only).
        """
        x = self.tokenize(batch)
        bs, T = x.shape[:2]
        keep = token_keep_mask(bs, T, mask_strategy, max_mask_ratio, device=x.device)
        if keep is not None:
            mask_token = torch.zeros(self.ninp, device=x.device)
            x = torch.where(keep.unsqueeze(-1).expand(-1, -1, -1, self.ninp), x, mask_token)
        x = x.view(bs, T * TOKENS_PER_STEP, self.ninp)
        attn_mask = (
            causal_attention_mask(T * TOKENS_PER_STEP, x.device) if self.config.causal else None
        )

        if embedding_layer is not None and not self.training:
            layer_outputs = []
            for layer in self.transformer_encoder.layers:
                x = layer(x, src_mask=attn_mask)
                layer_outputs.append(x)
            n = len(layer_outputs)
            if not -n <= embedding_layer < n:
                raise IndexError(f"embedding_layer={embedding_layer} for a {n}-layer encoder")
            idx = embedding_layer % n  # python-style: -1 = last layer
            embeddings = layer_outputs[idx].view(bs, T, TOKENS_PER_STEP, self.ninp)
            hidden = layer_outputs[-1].view(bs, T, TOKENS_PER_STEP, self.ninp)
        else:
            hidden = self.transformer_encoder(x, attn_mask).view(bs, T, TOKENS_PER_STEP, self.ninp)
            embeddings = hidden

        contactile_pred, action_pred = self.decode(hidden)
        return contactile_pred, action_pred, embeddings

    @torch.no_grad()
    def encode(
        self,
        batch: dict[str, torch.Tensor],
        layer: int | None = -1,
        mask_current_action: bool = True,
    ) -> torch.Tensor:
        """Token embeddings ``[bs, T, 10, ninp]`` used as retrieval keys.

        As in the paper, the action token of the current (last) step is masked, since the current
        action is unknown at execution time; the same keys are stored in the tactile memory.
        (The original research code did not apply this mask: ``mask_current_action=False``.)

        ``layer=None`` runs the fused ``nn.TransformerEncoder``; an integer taps that layer's output
        (``-1`` = last). Both give the same values up to kernel-level floating point differences.
        """
        strategy = MaskStrategy.LAST_ACTION if mask_current_action else MaskStrategy.NONE
        was_training = self.training
        self.eval()
        try:
            return self.forward(batch, strategy, embedding_layer=layer)[2]
        finally:
            self.train(was_training)
