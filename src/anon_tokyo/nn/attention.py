"""Sparse top-k multi-head attention with RoPE / DRoPE.

Each query attends only to its *k* nearest neighbours (by Euclidean
distance on 2-D positions).  Even-indexed heads use Position-RoPE;
odd-indexed heads use Heading-DRoPE.

``position_encoding`` controls how coordinates/headings enter attention:
``rope_drope`` splits heads by parity, ``rope`` and ``drope`` apply one
encoding to every head, and ``sine`` uses additive sinusoidal PE.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def _apply_rope_2d_batched(q: Tensor, k: Tensor, pos_q: Tensor, pos_k: Tensor, base: float = 10000.0) -> tuple[Tensor, Tensor]:
    """Vectorized 2-D RoPE for ``q [B,N,H,D]`` and ``k [B,N,K,H,D]``."""
    d = q.shape[-1]
    half = d // 2
    idx = torch.arange(half // 2, device=q.device, dtype=torch.float32)
    freqs = 1.0 / (base ** (2 * idx / half))

    qx = pos_q[..., 0, None] * freqs
    qy = pos_q[..., 1, None] * freqs
    q_cos = torch.cat((qx.cos(), qy.cos()), dim=-1).unsqueeze(2)
    q_sin = torch.cat((qx.sin(), qy.sin()), dim=-1).unsqueeze(2)

    kx = pos_k[..., 0, None] * freqs
    ky = pos_k[..., 1, None] * freqs
    k_cos = torch.cat((kx.cos(), ky.cos()), dim=-1).unsqueeze(3)
    k_sin = torch.cat((kx.sin(), ky.sin()), dim=-1).unsqueeze(3)

    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    q_rot = torch.cat((q1 * q_cos - q2 * q_sin, q2 * q_cos + q1 * q_sin), dim=-1)
    k_rot = torch.cat((k1 * k_cos - k2 * k_sin, k2 * k_cos + k1 * k_sin), dim=-1)
    return q_rot, k_rot


def _apply_drope_batched(q: Tensor, k: Tensor, heading_q: Tensor, heading_k: Tensor) -> tuple[Tensor, Tensor]:
    """Vectorized heading DRoPE for ``q [B,N,H,D]`` and ``k [B,N,K,H,D]``."""
    half = q.shape[-1] // 2
    q_cos = heading_q.cos().unsqueeze(-1).unsqueeze(-1).expand(*q.shape[:-1], half)
    q_sin = heading_q.sin().unsqueeze(-1).unsqueeze(-1).expand(*q.shape[:-1], half)
    k_cos = heading_k.cos().unsqueeze(-1).unsqueeze(-1).expand(*k.shape[:-1], half)
    k_sin = heading_k.sin().unsqueeze(-1).unsqueeze(-1).expand(*k.shape[:-1], half)

    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    q_rot = torch.cat((q1 * q_cos - q2 * q_sin, q2 * q_cos + q1 * q_sin), dim=-1)
    k_rot = torch.cat((k1 * k_cos - k2 * k_sin, k2 * k_cos + k1 * k_sin), dim=-1)
    return q_rot, k_rot


def _sinusoidal_pe(pos: Tensor, d_model: int) -> Tensor:
    """Fallback sinusoidal PE for ``[..., 2]`` coordinates → ``[..., d_model]``."""
    half = d_model // 2
    dim_t = torch.arange(half, dtype=torch.float32, device=pos.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half)
    x = pos[..., 0:1] / dim_t
    y = pos[..., 1:2] / dim_t
    pe = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=-1)
    return pe[..., :d_model]


@torch.no_grad()
def select_topk(
    pos_q: Tensor,
    pos_k: Tensor,
    mask_k: Tensor,
    k: int,
) -> Tensor:
    """Select top-k nearest keys for each query by Euclidean distance.

    Args:
        pos_q:  ``[B, N_q, 2]``
        pos_k:  ``[B, N_k, 2]``
        mask_k: ``[B, N_k]`` bool — True = valid key.
        k:      number of neighbours.

    Returns:
        ``[B, N_q, k]`` — indices into the key axis. Clamped to valid range.
    """
    # [B, N_q, N_k]
    dist = (pos_q.unsqueeze(2) - pos_k.unsqueeze(1)).norm(dim=-1)
    dist = dist.masked_fill(~mask_k.unsqueeze(1), float("inf"))
    actual_k = min(k, dist.shape[-1])
    _, topk_idx = dist.topk(actual_k, dim=-1, largest=False)  # [B, N_q, actual_k]
    if actual_k < k:
        topk_idx = F.pad(topk_idx, (0, k - actual_k), value=0)
    return topk_idx


class SparseTopKAttention(nn.Module):
    """Multi-head attention over top-k spatial neighbours with RoPE/DRoPE.

    Supports both self-attention (Q=K source) and cross-attention.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        sparse_k: int = 32,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_drope: bool = True,
        position_encoding: str | None = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.sparse_k = sparse_k
        if position_encoding is None:
            if use_rope and use_drope:
                position_encoding = "rope_drope"
            elif use_rope:
                position_encoding = "rope"
            elif use_drope:
                position_encoding = "drope"
            else:
                position_encoding = "sine"
        if position_encoding not in {"rope_drope", "rope", "drope", "sine"}:
            raise ValueError(f"Unsupported position_encoding: {position_encoding}")
        self.position_encoding = position_encoding
        self.use_rope = position_encoding in {"rope_drope", "rope"}
        self.use_drope = position_encoding in {"rope_drope", "drope"}
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        q_feat: Tensor,
        kv_feat: Tensor,
        pos_q: Tensor,
        pos_k: Tensor,
        heading_q: Tensor,
        heading_k: Tensor,
        mask_k: Tensor,
        topk_idx: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            q_feat:    ``[B, N_q, D]``
            kv_feat:   ``[B, N_k, D]``
            pos_q:     ``[B, N_q, 2]`` — (x, y) positions for queries.
            pos_k:     ``[B, N_k, 2]`` — (x, y) positions for keys.
            heading_q: ``[B, N_q]``    — heading angle for queries.
            heading_k: ``[B, N_k]``    — heading angle for keys.
            mask_k:    ``[B, N_k]``    — bool, True = valid key.
            topk_idx:  ``[B, N_q, k]`` — optional precomputed indices.

        Returns:
            ``[B, N_q, D]``
        """
        B, N_q, D = q_feat.shape
        N_k = kv_feat.shape[1]
        H = self.num_heads
        d_h = self.d_head
        k = self.sparse_k

        # Top-k selection
        if topk_idx is None:
            topk_idx = select_topk(pos_q, pos_k, mask_k, k)  # [B, N_q, k]

        actual_k = topk_idx.shape[2]

        # Gather KV features and positions for top-k
        # Expand topk_idx for gathering: [B, N_q, k] → [B, N_q*k]
        flat_idx = topk_idx.reshape(B, N_q * actual_k)
        b_idx = torch.arange(B, device=kv_feat.device)[:, None].expand(B, N_q * actual_k)
        kv_gathered = kv_feat[b_idx, flat_idx].reshape(B, N_q, actual_k, D)  # [B, N_q, k, D]
        pos_k_gathered = pos_k[b_idx, flat_idx].reshape(B, N_q, actual_k, 2)
        heading_k_gathered = heading_k[b_idx, flat_idx].reshape(B, N_q, actual_k)

        # Key mask: check if gathered keys are valid
        k_mask_gathered = mask_k[b_idx, flat_idx].reshape(B, N_q, actual_k)  # [B, N_q, k]

        # Project Q, K, V
        Q = self.q_proj(q_feat)  # [B, N_q, D]
        K = self.k_proj(kv_gathered)  # [B, N_q, k, D]
        V = self.v_proj(kv_gathered)  # [B, N_q, k, D]

        # Reshape to multi-head: [B, N_q, H, d_h] and [B, N_q, k, H, d_h]
        Q = Q.reshape(B, N_q, H, d_h)
        K = K.reshape(B, N_q, actual_k, H, d_h)
        V = V.reshape(B, N_q, actual_k, H, d_h)

        if self.position_encoding != "sine":
            Q_all = Q.unsqueeze(2)
            K_all = K
            if self.position_encoding == "rope":
                Q_rot, K_all = _apply_rope_2d_batched(Q, K, pos_q, pos_k_gathered)
                Q_all = Q_rot.unsqueeze(2)
            elif self.position_encoding == "drope":
                Q_rot, K_all = _apply_drope_batched(Q, K, heading_q, heading_k_gathered)
                Q_all = Q_rot.unsqueeze(2)
            else:
                Q_enc = torch.empty_like(Q)
                K_enc = torch.empty_like(K)
                qr, kr = _apply_rope_2d_batched(Q[:, :, 0::2], K[:, :, :, 0::2], pos_q, pos_k_gathered)
                Q_enc[:, :, 0::2] = qr
                K_enc[:, :, :, 0::2] = kr
                qd, kd = _apply_drope_batched(Q[:, :, 1::2], K[:, :, :, 1::2], heading_q, heading_k_gathered)
                Q_enc[:, :, 1::2] = qd
                K_enc[:, :, :, 1::2] = kd
                Q_all = Q_enc.unsqueeze(2)
                K_all = K_enc

            attn_logits = (Q_all * K_all).sum(dim=-1) * self.scale  # [B, N_q, k, H]
        else:
            # Fallback: additive sinusoidal PE
            pe_q = _sinusoidal_pe(pos_q, self.d_model)  # [B, N_q, D]
            pe_k = _sinusoidal_pe(pos_k_gathered.reshape(B, N_q * actual_k, 2), self.d_model)
            pe_k = pe_k.reshape(B, N_q, actual_k, D)
            Q_pe = (Q.reshape(B, N_q, D) + pe_q).reshape(B, N_q, H, d_h)
            K_pe = (K.reshape(B, N_q, actual_k, D) + pe_k).reshape(B, N_q, actual_k, H, d_h)
            # [B, N_q, 1, H, d_h] * [B, N_q, k, H, d_h] → [B, N_q, k, H]
            attn_logits = (Q_pe.unsqueeze(2) * K_pe).sum(dim=-1) * self.scale

        # Mask invalid keys
        attn_logits = attn_logits.masked_fill(~k_mask_gathered.unsqueeze(-1), float("-inf"))

        attn_weights = F.softmax(attn_logits, dim=2)  # [B, N_q, k, H]
        attn_weights = self.attn_drop(attn_weights)

        # Weighted sum of values: [B, N_q, k, H, d_h] → [B, N_q, H, d_h]
        out = (attn_weights.unsqueeze(-1) * V).sum(dim=2)  # [B, N_q, H, d_h]

        out = out.reshape(B, N_q, D)
        return self.out_proj(out)
