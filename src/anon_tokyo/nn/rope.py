"""Rotary Position Embedding (RoPE) and Directional RoPE (DRoPE).

Position-RoPE encodes 2-D spatial coordinates into Q/K vectors via
multi-frequency rotary transforms so that the dot product depends only on
the *relative* displacement between two tokens.

Heading-DRoPE encodes a scalar heading angle with frequency = 1, so
that the dot product depends only on the *relative* heading difference.

Reference: Zhao et al., "DRoPE: Directional Rotary Position Embedding
for Language and Multi-Agent Motion Modeling", 2025.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _rope_freqs(d: int, device: torch.device, base: float = 10000.0) -> Tensor:
    """Compute RoPE frequency bands.

    Returns:
        ``[d // 2]`` inverse-frequency vector.
    """
    half = d // 2
    idx = torch.arange(half, device=device, dtype=torch.float32)
    return 1.0 / (base ** (2 * idx / d))


def _apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary transform to the last dimension of *x*.

    Args:
        x:   ``[..., d]`` where d is even.
        cos: ``[..., d // 2]`` broadcastable.
        sin: ``[..., d // 2]`` broadcastable.

    Returns:
        ``[..., d]``
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def apply_rope_2d(
    q: Tensor,
    k: Tensor,
    pos_q: Tensor,
    pos_k: Tensor,
    base: float = 10000.0,
) -> tuple[Tensor, Tensor]:
    """Apply 2-D Position-RoPE to query and key tensors.

    The head dimension *d* is split into two halves: the first half encodes
    the *x* coordinate and the second half encodes the *y* coordinate.

    Args:
        q:     ``[..., d]`` query vectors.
        k:     ``[..., d]`` key vectors.
        pos_q: ``[..., 2]`` (x, y) positions for queries.
        pos_k: ``[..., 2]`` (x, y) positions for keys.
        base:  frequency base (default 10000).

    Returns:
        Rotated ``(q, k)`` with same shapes.
    """
    d = q.shape[-1]
    half = d // 2

    freqs = _rope_freqs(half, q.device, base)  # [half // 2]

    # X component → first half, Y component → second half
    cos_qx = (pos_q[..., 0:1] * freqs).cos()  # [..., half // 2]
    sin_qx = (pos_q[..., 0:1] * freqs).sin()
    cos_qy = (pos_q[..., 1:2] * freqs).cos()
    sin_qy = (pos_q[..., 1:2] * freqs).sin()
    cos_q = torch.cat([cos_qx, cos_qy], dim=-1)  # [..., half]
    sin_q = torch.cat([sin_qx, sin_qy], dim=-1)

    cos_kx = (pos_k[..., 0:1] * freqs).cos()
    sin_kx = (pos_k[..., 0:1] * freqs).sin()
    cos_ky = (pos_k[..., 1:2] * freqs).cos()
    sin_ky = (pos_k[..., 1:2] * freqs).sin()
    cos_k = torch.cat([cos_kx, cos_ky], dim=-1)
    sin_k = torch.cat([sin_kx, sin_ky], dim=-1)

    q_rot = _apply_rotary(q, cos_q, sin_q)
    k_rot = _apply_rotary(k, cos_k, sin_k)
    return q_rot, k_rot


def apply_drope(
    q: Tensor,
    k: Tensor,
    heading_q: Tensor,
    heading_k: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply Heading-DRoPE to query and key tensors.

    Uses a *single* frequency (= 1) for all dimension pairs, so that
    ``<q_i, k_j>`` depends only on ``heading_i - heading_j``.

    Args:
        q:         ``[..., d]`` query vectors.
        k:         ``[..., d]`` key vectors.
        heading_q: ``[...]`` heading angle in radians for queries.
        heading_k: ``[...]`` heading angle in radians for keys.

    Returns:
        Rotated ``(q, k)`` with same shapes.
    """
    cos_q = heading_q.unsqueeze(-1).cos().expand_as(q[..., : q.shape[-1] // 2])
    sin_q = heading_q.unsqueeze(-1).sin().expand_as(q[..., : q.shape[-1] // 2])
    cos_k = heading_k.unsqueeze(-1).cos().expand_as(k[..., : k.shape[-1] // 2])
    sin_k = heading_k.unsqueeze(-1).sin().expand_as(k[..., : k.shape[-1] // 2])

    q_rot = _apply_rotary(q, cos_q, sin_q)
    k_rot = _apply_rotary(k, cos_k, sin_k)
    return q_rot, k_rot
