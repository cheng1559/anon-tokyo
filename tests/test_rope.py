"""Tests for RoPE/DRoPE invariance properties."""

from __future__ import annotations

import math

import pytest
import torch

from anon_tokyo.nn.rope import apply_drope, apply_rope_2d


# ── Helpers ──────────────────────────────────────────────────────────────────

D_HEAD = 32


def _dot(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Inner product over last dim."""
    return (q * k).sum(dim=-1)


# ── RoPE 2D tests ───────────────────────────────────────────────────────────


class TestRoPE2D:
    """Position-RoPE: dot product should depend only on relative position."""

    def test_translation_invariance(self) -> None:
        """Shifting both q and k by the same offset should not change dot."""
        torch.manual_seed(0)
        q = torch.randn(4, D_HEAD)
        k = torch.randn(4, D_HEAD)
        pos_q = torch.randn(4, 2)
        pos_k = torch.randn(4, 2)
        offset = torch.randn(1, 2) * 100

        q1, k1 = apply_rope_2d(q, k, pos_q, pos_k)
        q2, k2 = apply_rope_2d(q, k, pos_q + offset, pos_k + offset)

        dot1 = _dot(q1, k1)
        dot2 = _dot(q2, k2)
        torch.testing.assert_close(dot1, dot2, atol=1e-5, rtol=1e-5)

    def test_different_relative_positions_differ(self) -> None:
        """Different relative positions should give different dots."""
        torch.manual_seed(1)
        q = torch.randn(D_HEAD)
        k = torch.randn(D_HEAD)
        pos_q = torch.tensor([0.0, 0.0])
        pos_k1 = torch.tensor([1.0, 0.0])
        pos_k2 = torch.tensor([5.0, 3.0])

        q1, k1 = apply_rope_2d(q, k, pos_q, pos_k1)
        q2, k2 = apply_rope_2d(q, k, pos_q, pos_k2)

        dot1 = _dot(q1, k1)
        dot2 = _dot(q2, k2)
        assert not torch.allclose(dot1, dot2, atol=1e-3)

    def test_output_shape(self) -> None:
        """Output shapes must match input shapes."""
        q = torch.randn(2, 8, D_HEAD)
        k = torch.randn(2, 8, D_HEAD)
        pos_q = torch.randn(2, 8, 2)
        pos_k = torch.randn(2, 8, 2)

        q_out, k_out = apply_rope_2d(q, k, pos_q, pos_k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_zero_position_identity(self) -> None:
        """At zero position, RoPE should produce cos(0)=1, sin(0)=0 → identity-like."""
        torch.manual_seed(2)
        q = torch.randn(D_HEAD)
        k = torch.randn(D_HEAD)
        pos_zero = torch.zeros(2)

        q_out, k_out = apply_rope_2d(q, k, pos_zero, pos_zero)
        torch.testing.assert_close(_dot(q_out, k_out), _dot(q, k), atol=1e-5, rtol=1e-5)

    def test_batch_consistency(self) -> None:
        """Results should be the same whether batched or computed individually."""
        torch.manual_seed(3)
        B = 3
        q = torch.randn(B, D_HEAD)
        k = torch.randn(B, D_HEAD)
        pos_q = torch.randn(B, 2)
        pos_k = torch.randn(B, 2)

        q_batch, k_batch = apply_rope_2d(q, k, pos_q, pos_k)
        for i in range(B):
            qi, ki = apply_rope_2d(q[i:i+1], k[i:i+1], pos_q[i:i+1], pos_k[i:i+1])
            torch.testing.assert_close(q_batch[i], qi.squeeze(0), atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(k_batch[i], ki.squeeze(0), atol=1e-6, rtol=1e-6)


# ── DRoPE tests ──────────────────────────────────────────────────────────────


class TestDRoPE:
    """Heading-DRoPE: dot product should depend only on relative heading."""

    def test_heading_offset_invariance(self) -> None:
        """Adding the same offset to both headings should not change dot."""
        torch.manual_seed(10)
        q = torch.randn(4, D_HEAD)
        k = torch.randn(4, D_HEAD)
        h_q = torch.randn(4)
        h_k = torch.randn(4)
        offset = torch.tensor(1.5)

        q1, k1 = apply_drope(q, k, h_q, h_k)
        q2, k2 = apply_drope(q, k, h_q + offset, h_k + offset)

        dot1 = _dot(q1, k1)
        dot2 = _dot(q2, k2)
        torch.testing.assert_close(dot1, dot2, atol=1e-5, rtol=1e-5)

    def test_2pi_periodicity(self) -> None:
        """Heading + 2π should give the same result."""
        torch.manual_seed(11)
        q = torch.randn(D_HEAD)
        k = torch.randn(D_HEAD)
        h_q = torch.tensor(0.5)
        h_k = torch.tensor(1.2)

        q1, k1 = apply_drope(q, k, h_q, h_k)
        q2, k2 = apply_drope(q, k, h_q + 2 * math.pi, h_k + 2 * math.pi)

        torch.testing.assert_close(_dot(q1, k1), _dot(q2, k2), atol=1e-5, rtol=1e-5)

    def test_different_relative_headings_differ(self) -> None:
        """Different relative headings should produce different dots."""
        torch.manual_seed(12)
        q = torch.randn(D_HEAD)
        k = torch.randn(D_HEAD)

        q1, k1 = apply_drope(q, k, torch.tensor(0.0), torch.tensor(0.0))
        q2, k2 = apply_drope(q, k, torch.tensor(0.0), torch.tensor(math.pi / 2))

        dot1 = _dot(q1, k1)
        dot2 = _dot(q2, k2)
        assert not torch.allclose(dot1, dot2, atol=1e-3)

    def test_output_shape(self) -> None:
        """Output shapes must match input shapes."""
        q = torch.randn(2, 8, D_HEAD)
        k = torch.randn(2, 8, D_HEAD)
        h_q = torch.randn(2, 8)
        h_k = torch.randn(2, 8)

        q_out, k_out = apply_drope(q, k, h_q, h_k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_zero_heading_identity(self) -> None:
        """At zero heading, DRoPE is identity (cos(0)=1, sin(0)=0)."""
        torch.manual_seed(13)
        q = torch.randn(D_HEAD)
        k = torch.randn(D_HEAD)

        q_out, k_out = apply_drope(q, k, torch.tensor(0.0), torch.tensor(0.0))
        torch.testing.assert_close(_dot(q_out, k_out), _dot(q, k), atol=1e-5, rtol=1e-5)

    def test_uniform_frequency(self) -> None:
        """All dim pairs use frequency=1, so rotating by π should negate pairs."""
        q = torch.ones(D_HEAD)
        q_pi, _ = apply_drope(q, q, torch.tensor(math.pi), torch.tensor(0.0))
        # cos(π) = -1, sin(π) ≈ 0 → first half ≈ -q, second half ≈ -q
        half = D_HEAD // 2
        torch.testing.assert_close(q_pi[:half], -q[:half], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(q_pi[half:], -q[half:], atol=1e-5, rtol=1e-5)
