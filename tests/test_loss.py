"""Tests for prediction loss functions."""

from __future__ import annotations

import math

import torch
import pytest

from anon_tokyo.prediction.loss import (
    nll_loss_gmm,
    score_loss,
    velocity_loss,
    prediction_loss,
)


# ── nll_loss_gmm ──────────────────────────────────────────────────────────────


class TestNllLossGmm:
    def test_winner_selection(self) -> None:
        """Winner should be the mode closest to GT."""
        B, K, M, T = 2, 1, 3, 4
        pred = torch.zeros(B, K, M, T, 5)
        gt = torch.zeros(B, K, T, 2)

        # Mode 0: far from gt, mode 1: close, mode 2: medium
        pred[0, 0, 0, :, 0] = 10.0  # x offset
        pred[0, 0, 1, :, 0] = 0.1  # closest
        pred[0, 0, 2, :, 0] = 5.0
        # log_std = 0 => std = 1, rho = 0
        mask = torch.ones(B, K, T)

        loss, winner = nll_loss_gmm(pred, gt, mask)
        assert winner[0, 0] == 1  # mode 1 is closest

    def test_isotropic_gaussian_nll(self) -> None:
        """Check NLL against hand-computed value for isotropic Gaussian."""
        B, K, M, T = 1, 1, 1, 1
        # pred: (mu_x=0, mu_y=0, log_std1=0, log_std2=0, rho=0)
        pred = torch.zeros(B, K, M, T, 5)
        gt = torch.tensor([[[[1.0, 0.0]]]])  # dx=1, dy=0
        mask = torch.ones(B, K, T)

        loss, _ = nll_loss_gmm(pred, gt, mask)

        # NLL = log(1) + log(1) + 0.5*log(1) + 0.5*(1+0)/(1) = 0.5
        # = log_std1 + log_std2 + 0.5*log(1-rho^2) + 0.5*(dx^2/s1^2 + dy^2/s2^2)/(1-rho^2)
        # = 0 + 0 + 0 + 0.5 * 1 = 0.5
        torch.testing.assert_close(loss, torch.tensor([[0.5]]), atol=1e-5, rtol=1e-5)

    def test_mask_zeroes_invalid_frames(self) -> None:
        """Invalid frames should not contribute to loss."""
        B, K, M, T = 1, 1, 1, 4
        pred = torch.zeros(B, K, M, T, 5)
        # Frame 0 has small offset, frames 1-3 have large offset
        gt = torch.zeros(B, K, T, 2)
        gt[0, 0, 1:, :] = 100.0
        mask = torch.zeros(B, K, T)
        mask[0, 0, 0] = 1.0  # only first frame valid (small offset)

        loss_partial, _ = nll_loss_gmm(pred, gt, mask)

        mask_full = torch.ones(B, K, T)
        loss_full, _ = nll_loss_gmm(pred, gt, mask_full)

        # Partial: only frame 0 (offset=0), Full: includes frames with offset=100
        assert loss_partial.item() < loss_full.item()

    def test_shape(self) -> None:
        B, K, M, T = 4, 2, 6, 80
        pred = torch.randn(B, K, M, T, 5)
        gt = torch.randn(B, K, T, 2)
        mask = torch.ones(B, K, T)
        loss, winner = nll_loss_gmm(pred, gt, mask)
        assert loss.shape == (B, K)
        assert winner.shape == (B, K)
        assert winner.min() >= 0
        assert winner.max() < M


# ── score_loss ────────────────────────────────────────────────────────────────


class TestScoreLoss:
    def test_perfect_prediction(self) -> None:
        """CE = 0 when model assigns all probability to the winner."""
        B, K, M = 1, 1, 6
        scores = torch.full((B, K, M), -100.0)
        scores[0, 0, 2] = 100.0  # all probability on mode 2
        winner = torch.tensor([[2]])
        loss = score_loss(scores, winner)
        assert loss.item() < 1e-3

    def test_uniform_baseline(self) -> None:
        """CE = log(M) when uniform scores."""
        B, K, M = 1, 1, 6
        scores = torch.zeros(B, K, M)
        winner = torch.tensor([[0]])
        loss = score_loss(scores, winner)
        torch.testing.assert_close(loss, torch.tensor([[math.log(6)]]), atol=1e-4, rtol=1e-4)

    def test_shape(self) -> None:
        B, K, M = 4, 2, 6
        scores = torch.randn(B, K, M)
        winner = torch.randint(0, M, (B, K))
        loss = score_loss(scores, winner)
        assert loss.shape == (B, K)


# ── velocity_loss ─────────────────────────────────────────────────────────────


class TestVelocityLoss:
    def test_zero_when_perfect(self) -> None:
        B, K, M, T = 1, 1, 2, 10
        gt = torch.randn(B, K, T, 2)
        pred = torch.zeros(B, K, M, T, 2)
        pred[:, :, 0] = gt  # mode 0 matches exactly
        mask = torch.ones(B, K, T)
        winner = torch.zeros(B, K, dtype=torch.long)
        loss = velocity_loss(pred, gt, mask, winner)
        torch.testing.assert_close(loss, torch.zeros(B, K), atol=1e-6, rtol=1e-6)

    def test_shape(self) -> None:
        B, K, M, T = 4, 2, 6, 80
        pred = torch.randn(B, K, M, T, 2)
        gt = torch.randn(B, K, T, 2)
        mask = torch.ones(B, K, T)
        winner = torch.randint(0, M, (B, K))
        loss = velocity_loss(pred, gt, mask, winner)
        assert loss.shape == (B, K)


# ── prediction_loss ───────────────────────────────────────────────────────────


class TestPredictionLoss:
    def test_combined_runs(self) -> None:
        """Smoke test: combined loss runs without error."""
        B, K, M, T, A = 2, 2, 6, 80, 32
        output = {
            "pred_trajs": torch.randn(B, K, M, T, 7, requires_grad=True),
            "pred_scores": torch.randn(B, K, M, requires_grad=True),
        }
        batch = {
            "obj_trajs_future_local": torch.randn(B, A, T, 4),
            "obj_trajs_future_mask": torch.ones(B, A, T),
            "tracks_to_predict": torch.tensor([[0, 1, -1, -1, -1, -1, -1, -1]] * B),
            "obj_trajs": torch.randn(B, A, 11, 10),
            "obj_types": torch.randint(1, 4, (B, A)),
        }
        weights = {"reg": 1.0, "score": 1.0, "vel": 0.2}
        total, loss_dict = prediction_loss(output, batch, weights)
        assert total.shape == ()
        assert total.requires_grad
        base_keys = {k for k in loss_dict if not k.startswith("ade/")}
        assert base_keys == {
            "loss/layer0",
            "loss/layer0_reg_gmm",
            "loss/layer0_reg_vel",
            "loss/layer0_cls",
            "loss/decoder",
            "loss/total",
        }
        assert any(k.startswith("ade/") for k in loss_dict)

    def test_combined_with_dense(self) -> None:
        """Loss includes dense when pred_dense_trajs is present."""
        B, K, M, T, A = 2, 2, 6, 80, 32
        output = {
            "pred_trajs": torch.randn(B, K, M, T, 7, requires_grad=True),
            "pred_scores": torch.randn(B, K, M, requires_grad=True),
            "pred_dense_trajs": torch.randn(B, K, A, T, 7, requires_grad=True),
        }
        batch = {
            "obj_trajs_future_local": torch.randn(B, A, T, 4),
            "obj_trajs_future_mask": torch.ones(B, A, T),
            "obj_trajs_future": torch.randn(B, A, T, 4),
            "tracks_to_predict": torch.tensor([[0, 1, -1, -1, -1, -1, -1, -1]] * B),
            "obj_trajs": torch.randn(B, A, 11, 10),
            "obj_types": torch.randint(1, 4, (B, A)),
        }
        total, loss_dict = prediction_loss(output, batch, {"reg": 1.0, "score": 1.0, "vel": 0.2})
        assert "loss/dense" in loss_dict
        assert total.requires_grad

    def test_gradients_flow(self) -> None:
        """Ensure gradients flow through all loss components."""
        B, K, M, T, A = 1, 1, 2, 10, 4
        pred_trajs = torch.randn(B, K, M, T, 7, requires_grad=True)
        pred_scores = torch.randn(B, K, M, requires_grad=True)
        output = {"pred_trajs": pred_trajs, "pred_scores": pred_scores}
        batch = {
            "obj_trajs_future_local": torch.randn(B, A, T, 4),
            "obj_trajs_future_mask": torch.ones(B, A, T),
            "tracks_to_predict": torch.tensor([[0, -1, -1, -1, -1, -1, -1, -1]]),
            "obj_trajs": torch.randn(B, A, 11, 10),
            "obj_types": torch.randint(1, 4, (B, A)),
        }
        total, _ = prediction_loss(output, batch, {"reg": 1.0, "score": 1.0, "vel": 0.2})
        total.backward()
        assert pred_trajs.grad is not None
        assert pred_scores.grad is not None
