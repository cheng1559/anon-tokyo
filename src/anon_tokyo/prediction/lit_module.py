"""LightningModule for open-loop trajectory prediction."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

from anon_tokyo.prediction.loss import mtr_prediction_loss, prediction_loss
from anon_tokyo.prediction.metrics import compute_prediction_metrics


class PredictionModule(L.LightningModule):
    """Lightning wrapper for open-loop trajectory prediction training."""

    def __init__(
        self,
        model: nn.Module,
        optimizer_class: str = "torch.optim.AdamW",
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler_kwargs: dict[str, Any] | None = None,
        loss_weights: dict[str, float] | None = None,
        compile_model: bool = False,
        train_metrics_interval: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        if compile_model:
            self.model = torch.compile(self.model)

        self.loss_weights = loss_weights or {"reg": 1.0, "score": 1.0, "vel": 0.2}
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or {"lr": 1e-4, "weight_decay": 0.01}
        self._scheduler_kwargs = scheduler_kwargs or {
            "warmup_ratio": 0.1,
            "eta_min": 1e-6,
        }
        self.train_metrics_interval = train_metrics_interval

    # ── Training ──────────────────────────────────────────────────────────

    def _is_agent_centric(self, output: dict[str, Tensor]) -> bool:
        return "batch_idx" in output

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        output = self.model(batch)

        if self._is_agent_centric(output):
            total_loss, loss_dict = mtr_prediction_loss(output, self.loss_weights)
            bs = output["center_gt_trajs"].shape[0]
        else:
            total_loss, loss_dict = prediction_loss(output, batch, self.loss_weights)
            bs = batch["obj_trajs"].shape[0]

        self.log_dict(
            {f"train/{k}": v for k, v in loss_dict.items()},
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=bs,
        )
        self.log("train/loss", loss_dict["loss/total"], prog_bar=True, batch_size=bs)

        # Compute trajectory metrics periodically to avoid extra per-step work.
        log_train_metrics = self.train_metrics_interval > 0 and batch_idx % self.train_metrics_interval == 0
        if log_train_metrics and self._is_agent_centric(output):
            with torch.no_grad():
                pred_trajs = output["pred_trajs"]
                pred_scores = output["pred_scores"]
                center_gt = output["center_gt_trajs"]
                center_mask = output["center_gt_mask"]
                pred_xy = pred_trajs[:, :, :, 0:2].unsqueeze(1)
                scores = pred_scores.unsqueeze(1)
                gt_xy = center_gt[:, :, 0:2].unsqueeze(1)
                mask = center_mask.unsqueeze(1)
                metrics = compute_prediction_metrics(pred_xy, scores, gt_xy, mask)
                for name, val in metrics.items():
                    self.log(
                        f"train/{name}",
                        val.mean(),
                        on_step=True,
                        on_epoch=False,
                        batch_size=bs,
                    )
        elif log_train_metrics:
            with torch.no_grad():
                B = output["pred_trajs"].shape[0]
                K = output["pred_trajs"].shape[1]
                ttp = batch["tracks_to_predict"]
                ttp_clamped = ttp.clamp(min=0)
                b_idx = torch.arange(B, device=ttp.device)[:, None].expand(B, K)
                gt_local = batch["obj_trajs_future_local"][b_idx, ttp_clamped[:, :K]]
                gt_mask = batch["obj_trajs_future_mask"][b_idx, ttp_clamped[:, :K]]
                pred_xy = output["pred_trajs"][:, :, :, :, 0:2]
                metrics = compute_prediction_metrics(
                    pred_xy,
                    output["pred_scores"],
                    gt_local[:, :, :, 0:2],
                    gt_mask,
                )
                ttp_valid = (ttp[:, :K] >= 0).float()
                valid_count = ttp_valid.sum().clamp(min=1)
                for name, val in metrics.items():
                    self.log(
                        f"train/{name}",
                        (val * ttp_valid).sum() / valid_count,
                        on_step=True,
                        on_epoch=False,
                        batch_size=bs,
                    )

        return total_loss

    # ── Validation ────────────────────────────────────────────────────────

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        output = self.model(batch)

        if self._is_agent_centric(output):
            _, loss_dict = mtr_prediction_loss(output, self.loss_weights)
            bs = output["center_gt_trajs"].shape[0]
            self.log_dict(
                {f"val/{k}": v for k, v in loss_dict.items()},
                on_epoch=True,
                sync_dist=True,
                batch_size=bs,
            )

            # Metrics: use final layer predictions
            pred_trajs = output["pred_trajs"]  # [K, M, T, 7]
            pred_scores = output["pred_scores"]  # [K, M]
            center_gt = output["center_gt_trajs"]  # [K, T, 4]
            center_mask = output["center_gt_mask"]  # [K, T]

            # Add dummy K (agents) dim → [K, 1, M, T, 2] / [K, 1, M]
            pred_xy = pred_trajs[:, :, :, 0:2].unsqueeze(1)
            scores = pred_scores.unsqueeze(1)
            gt_xy = center_gt[:, :, 0:2].unsqueeze(1)
            mask = center_mask.unsqueeze(1)

            metrics = compute_prediction_metrics(pred_xy, scores, gt_xy, mask)
            for name, val in metrics.items():
                self.log(
                    f"val/{name}",
                    val.mean(),
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=bs,
                )
        else:
            _, loss_dict = prediction_loss(output, batch, self.loss_weights)
            self.log_dict(
                {f"val/{k}": v for k, v in loss_dict.items()},
                on_epoch=True,
                sync_dist=True,
                batch_size=batch["obj_trajs"].shape[0],
            )

            B = output["pred_trajs"].shape[0]
            K = output["pred_trajs"].shape[1]
            ttp = batch["tracks_to_predict"]
            ttp_clamped = ttp.clamp(min=0)

            b_idx = torch.arange(B, device=ttp.device)[:, None].expand(B, K)
            gt_local = batch["obj_trajs_future_local"][b_idx, ttp_clamped[:, :K]]
            gt_mask = batch["obj_trajs_future_mask"][b_idx, ttp_clamped[:, :K]]

            pred_xy = output["pred_trajs"][:, :, :, :, 0:2]
            metrics = compute_prediction_metrics(
                pred_xy,
                output["pred_scores"],
                gt_local[:, :, :, 0:2],
                gt_mask,
            )

            ttp_valid = (ttp[:, :K] >= 0).float()
            valid_count = ttp_valid.sum().clamp(min=1)
            for name, val in metrics.items():
                self.log(
                    f"val/{name}",
                    (val * ttp_valid).sum() / valid_count,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=B,
                )

    # ── Optimizer / Scheduler ─────────────────────────────────────────────

    def configure_optimizers(self) -> dict[str, Any]:
        module_path, _, cls_name = self._optimizer_class.rpartition(".")
        import importlib

        opt_cls = getattr(importlib.import_module(module_path), cls_name)
        optimizer = opt_cls(self.parameters(), **self._optimizer_kwargs)

        sk = self._scheduler_kwargs
        total_steps = self.trainer.estimated_stepping_batches
        max_epochs = self.trainer.max_epochs or 30
        steps_per_epoch = max(total_steps // max_epochs, 1)
        lr_base = self._optimizer_kwargs.get("lr", 1e-4)
        lr_clip = sk.get("lr_clip", 1e-6)

        # Step-decay schedule matching official MTR:
        # constant LR until decay epochs, then multiply by lr_decay at each.
        decay_epochs = sk.get("decay_step_list", [22, 24, 26, 28])
        lr_decay = sk.get("lr_decay", 0.5)
        decay_steps = [e * steps_per_epoch for e in decay_epochs]

        def lr_lambda(cur_step: int) -> float:
            factor = 1.0
            for ds in decay_steps:
                if cur_step >= ds:
                    factor *= lr_decay
            return max(factor, lr_clip / lr_base)

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
