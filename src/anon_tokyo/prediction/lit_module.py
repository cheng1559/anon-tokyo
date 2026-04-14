"""LightningModule for open-loop trajectory prediction."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import lightning as L
import torch
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from anon_tokyo.prediction.loss import prediction_loss
from anon_tokyo.prediction.metrics import compute_prediction_metrics


@runtime_checkable
class PredictionModel(Protocol):
    """Interface that concrete prediction models must implement."""

    def __call__(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Forward pass.

        Returns:
            ``pred_trajs``: ``[B, K, num_modes, T, 7]``
                (μx, μy, log_σ1, log_σ2, ρ, vx, vy) in agent-local frame.
            ``pred_scores``: ``[B, K, num_modes]`` — raw logits.
        """
        ...


def _import_class(dotpath: str) -> type:
    """Import a class from a dotted path like ``pkg.module.ClassName``."""
    module_path, _, class_name = dotpath.rpartition(".")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class PredictionModule(L.LightningModule):
    """Lightning wrapper for open-loop trajectory prediction training."""

    def __init__(
        self,
        model_class: str,
        model_kwargs: dict[str, Any] | None = None,
        optimizer_class: str = "torch.optim.AdamW",
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler_kwargs: dict[str, Any] | None = None,
        loss_weights: dict[str, float] | None = None,
        intention_points_file: str | None = None,
        compile_model: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Build model
        mk = dict(model_kwargs or {})
        if intention_points_file is not None:
            mk["intention_points_file"] = intention_points_file
        cls = _import_class(model_class)
        self.model = cls(**mk)

        if compile_model:
            self.model = torch.compile(self.model, mode="reduce-overhead")

        self.loss_weights = loss_weights or {"reg": 1.0, "score": 1.0, "vel": 0.2}
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or {"lr": 1e-4, "weight_decay": 0.01}
        self._scheduler_kwargs = scheduler_kwargs or {
            "warmup_ratio": 0.1,
            "eta_min": 1e-6,
        }

    # ── Training ──────────────────────────────────────────────────────────

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        output = self.model(batch)
        total_loss, loss_dict = prediction_loss(output, batch, self.loss_weights)
        self.log_dict(
            {f"train/{k}": v for k, v in loss_dict.items()},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=batch["obj_trajs"].shape[0],
        )
        return total_loss

    # ── Validation ────────────────────────────────────────────────────────

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        output = self.model(batch)
        _, loss_dict = prediction_loss(output, batch, self.loss_weights)
        self.log_dict(
            {f"val/{k}": v for k, v in loss_dict.items()},
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["obj_trajs"].shape[0],
        )

        # Metrics over tracks_to_predict
        B = output["pred_trajs"].shape[0]
        K = output["pred_trajs"].shape[1]
        ttp = batch["tracks_to_predict"]
        ttp_clamped = ttp.clamp(min=0)

        b_idx = torch.arange(B, device=ttp.device)[:, None].expand(B, K)
        gt_local = batch["obj_trajs_future_local"][b_idx, ttp_clamped[:, :K]]
        gt_mask = batch["obj_trajs_future_mask"][b_idx, ttp_clamped[:, :K]]

        pred_xy = output["pred_trajs"][:, :, :, :, 0:2]
        metrics = compute_prediction_metrics(pred_xy, output["pred_scores"], gt_local[:, :, :, 0:2], gt_mask)

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
        opt_cls = _import_class(self._optimizer_class)
        optimizer = opt_cls(self.parameters(), **self._optimizer_kwargs)

        sk = self._scheduler_kwargs
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * sk.get("warmup_ratio", 0.1))
        warmup_steps = max(warmup_steps, 1)

        warmup = LinearLR(
            optimizer,
            start_factor=sk.get("start_factor", 0.1),
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=sk.get("eta_min", 1e-6),
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
