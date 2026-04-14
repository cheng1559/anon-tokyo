"""Per-agent ego-centric preprocessing for MTR.

Converts a scene-centric batch (SDC coordinate frame, ``[B, ...]``) to a
per-agent ego-centric batch (``[K_total, ...]``) where each sample is centred
on one agent-to-predict with its heading aligned to the +x axis.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _rotate_2d(xy: Tensor, angle: Tensor) -> Tensor:
    """Rotate 2-D vectors by *angle* (rad).

    Args:
        xy: ``[..., 2]``
        angle: broadcastable with ``xy[..., 0]``
    """
    c = torch.cos(angle).unsqueeze(-1)
    s = torch.sin(angle).unsqueeze(-1)
    x, y = xy[..., 0:1], xy[..., 1:2]
    return torch.cat([x * c - y * s, x * s + y * c], dim=-1)


@torch.no_grad()
def agent_centric_preprocess(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    """Expand scene-centric batch to per-agent ego-centric for MTR.

    For every valid entry in ``tracks_to_predict`` the whole scene is
    re-centred on that agent and rotated so its heading points along +x.

    Batch dimension changes from ``B`` (scenes) to ``K_total`` (agents to
    predict summed over scenes).

    Input keys (``[B, ...]``):
        obj_trajs            [B, A, T, 10]  x y z dx dy dz sin_h cos_h vx vy
        obj_trajs_mask       [B, A, T]
        obj_positions        [B, A, 2]
        obj_headings         [B, A]
        obj_types            [B, A]
        agent_mask           [B, A]
        tracks_to_predict    [B, K_max]     (-1 = padding)
        map_polylines        [B, M, P, 7]   x y z dir_x dir_y dir_z type
        map_polylines_mask   [B, M, P]
        map_polylines_center [B, M, 2]
        map_headings         [B, M]
        map_mask             [B, M]
        obj_trajs_future      [B, A, T_f, 4]  x y vx vy
        obj_trajs_future_mask [B, A, T_f]

    Returns dict with ``[K_total, ...]`` tensors plus:
        track_index_to_predict  [K_total]   index of centre agent in A dim
        center_obj_type         [K_total]   agent type
        batch_sample_count      [B]         agents per scene
        batch_idx               [K_total]   original scene index
    """
    ttp = batch["tracks_to_predict"]  # [B, K_max]
    valid = ttp >= 0
    B, K_max = ttp.shape
    device = ttp.device

    # Centre agent states (in SDC frame)
    obj_pos = batch["obj_positions"]  # [B, A, 2]
    obj_hd = batch["obj_headings"]  # [B, A]
    obj_tp = batch["obj_types"]  # [B, A]

    ttp_c = ttp.clamp(min=0)
    bi = torch.arange(B, device=device)[:, None].expand(B, K_max)
    c_pos = obj_pos[bi, ttp_c]  # [B, K_max, 2]
    c_head = obj_hd[bi, ttp_c]  # [B, K_max]
    c_type = obj_tp[bi, ttp_c]  # [B, K_max]

    # Flatten to valid entries
    vb, vk = valid.nonzero(as_tuple=True)
    K_total = len(vb)
    cp = c_pos[vb, vk]  # [K, 2]
    ch = c_head[vb, vk]  # [K]
    ct = c_type[vb, vk]  # [K]
    cidx = ttp_c[vb, vk]  # [K]
    neg = -ch

    batch_sample_count = valid.sum(dim=1)  # [B]

    def sel(t: Tensor) -> Tensor:
        return t[vb]

    # -- obj_trajs [K, A, T, 10] --
    # features: x y z dx dy dz sin_h cos_h vx vy
    ot = sel(batch["obj_trajs"]).clone()
    # translate & rotate positions [0:2]
    ot[..., 0:2] -= cp[:, None, None, :]
    ot[..., 0:2] = _rotate_2d(ot[..., 0:2], neg[:, None, None])
    # rotate velocities [8:10]
    ot[..., 8:10] = _rotate_2d(ot[..., 8:10], neg[:, None, None])
    # adjust heading encoding [6:8]  —  sin(h-θ), cos(h-θ)
    sin_h = ot[..., 6].clone()
    cos_h = ot[..., 7].clone()
    cos_t = torch.cos(ch)[:, None, None]
    sin_t = torch.sin(ch)[:, None, None]
    ot[..., 6] = sin_h * cos_t - cos_h * sin_t
    ot[..., 7] = cos_h * cos_t + sin_h * sin_t

    # -- obj_positions [K, A, 2] --
    op = sel(batch["obj_positions"]).clone()
    op -= cp[:, None, :]
    op = _rotate_2d(op, neg[:, None])

    # -- obj_headings [K, A] --
    oh = sel(batch["obj_headings"]).clone() - ch[:, None]

    # -- map_polylines [K, M, P, 7]: x y z dir_x dir_y dir_z type --
    mp = sel(batch["map_polylines"]).clone()
    mp[..., 0:2] -= cp[:, None, None, :]
    mp[..., 0:2] = _rotate_2d(mp[..., 0:2], neg[:, None, None])
    mp[..., 3:5] = _rotate_2d(mp[..., 3:5], neg[:, None, None])

    # -- map_polylines_center [K, M, 2] --
    mc = sel(batch["map_polylines_center"]).clone()
    mc -= cp[:, None, :]
    mc = _rotate_2d(mc, neg[:, None])

    # -- map_headings [K, M] --
    mh = sel(batch["map_headings"]).clone() - ch[:, None]

    # -- obj_trajs_future [K, A, T_f, 4]: x y vx vy --
    ft = sel(batch["obj_trajs_future"]).clone()
    ft[..., 0:2] -= cp[:, None, None, :]
    ft[..., 0:2] = _rotate_2d(ft[..., 0:2], neg[:, None, None])
    ft[..., 2:4] = _rotate_2d(ft[..., 2:4], neg[:, None, None])

    out: dict[str, Tensor] = {
        "obj_trajs": ot,
        "obj_trajs_mask": sel(batch["obj_trajs_mask"]),
        "obj_positions": op,
        "obj_headings": oh,
        "obj_types": sel(batch["obj_types"]),
        "agent_mask": sel(batch["agent_mask"]),
        "map_polylines": mp,
        "map_polylines_mask": sel(batch["map_polylines_mask"]),
        "map_polylines_center": mc,
        "map_headings": mh,
        "map_mask": sel(batch["map_mask"]),
        "obj_trajs_future": ft,
        "obj_trajs_future_mask": sel(batch["obj_trajs_future_mask"]),
        # tracks_to_predict as [K, 1] for encoder compatibility
        "tracks_to_predict": cidx.unsqueeze(-1),
        # MTR-specific metadata
        "track_index_to_predict": cidx,
        "center_obj_type": ct,
        "batch_sample_count": batch_sample_count,
        "batch_idx": vb,
    }

    if "obj_trajs_future_local" in batch:
        out["obj_trajs_future_local"] = sel(batch["obj_trajs_future_local"])

    return out
