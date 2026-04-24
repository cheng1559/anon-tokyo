"""MTR-compatible attention with optional official CUDA operators."""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import constant_, xavier_uniform_


_OFFICIAL_KNN_BATCH_MLOGK = None
_OFFICIAL_ATTENTION_WEIGHT = None
_OFFICIAL_ATTENTION_VALUE = None


def _load_official_cuda_ops() -> None:
    global _OFFICIAL_KNN_BATCH_MLOGK, _OFFICIAL_ATTENTION_WEIGHT, _OFFICIAL_ATTENTION_VALUE
    if os.environ.get("ANON_TOKYO_DISABLE_MTR_CUDA_OPS"):
        return
    if _OFFICIAL_KNN_BATCH_MLOGK is not None and _OFFICIAL_ATTENTION_WEIGHT is not None:
        return

    try:
        from anon_tokyo.prediction.mtr.ops.attention.attention_utils_v2 import (
            attention_value_computation,
            attention_weight_computation,
        )
        from anon_tokyo.prediction.mtr.ops.knn.knn_utils import knn_batch_mlogk as official_knn_batch_mlogk
    except Exception:
        return

    _OFFICIAL_KNN_BATCH_MLOGK = official_knn_batch_mlogk
    _OFFICIAL_ATTENTION_WEIGHT = attention_weight_computation
    _OFFICIAL_ATTENTION_VALUE = attention_value_computation


def cuda_ops_available() -> bool:
    _load_official_cuda_ops()
    return _OFFICIAL_KNN_BATCH_MLOGK is not None and _OFFICIAL_ATTENTION_WEIGHT is not None


def _as_knn_xyz3(pos: Tensor) -> Tensor:
    if pos.shape[-1] >= 3:
        return pos[..., :3].contiguous()
    if pos.shape[-1] == 2:
        return F.pad(pos, (0, 1)).contiguous()
    raise ValueError(f"KNN positions must have at least 2 coordinates, got shape {tuple(pos.shape)}")


def knn_batch_mlogk(
    xyz: Tensor,
    query_xyz: Tensor,
    batch_idxs: Tensor,
    query_batch_offsets: Tensor,
    k: int,
) -> Tensor:
    """MTR-compatible batched KNN implemented with ``torch.cdist``.

    Args mirror ``mtr.ops.knn.knn_utils.knn_batch_mlogk``. Returned indices
    are local row indices within each batch with ``-1`` padding when a batch
    has fewer than ``k`` keys.
    """
    _load_official_cuda_ops()
    if (
        _OFFICIAL_KNN_BATCH_MLOGK is not None
        and xyz.is_cuda
        and query_xyz.is_cuda
        and batch_idxs.is_cuda
        and query_batch_offsets.is_cuda
        and xyz.shape[0] == query_xyz.shape[0]
        and k <= 128
    ):
        return _OFFICIAL_KNN_BATCH_MLOGK(
            _as_knn_xyz3(xyz),
            _as_knn_xyz3(query_xyz),
            batch_idxs.int().contiguous(),
            query_batch_offsets.int().contiguous(),
            k,
        )

    n_query = query_xyz.shape[0]
    out = torch.full((n_query, k), -1, dtype=torch.int32, device=query_xyz.device)
    if n_query == 0 or xyz.shape[0] == 0 or k <= 0:
        return out

    batch_idxs = batch_idxs.to(device=query_xyz.device)
    query_batch_offsets = query_batch_offsets.to(device=query_xyz.device)
    num_batches = query_batch_offsets.numel() - 1
    for b in range(num_batches):
        q_start = int(query_batch_offsets[b].item())
        q_end = int(query_batch_offsets[b + 1].item())
        if q_end <= q_start:
            continue
        key_idx = torch.nonzero(batch_idxs == b, as_tuple=False).flatten()
        if key_idx.numel() == 0:
            continue
        n_take = min(k, key_idx.numel())
        dist = torch.cdist(query_xyz[q_start:q_end, :2], xyz[key_idx, :2])
        _, local_idx = dist.topk(n_take, dim=-1, largest=False)
        out[q_start:q_end, :n_take] = local_idx.to(torch.int32)
    return out


class MultiheadAttention(nn.Module):
    """Subset of official MTR MultiheadAttention with compatible parameters."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        vdim: int | None = None,
        local_indices: bool = False,
        without_weight: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.v_head_dim = self.vdim // num_heads
        if self.head_dim * num_heads != embed_dim or self.v_head_dim * num_heads != self.vdim:
            raise ValueError("embed_dim and vdim must be divisible by num_heads")

        self.without_weight = without_weight
        if without_weight:
            self.register_parameter("in_proj_weight", None)
            self.register_parameter("in_proj_bias", None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None

        self.out_proj = nn.Linear(self.vdim, self.vdim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.in_proj_weight is not None:
            xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
        constant_(self.out_proj.bias, 0.0)

    def _project(self, query: Tensor, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.without_weight:
            return query, key, value
        assert self.in_proj_weight is not None
        if self.in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = self.in_proj_bias.chunk(3)
        w_q, w_k, w_v = self.in_proj_weight.chunk(3)
        return F.linear(query, w_q, b_q), F.linear(key, w_k, b_k), F.linear(value, w_v, b_v)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        **_: object,
    ) -> tuple[Tensor, Tensor]:
        tgt_len, batch_size, _ = query.shape
        src_len = key.shape[0]
        q, k, v = self._project(query, key, value)

        q = q.contiguous().view(tgt_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, batch_size * self.num_heads, self.v_head_dim).transpose(0, 1)

        q = q / math.sqrt(self.head_dim)
        scores = torch.bmm(q, k.transpose(-2, -1))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask[None], float("-inf"))
            else:
                scores = scores + attn_mask[None]
        if key_padding_mask is not None:
            padding_mask = key_padding_mask.bool().view(batch_size, 1, 1, src_len)
            padding_mask = padding_mask.expand(-1, self.num_heads, -1, -1).reshape(
                batch_size * self.num_heads, 1, src_len
            )
            scores = scores.masked_fill(padding_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        out = torch.bmm(weights, v)
        out = out.transpose(0, 1).contiguous().view(tgt_len, batch_size, self.vdim)
        weights = weights.view(batch_size, self.num_heads, tgt_len, src_len)
        return self.out_proj(out), weights.sum(dim=1) / self.num_heads


class MultiheadAttentionLocal(nn.Module):
    """Local indexed multi-head attention matching MTR's CUDA op semantics."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        without_weight: bool = False,
        vdim: int | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.v_head_dim = self.vdim // num_heads
        if self.head_dim * num_heads != embed_dim or self.v_head_dim * num_heads != self.vdim:
            raise ValueError("embed_dim and vdim must be divisible by num_heads")

        self.without_weight = without_weight
        if without_weight:
            self.register_parameter("in_proj_weight", None)
            self.register_parameter("in_proj_bias", None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(self.vdim, self.vdim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.in_proj_weight is not None:
            xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
        constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        index_pair: Tensor,
        query_batch_cnt: Tensor | None = None,
        key_batch_cnt: Tensor | None = None,
        index_pair_batch: Tensor | None = None,
        attn_mask: Tensor | None = None,
        vdim: int | None = None,
        local_indices: bool = False,
        **_: object,
    ) -> tuple[Tensor, Tensor]:
        vdim = self.vdim if vdim is None else vdim
        v_head_dim = vdim // self.num_heads
        total_query_len, _ = query.shape
        max_memory_len = index_pair.shape[1]

        if self.without_weight:
            q, k, v = query, key, value
        else:
            assert self.in_proj_weight is not None and self.in_proj_bias is not None
            q = F.linear(query, self.in_proj_weight[: self.embed_dim], self.in_proj_bias[: self.embed_dim])
            k = F.linear(
                key,
                self.in_proj_weight[self.embed_dim : self.embed_dim * 2],
                self.in_proj_bias[self.embed_dim : self.embed_dim * 2],
            )
            v = F.linear(
                value,
                self.in_proj_weight[self.embed_dim * 2 :],
                self.in_proj_bias[self.embed_dim * 2 :],
            )

        q = q.reshape(total_query_len, self.num_heads, self.head_dim) / math.sqrt(self.head_dim)
        k = k.reshape(-1, self.num_heads, self.head_dim)
        v = v.reshape(-1, self.num_heads, v_head_dim)

        invalid = index_pair < 0
        if attn_mask is not None:
            invalid = invalid | attn_mask.bool()
        if local_indices and index_pair_batch is not None and key_batch_cnt is not None:
            max_local_index = key_batch_cnt.long()[index_pair_batch.long()].to(index_pair.device)
            invalid = invalid | (index_pair >= max_local_index[:, None])
        all_invalid = invalid.all(dim=-1)
        safe_pair = index_pair.clamp(min=0).long()

        _load_official_cuda_ops()
        if (
            _OFFICIAL_ATTENTION_WEIGHT is not None
            and _OFFICIAL_ATTENTION_VALUE is not None
            and q.is_cuda
            and k.is_cuda
            and v.is_cuda
            and local_indices
            and query_batch_cnt is not None
            and key_batch_cnt is not None
            and index_pair_batch is not None
        ):
            original_dtype = q.dtype
            index_pair_for_op = index_pair.masked_fill(invalid, -1).int().contiguous()
            q_cnt = query_batch_cnt.int().contiguous()
            k_cnt = key_batch_cnt.int().contiguous()
            pair_batch = index_pair_batch.int().contiguous()
            weights = _OFFICIAL_ATTENTION_WEIGHT(
                q_cnt,
                k_cnt,
                pair_batch,
                index_pair_for_op,
                q.float().contiguous(),
                k.float().contiguous(),
            )
            weights = weights.masked_fill(index_pair_for_op[..., None] < 0, float("-inf"))
            if all_invalid.any():
                weights[all_invalid] = 0.0
            weights = F.softmax(weights, dim=1)
            weights = F.dropout(weights, p=self.dropout, training=self.training)
            weights = weights.masked_fill(index_pair_for_op[..., None] < 0, 0.0).contiguous()
            out = _OFFICIAL_ATTENTION_VALUE(
                q_cnt,
                k_cnt,
                pair_batch,
                index_pair_for_op,
                weights,
                v.float().contiguous(),
            )
            out = out.reshape(total_query_len, vdim).to(original_dtype)
            out = self.out_proj(out)
            return out, weights.sum(dim=-1).to(out.dtype) / self.num_heads

        if (
            query_batch_cnt is not None
            and key_batch_cnt is not None
            and index_pair_batch is not None
            and local_indices
            and query_batch_cnt.numel() > 0
            and bool((query_batch_cnt == query_batch_cnt[0]).all())
            and bool((key_batch_cnt == key_batch_cnt[0]).all())
        ):
            batch_size = int(key_batch_cnt.numel())
            num_query = int(query_batch_cnt[0].item())
            num_key = int(key_batch_cnt[0].item())
            if total_query_len == batch_size * num_query and k.shape[0] == batch_size * num_key:
                pair_b = safe_pair.view(batch_size, num_query, max_memory_len)
                k_b = k.view(batch_size, num_key, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                v_b = v.view(batch_size, num_key, self.num_heads, v_head_dim).permute(0, 2, 1, 3)
                gather_k_idx = pair_b[:, None, :, :, None].expand(
                    batch_size, self.num_heads, num_query, max_memory_len, self.head_dim
                )
                gather_v_idx = pair_b[:, None, :, :, None].expand(
                    batch_size, self.num_heads, num_query, max_memory_len, v_head_dim
                )
                gathered_k = torch.gather(
                    k_b[:, :, None, :, :].expand(batch_size, self.num_heads, num_query, num_key, self.head_dim),
                    dim=3,
                    index=gather_k_idx,
                ).permute(0, 2, 3, 1, 4).reshape(total_query_len, max_memory_len, self.num_heads, self.head_dim)
                gathered_v = torch.gather(
                    v_b[:, :, None, :, :].expand(batch_size, self.num_heads, num_query, num_key, v_head_dim),
                    dim=3,
                    index=gather_v_idx,
                ).permute(0, 2, 3, 1, 4).reshape(total_query_len, max_memory_len, self.num_heads, v_head_dim)
            else:
                key_offsets = F.pad(key_batch_cnt.long().cumsum(dim=0), (1, 0))[:-1]
                global_pair = safe_pair + key_offsets[index_pair_batch.long(), None]
                gathered_k = k[global_pair]
                gathered_v = v[global_pair]
        else:
            if local_indices and index_pair_batch is not None and key_batch_cnt is not None:
                key_offsets = F.pad(key_batch_cnt.long().cumsum(dim=0), (1, 0))[:-1]
                safe_pair = safe_pair + key_offsets[index_pair_batch.long(), None]
            gathered_k = k[safe_pair]
            gathered_v = v[safe_pair]

        weights = torch.einsum("qhd,qkhd->qkh", q, gathered_k)
        weights = weights.masked_fill(invalid[..., None], float("-inf"))
        if all_invalid.any():
            weights[all_invalid] = 0.0
        weights = F.softmax(weights, dim=1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        weights = weights.masked_fill(invalid[..., None], 0.0)

        out = torch.einsum("qkh,qkhd->qhd", weights, gathered_v).reshape(total_query_len, vdim)
        out = self.out_proj(out)
        return out, weights.sum(dim=-1) / self.num_heads
