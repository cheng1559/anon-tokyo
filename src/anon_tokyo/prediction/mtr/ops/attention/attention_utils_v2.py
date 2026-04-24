"""Autograd wrappers for the bundled MTR local attention CUDA operator."""

from __future__ import annotations

import torch
from torch.autograd import Function

from . import attention_cuda


class AttentionWeightComputation(Function):
    @staticmethod
    def forward(
        ctx,
        query_batch_cnt: torch.Tensor,
        key_batch_cnt: torch.Tensor,
        index_pair_batch: torch.Tensor,
        index_pair: torch.Tensor,
        query_features: torch.Tensor,
        key_features: torch.Tensor,
    ) -> torch.Tensor:
        assert query_batch_cnt.is_contiguous()
        assert key_batch_cnt.is_contiguous()
        assert index_pair_batch.is_contiguous()
        assert index_pair.is_contiguous()
        assert query_features.is_contiguous()
        assert key_features.is_contiguous()

        b = query_batch_cnt.shape[0]
        total_query_num, local_size = index_pair.size()
        total_key_num, nhead, hdim = key_features.size()
        output = torch.zeros((total_query_num, local_size, nhead), device=query_features.device, dtype=torch.float32)

        attention_cuda.attention_weight_computation_wrapper_v2(
            b,
            total_query_num,
            local_size,
            total_key_num,
            nhead,
            hdim,
            query_batch_cnt,
            key_batch_cnt,
            index_pair_batch,
            index_pair,
            query_features,
            key_features,
            output,
        )
        ctx.for_backwards = (
            b,
            total_query_num,
            local_size,
            total_key_num,
            nhead,
            hdim,
            query_batch_cnt,
            key_batch_cnt,
            index_pair_batch,
            index_pair,
            query_features,
            key_features,
        )
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (
            b,
            total_query_num,
            local_size,
            total_key_num,
            nhead,
            hdim,
            query_batch_cnt,
            key_batch_cnt,
            index_pair_batch,
            index_pair,
            query_features,
            key_features,
        ) = ctx.for_backwards

        grad_query_features = torch.zeros((total_query_num, nhead, hdim), device=grad_out.device, dtype=torch.float32)
        grad_key_features = torch.zeros((total_key_num, nhead, hdim), device=grad_out.device, dtype=torch.float32)
        attention_cuda.attention_weight_computation_grad_wrapper_v2(
            b,
            total_query_num,
            local_size,
            total_key_num,
            nhead,
            hdim,
            query_batch_cnt,
            key_batch_cnt,
            index_pair_batch,
            index_pair,
            query_features,
            key_features,
            grad_out.contiguous(),
            grad_query_features,
            grad_key_features,
        )
        return None, None, None, None, grad_query_features, grad_key_features


attention_weight_computation = AttentionWeightComputation.apply


class AttentionValueComputation(Function):
    @staticmethod
    def forward(
        ctx,
        query_batch_cnt: torch.Tensor,
        key_batch_cnt: torch.Tensor,
        index_pair_batch: torch.Tensor,
        index_pair: torch.Tensor,
        attn_weight: torch.Tensor,
        value_features: torch.Tensor,
    ) -> torch.Tensor:
        assert query_batch_cnt.is_contiguous()
        assert key_batch_cnt.is_contiguous()
        assert index_pair_batch.is_contiguous()
        assert index_pair.is_contiguous()
        assert attn_weight.is_contiguous()
        assert value_features.is_contiguous()

        b = query_batch_cnt.shape[0]
        total_query_num, local_size = index_pair.size()
        total_key_num, nhead, hdim = value_features.size()
        output = torch.zeros((total_query_num, nhead, hdim), device=value_features.device, dtype=torch.float32)

        attention_cuda.attention_value_computation_wrapper_v2(
            b,
            total_query_num,
            local_size,
            total_key_num,
            nhead,
            hdim,
            query_batch_cnt,
            key_batch_cnt,
            index_pair_batch,
            index_pair,
            attn_weight,
            value_features,
            output,
        )
        ctx.for_backwards = (
            b,
            total_query_num,
            local_size,
            total_key_num,
            nhead,
            hdim,
            query_batch_cnt,
            key_batch_cnt,
            index_pair_batch,
            index_pair,
            attn_weight,
            value_features,
        )
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (
            b,
            total_query_num,
            local_size,
            total_key_num,
            nhead,
            hdim,
            query_batch_cnt,
            key_batch_cnt,
            index_pair_batch,
            index_pair,
            attn_weight,
            value_features,
        ) = ctx.for_backwards

        grad_attn_weight = torch.zeros((total_query_num, local_size, nhead), device=grad_out.device, dtype=torch.float32)
        grad_value_features = torch.zeros((total_key_num, nhead, hdim), device=grad_out.device, dtype=torch.float32)
        attention_cuda.attention_value_computation_grad_wrapper_v2(
            b,
            total_query_num,
            local_size,
            total_key_num,
            nhead,
            hdim,
            query_batch_cnt,
            key_batch_cnt,
            index_pair_batch,
            index_pair,
            attn_weight,
            value_features,
            grad_out.contiguous(),
            grad_attn_weight,
            grad_value_features,
        )
        return None, None, None, None, grad_attn_weight, grad_value_features


attention_value_computation = AttentionValueComputation.apply
