"""Autograd wrapper for the bundled MTR KNN CUDA operator."""

from __future__ import annotations

import torch
from torch.autograd import Function

from . import knn_cuda


class KNNBatchMlogK(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, query_xyz: torch.Tensor, batch_idxs: torch.Tensor, query_batch_offsets: torch.Tensor, k: int):
        n = xyz.size(0)
        m = query_xyz.size(0)
        assert xyz.is_contiguous() and xyz.is_cuda
        assert query_xyz.is_contiguous() and query_xyz.is_cuda
        assert batch_idxs.is_contiguous() and batch_idxs.is_cuda
        assert query_batch_offsets.is_contiguous() and query_batch_offsets.is_cuda
        assert k <= 128
        idx = torch.zeros((n, k), device=xyz.device, dtype=torch.int32)
        knn_cuda.knn_batch_mlogk(xyz, query_xyz, batch_idxs, query_batch_offsets, idx, n, m, k)
        return idx

    @staticmethod
    def backward(ctx, grad_out=None):
        return None, None, None, None, None


knn_batch_mlogk = KNNBatchMlogK.apply
