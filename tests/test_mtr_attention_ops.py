"""Tests for MTR-compatible KNN/local attention wrappers."""

from __future__ import annotations

import torch

from anon_tokyo.prediction.mtr.attention import knn_batch_mlogk


def test_knn_batch_mlogk_returns_local_indices() -> None:
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
        ]
    )
    batch_idxs = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32)
    query_batch_offsets = torch.tensor([0, 3, 6], dtype=torch.int32)

    idx = knn_batch_mlogk(xyz, xyz, batch_idxs, query_batch_offsets, k=2)

    assert idx.shape == (6, 2)
    assert idx.dtype == torch.int32
    assert idx[:3].max() < 3
    assert idx[3:].max() < 3
    assert 3 not in idx[3:].flatten().tolist()


def test_knn_batch_mlogk_accepts_2d_positions() -> None:
    xyz = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [10.0, 0.0],
            [11.0, 0.0],
        ]
    )
    batch_idxs = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    query_batch_offsets = torch.tensor([0, 2, 4], dtype=torch.int32)

    idx = knn_batch_mlogk(xyz, xyz, batch_idxs, query_batch_offsets, k=2)

    assert idx.shape == (4, 2)
    assert idx[:2].max() < 2
    assert idx[2:].max() < 2
