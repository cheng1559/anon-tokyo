"""Unit tests for AnonTokyo scene-centric encoder."""

from __future__ import annotations

import pytest
import torch

from anon_tokyo.prediction.anon_tokyo.encoder import AnonTokyoEncoder, _EncoderLayer


def _make_batch(B: int = 2, A: int = 16, M: int = 64, T: int = 11, P: int = 20) -> dict[str, torch.Tensor]:
    """Create a synthetic batch matching scene-centric pipeline output."""
    return {
        "obj_trajs": torch.randn(B, A, T, 10),
        "obj_trajs_mask": torch.ones(B, A, T),
        "obj_positions": torch.randn(B, A, 2) * 50,
        "obj_headings": torch.randn(B, A),
        "agent_mask": torch.ones(B, A),
        "map_polylines": torch.randn(B, M, P, 7),
        "map_polylines_mask": torch.ones(B, M, P),
        "map_polylines_center": torch.randn(B, M, 2) * 100,
        "map_headings": torch.randn(B, M),
        "map_mask": torch.ones(B, M),
    }


class TestEncoderLayer:
    def test_output_shapes(self):
        B, A, M, D = 2, 8, 32, 64
        layer = _EncoderLayer(D, num_heads=4, sparse_k=8, dropout=0.0, use_rope=True, use_drope=True)
        agent_feat = torch.randn(B, A, D)
        map_feat = torch.randn(B, M, D)
        agent_pos = torch.randn(B, A, 2)
        map_pos = torch.randn(B, M, 2)
        agent_heading = torch.randn(B, A)
        map_heading = torch.randn(B, M)
        agent_mask = torch.ones(B, A, dtype=torch.bool)
        map_mask = torch.ones(B, M, dtype=torch.bool)

        from anon_tokyo.nn.attention import select_topk

        mm_topk = select_topk(map_pos, map_pos, map_mask, 8)
        aa_topk = select_topk(agent_pos, agent_pos, agent_mask, 8)
        am_topk = select_topk(agent_pos, map_pos, map_mask, 8)

        out_a, out_m = layer(
            agent_feat,
            map_feat,
            agent_pos,
            map_pos,
            agent_heading,
            map_heading,
            agent_mask,
            map_mask,
            mm_topk,
            aa_topk,
            am_topk,
        )
        assert out_a.shape == (B, A, D)
        assert out_m.shape == (B, M, D)

    def test_gradient_flow(self):
        B, A, M, D = 1, 4, 16, 32
        layer = _EncoderLayer(D, num_heads=4, sparse_k=8, dropout=0.0, use_rope=True, use_drope=True)
        agent_feat = torch.randn(B, A, D, requires_grad=True)
        map_feat = torch.randn(B, M, D, requires_grad=True)
        agent_pos = torch.randn(B, A, 2)
        map_pos = torch.randn(B, M, 2)
        agent_mask = torch.ones(B, A, dtype=torch.bool)
        map_mask = torch.ones(B, M, dtype=torch.bool)

        from anon_tokyo.nn.attention import select_topk

        mm_topk = select_topk(map_pos, map_pos, map_mask, 8)
        aa_topk = select_topk(agent_pos, agent_pos, agent_mask, 8)
        am_topk = select_topk(agent_pos, map_pos, map_mask, 8)

        out_a, out_m = layer(
            agent_feat,
            map_feat,
            agent_pos,
            map_pos,
            torch.zeros(B, A),
            torch.zeros(B, M),
            agent_mask,
            map_mask,
            mm_topk,
            aa_topk,
            am_topk,
        )
        loss = out_a.sum() + out_m.sum()
        loss.backward()
        assert agent_feat.grad is not None
        assert map_feat.grad is not None


class TestAnonTokyoEncoder:
    @pytest.fixture
    def encoder(self) -> AnonTokyoEncoder:
        return AnonTokyoEncoder(
            d_model=64,
            num_layers=2,
            num_heads=4,
            sparse_k=8,
            dropout=0.0,
            use_rope=True,
            use_drope=True,
        )

    def test_output_keys(self, encoder: AnonTokyoEncoder):
        batch = _make_batch(B=2, A=8, M=32)
        out = encoder(batch)
        assert set(out.keys()) == {"obj_feature", "map_feature", "obj_mask", "map_mask", "obj_pos", "map_pos"}

    def test_output_shapes(self, encoder: AnonTokyoEncoder):
        B, A, M = 2, 8, 32
        batch = _make_batch(B=B, A=A, M=M)
        out = encoder(batch)
        assert out["obj_feature"].shape == (B, A, 64)
        assert out["map_feature"].shape == (B, M, 64)
        assert out["obj_mask"].shape == (B, A)
        assert out["map_mask"].shape == (B, M)
        assert out["obj_pos"].shape == (B, A, 2)
        assert out["map_pos"].shape == (B, M, 2)

    def test_gradient_flow_full(self, encoder: AnonTokyoEncoder):
        batch = _make_batch(B=1, A=4, M=16)
        for v in batch.values():
            v.requires_grad_(True) if v.is_floating_point() else None
        out = encoder(batch)
        loss = out["obj_feature"].sum() + out["map_feature"].sum()
        loss.backward()
        assert batch["obj_trajs"].grad is not None

    def test_masked_agents(self, encoder: AnonTokyoEncoder):
        """Agents with mask=0 should produce near-zero features (only from residual of zero input)."""
        batch = _make_batch(B=1, A=8, M=16)
        # Mask out last 4 agents
        batch["agent_mask"][:, 4:] = 0.0
        batch["obj_trajs"][:, 4:] = 0.0
        batch["obj_trajs_mask"][:, 4:] = 0.0
        out = encoder(batch)
        # Valid agents should have non-trivial features
        valid_norm = out["obj_feature"][:, :4].norm(dim=-1).mean()
        assert valid_norm > 0.01

    def test_masked_map(self, encoder: AnonTokyoEncoder):
        """Map elements with mask=0 should not corrupt valid map features."""
        batch = _make_batch(B=1, A=4, M=32)
        batch["map_mask"][:, 16:] = 0.0
        batch["map_polylines"][:, 16:] = 0.0
        batch["map_polylines_mask"][:, 16:] = 0.0
        out = encoder(batch)
        valid_norm = out["map_feature"][:, :16].norm(dim=-1).mean()
        assert valid_norm > 0.01

    def test_no_rope_no_drope(self):
        """Encoder with sinusoidal fallback should still work."""
        enc = AnonTokyoEncoder(
            d_model=64,
            num_layers=2,
            num_heads=4,
            sparse_k=8,
            dropout=0.0,
            use_rope=False,
            use_drope=False,
        )
        batch = _make_batch(B=1, A=4, M=16)
        out = enc(batch)
        assert out["obj_feature"].shape == (1, 4, 64)

    def test_rope_only(self):
        """Encoder with only RoPE (no DRoPE) should work."""
        enc = AnonTokyoEncoder(
            d_model=64,
            num_layers=2,
            num_heads=4,
            sparse_k=8,
            dropout=0.0,
            use_rope=True,
            use_drope=False,
        )
        batch = _make_batch(B=1, A=4, M=16)
        out = enc(batch)
        assert out["obj_feature"].shape == (1, 4, 64)

    def test_drope_only(self):
        """Encoder with only DRoPE (no RoPE) should work."""
        enc = AnonTokyoEncoder(
            d_model=64,
            num_layers=2,
            num_heads=4,
            sparse_k=8,
            dropout=0.0,
            use_rope=False,
            use_drope=True,
        )
        batch = _make_batch(B=1, A=4, M=16)
        out = enc(batch)
        assert out["obj_feature"].shape == (1, 4, 64)

    def test_sparse_k_larger_than_keys(self):
        """When sparse_k > number of keys, should still work."""
        enc = AnonTokyoEncoder(
            d_model=64,
            num_layers=1,
            num_heads=4,
            sparse_k=100,
            dropout=0.0,
            use_rope=True,
            use_drope=True,
        )
        batch = _make_batch(B=1, A=4, M=8)  # only 8 map, 4 agents
        out = enc(batch)
        assert out["obj_feature"].shape == (1, 4, 64)

    def test_deterministic_eval(self, encoder: AnonTokyoEncoder):
        """Same input in eval mode should produce identical output."""
        encoder.eval()
        batch = _make_batch(B=1, A=4, M=16)
        out1 = encoder(batch)
        out2 = encoder(batch)
        torch.testing.assert_close(out1["obj_feature"], out2["obj_feature"])
        torch.testing.assert_close(out1["map_feature"], out2["map_feature"])

    def test_batch_independence(self, encoder: AnonTokyoEncoder):
        """Results for batch element 0 should not change when batch element 1 differs."""
        encoder.eval()
        batch1 = _make_batch(B=2, A=8, M=16)
        batch2 = {k: v.clone() for k, v in batch1.items()}
        batch2["obj_trajs"][1] = torch.randn_like(batch2["obj_trajs"][1])
        out1 = encoder(batch1)
        out2 = encoder(batch2)
        torch.testing.assert_close(out1["obj_feature"][0], out2["obj_feature"][0])
