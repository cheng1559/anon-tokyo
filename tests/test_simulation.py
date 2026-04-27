from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from anon_tokyo.data.datamodule import collate_fn
from anon_tokyo.data.transforms import simulation_transform
from anon_tokyo.simulation.dynamics import JerkPncConfig, JerkPncModel
from anon_tokyo.simulation.env import ClosedLoopEnv, ClosedLoopEnvConfig
from anon_tokyo.simulation.agent_centric.model import AgentCentricModel
from anon_tokyo.simulation.anon_tokyo.model import AnonTokyoModel
from anon_tokyo.simulation.ppo import PPOConfig, PPOTrainer
from anon_tokyo.simulation.query_centric.model import QueryCentricModel
from anon_tokyo.simulation.rewards import RewardConfig, compute_rewards


def _make_scenario(num_agents: int = 4, num_steps: int = 21) -> dict[str, np.ndarray]:
    trajs = np.zeros((num_agents, num_steps, 10), dtype=np.float32)
    for a in range(num_agents):
        trajs[a, :, 0] = np.linspace(0.0, 2.0, num_steps) + a * 5.0
        trajs[a, :, 1] = float(a)
        trajs[a, :, 3] = 4.5
        trajs[a, :, 4] = 2.0
        trajs[a, :, 5] = 1.5
        trajs[a, :, 6] = 0.0
        trajs[a, :, 7] = 1.0
        trajs[a, :, 8] = 0.0
        trajs[a, :, 9] = 1.0

    boundary = np.zeros((8, 7), dtype=np.float32)
    boundary[:, 0] = np.linspace(-10.0, 40.0, 8)
    boundary[:, 1] = 8.0
    boundary[:, 3] = 1.0
    boundary[:, 6] = 15.0

    return {
        "scenario_id": np.array("sim_test", dtype="U"),
        "timestamps": np.arange(num_steps, dtype=np.float64) * 0.1,
        "current_time_index": np.int32(3),
        "sdc_track_index": np.int32(0),
        "object_id": np.arange(num_agents, dtype=np.int64),
        "object_type": np.ones(num_agents, dtype=np.int8),
        "trajs": trajs,
        "map_polylines": boundary,
        "traffic_lights": np.zeros((num_steps, 0, 5), dtype=np.float32),
        "tracks_to_predict": np.array([1, 2], dtype=np.int32),
        "predict_difficulty": np.zeros(2, dtype=np.int32),
    }


def _tensor_sample(sample: dict) -> dict:
    out = {}
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            out[key] = torch.from_numpy(value)
        else:
            out[key] = value
    return out


def _batch(max_agents: int = 8) -> dict:
    return collate_fn([_tensor_sample(simulation_transform(_make_scenario(), max_agents=max_agents, max_polylines=8))])


def test_simulation_transform_adds_controlled_mask_and_full_trajs() -> None:
    out = simulation_transform(_make_scenario(), max_agents=8, max_polylines=8)
    assert out["obj_trajs_full"].shape == (8, 21, 10)
    assert out["obj_trajs_full_mask"].shape == (8, 21)
    assert out["controlled_mask"].dtype == np.bool_
    valid_ttp = out["tracks_to_predict"][out["tracks_to_predict"] >= 0]
    assert out["controlled_mask"][valid_ttp].all()
    assert int(out["current_time_index"]) == 3


def test_simulation_transform_all_agents_control_mode_controls_current_valid_agents() -> None:
    scenario = _make_scenario()
    current_t = int(scenario["current_time_index"])
    scenario["trajs"][3, current_t, 9] = 0.0

    out = simulation_transform(scenario, max_agents=8, max_polylines=8, control_mode="all_agents")
    expected = out["agent_mask"].astype(bool) & out["obj_trajs_full_mask"][:, current_t].astype(bool)

    np.testing.assert_array_equal(out["controlled_mask"], expected)
    assert out["controlled_mask"].sum() == 3


def test_jerk_pnc_longitudinal_and_lateral_actions() -> None:
    model = JerkPncModel(JerkPncConfig(dt=0.1, max_jerk_long=3.0, max_jerk_lat=1.0))
    positions = torch.zeros(1, 1, 2)
    velocities = torch.tensor([[[1.0, 0.0]]])
    headings = torch.zeros(1, 1)
    sizes = torch.tensor([[[4.5, 2.0]]])
    zeros = torch.zeros(1, 1)
    out = model.step(positions, velocities, headings, sizes, zeros, zeros, zeros, zeros, torch.tensor([[[3.0, 1.0]]]))
    assert out["velocities"][0, 0, 0] > velocities[0, 0, 0]
    assert out["headings"][0, 0] > 0
    assert out["steering"][0, 0] > 0
    assert out["jerk_lat"][0, 0] == 1.0


def test_jerk_pnc_does_not_turn_in_place_from_lateral_jerk() -> None:
    model = JerkPncModel(JerkPncConfig(dt=0.1, max_jerk_lat=1.0))
    positions = torch.zeros(1, 1, 2)
    velocities = torch.zeros(1, 1, 2)
    headings = torch.zeros(1, 1)
    sizes = torch.tensor([[[4.5, 2.0]]])
    a_long = torch.zeros(1, 1)
    a_lat = torch.zeros(1, 1)
    steering = torch.zeros(1, 1)
    yaw_rate = torch.zeros(1, 1)
    action = torch.tensor([[[0.0, 1.0]]])

    for _ in range(80):
        out = model.step(positions, velocities, headings, sizes, a_long, a_lat, steering, yaw_rate, action)
        positions = out["positions"]
        velocities = out["velocities"]
        headings = out["headings"]
        a_long = out["a_long"]
        a_lat = out["a_lat"]
        steering = out["steering"]
        yaw_rate = out["yaw_rate"]

    torch.testing.assert_close(positions, torch.zeros_like(positions))
    torch.testing.assert_close(velocities, torch.zeros_like(velocities))
    torch.testing.assert_close(yaw_rate, torch.zeros_like(yaw_rate))
    torch.testing.assert_close(headings, torch.zeros_like(headings))


def test_env_controls_tracks_to_predict_and_replays_others() -> None:
    batch = _batch()
    env = ClosedLoopEnv(ClosedLoopEnvConfig(device="cpu", num_steps=2, history_steps=4))
    obs = env.reset(batch)
    controlled = obs["controlled_mask"].bool()[0]
    action = torch.zeros(1, 8, 2)
    action[:, controlled, 0] = 3.0
    next_obs, _, _, _ = env.step(action)

    log_next = batch["obj_trajs_full"][0, :, 4, 0:2]
    non_controlled = (~controlled) & batch["agent_mask"][0].bool()
    torch.testing.assert_close(next_obs["obj_positions"][0, non_controlled], log_next[non_controlled])
    assert not torch.allclose(next_obs["obj_positions"][0, controlled], log_next[controlled])


def _reward_state() -> dict[str, torch.Tensor]:
    positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [8.0, 0.0]]])
    velocities = torch.tensor([[[3.0, 0.0], [0.0, 0.0], [-3.0, 0.0]]])
    headings = torch.zeros(1, 3)
    sizes = torch.tensor([[[4.5, 2.0], [4.5, 2.0], [4.5, 2.0]]])
    map_polys = torch.zeros(1, 1, 2, 7)
    map_polys[0, 0, :, 0] = torch.tensor([-3.0, 3.0])
    map_polys[0, 0, :, 1] = -1.1
    map_polys[0, 0, :, 6] = 15.0
    return {
        "positions": positions,
        "prev_positions": positions,
        "velocities": velocities,
        "headings": headings,
        "sizes": sizes,
        "valid_mask": torch.ones(1, 3, dtype=torch.bool),
        "controlled_mask": torch.tensor([[True, False, False]]),
        "map_polylines": map_polys,
        "map_polylines_mask": torch.ones(1, 1, 2),
        "goal_positions": torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]),
        "goal_reached": torch.zeros(1, 3, dtype=torch.bool),
        "a_long": torch.zeros(1, 3),
        "a_lat": torch.zeros(1, 3),
        "jerk_long": torch.zeros(1, 3),
        "jerk_lat": torch.zeros(1, 3),
    }


def test_rewards_collision_offroad_ttc_goal() -> None:
    state = _reward_state()
    cfg = RewardConfig(offroad_distance_threshold=0.2, ttc_horizon=4.0)
    reward, done, info, goal_reached = compute_rewards(state, cfg)
    assert info["collision"][0, 0]
    assert info["offroad"][0, 0]
    assert info["goal_reached"][0, 0]
    assert goal_reached[0, 0]
    assert done[0, 0]
    assert reward[0, 0] < 0

    state = _reward_state()
    state["positions"] = torch.tensor([[[0.0, 0.0], [8.0, 0.0], [20.0, 0.0]]])
    state["map_polylines"][:, :, :, 1] = 10.0
    _, _, info, _ = compute_rewards(state, cfg)
    assert info["ttc_alert"][0, 0]


def test_offroad_ignores_solid_lane_lines() -> None:
    state = _reward_state()
    state["positions"] = torch.tensor([[[0.0, 0.0], [8.0, 0.0], [20.0, 0.0]]])
    state["controlled_mask"] = torch.tensor([[True, False, False]])
    state["map_polylines"][0, 0, :, 6] = 7.0

    _, done, info, _ = compute_rewards(state, RewardConfig(offroad_distance_threshold=0.2))

    assert not info["offroad"][0, 0]
    assert not done[0, 0]


def test_agent_centric_policy_forward_shapes() -> None:
    batch = _batch(max_agents=4)
    env = ClosedLoopEnv(ClosedLoopEnvConfig(device="cpu", num_steps=2, history_steps=5))
    obs = env.reset(batch)
    policy = AgentCentricModel(
        d_model=32,
        num_heads=4,
        max_context_agents=4,
        max_lanes=4,
        history_steps=5,
        no_goal_allowed=True,
    )

    action, logprob, entropy, value = policy(obs, sampling_method="mean")

    assert action.shape == (1, 4, 2)
    assert logprob.shape == (1, 4)
    assert entropy.shape == (1, 4)
    assert value.shape == (1, 4)
    assert torch.all(action[..., 0] >= -5.0)
    assert torch.all(action[..., 0] <= 3.0)
    assert torch.all(action[..., 1] >= -1.0)
    assert torch.all(action[..., 1] <= 1.0)


def test_agent_centric_forward_and_checkpoint_keys() -> None:
    batch = _batch(max_agents=4)
    env = ClosedLoopEnv(ClosedLoopEnvConfig(device="cpu", num_steps=2, history_steps=5))
    obs = env.reset(batch)
    policy = AgentCentricModel(
        d_model=32,
        num_heads=4,
        max_context_agents=4,
        max_lanes=4,
        history_steps=5,
        no_goal_allowed=True,
    )

    checkpoint_state = {f"model.{key}": value for key, value in policy.model.model.state_dict().items()}
    missing, unexpected = policy.load_state_dict(checkpoint_state, strict=True)
    assert not missing
    assert not unexpected

    action, logprob, entropy, value = policy(obs, sampling_method="mean")

    assert action.shape == (1, 4, 2)
    assert logprob.shape == (1, 4)
    assert entropy.shape == (1, 4)
    assert value.shape == (1, 4)


def test_query_centric_and_anon_tokyo_policy_forward_shapes() -> None:
    batch = _batch(max_agents=4)
    env = ClosedLoopEnv(ClosedLoopEnvConfig(device="cpu", num_steps=2, history_steps=4))
    obs = env.reset(batch)
    target_mask = obs["controlled_mask"].bool() & obs["agent_mask"].bool()

    for policy in (
        QueryCentricModel(d_model=32, num_layers=1, num_heads=4, sparse_k=4),
        AnonTokyoModel(d_model=32, num_layers=1, num_heads=4, sparse_k=4),
    ):
        encoded = policy.encoder(obs)
        assert torch.equal(encoded["ego_mask"], target_mask)
        assert encoded["ego_feature"][~target_mask].abs().sum() == 0

        action, logprob, entropy, value = policy(obs, sampling_method="mean")

        assert action.shape == (1, 4, 2)
        assert logprob.shape == (1, 4)
        assert entropy.shape == (1, 4)
        assert value.shape == (1, 4)
        assert torch.all(action[..., 0] >= -5.0)
        assert torch.all(action[..., 0] <= 3.0)
        assert torch.all(action[..., 1] >= -1.0)
        assert torch.all(action[..., 1] <= 1.0)


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(2))
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        self.value = nn.Parameter(torch.zeros(()))

    def forward(self, obs: dict, action: torch.Tensor | None = None):
        shape = (*obs["controlled_mask"].shape, 2)
        mean = self.mean.view(1, 1, 2).expand(shape)
        std = self.log_std.exp().view(1, 1, 2).expand(shape)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value.expand(obs["controlled_mask"].shape)
        return action, logprob, entropy, value


def test_ppo_one_update_with_tiny_policy() -> None:
    batch = _batch(max_agents=4)
    env = ClosedLoopEnv(ClosedLoopEnvConfig(device="cpu", num_steps=2, history_steps=4))
    policy = TinyPolicy()
    before = [p.detach().clone() for p in policy.parameters()]
    trainer = PPOTrainer(env, policy, config=PPOConfig(num_steps=2, optimization_epochs=2, minibatch_size=1))
    metrics = trainer.train_one_update(batch)
    assert metrics["value_loss"] >= 0
    assert any(not torch.allclose(a, b) for a, b in zip(before, policy.parameters()))


def test_ppo_update_uses_map_tokens_without_raw_map_geometry() -> None:
    batch = _batch(max_agents=4)
    centers = batch["map_polylines_center"]
    B, M, _ = centers.shape
    token_features = torch.zeros(B, M, 11, dtype=centers.dtype)
    token_features[..., 0:2] = centers
    token_features[..., 2:4] = centers
    token_features[..., 4:6] = centers
    token_features[..., 6] = 1.0
    batch["map_token_features"] = token_features

    env = ClosedLoopEnv(ClosedLoopEnvConfig(device="cpu", num_steps=1, history_steps=4))
    policy = AnonTokyoModel(d_model=32, num_layers=1, num_heads=4, sparse_k=4)
    trainer = PPOTrainer(env, policy, config=PPOConfig(num_steps=1, optimization_epochs=1, minibatch_size=1))

    metrics = trainer.train_one_update(batch, sampling_method="mean")

    assert metrics["value_loss"] >= 0


class _ModuleWrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def test_ppo_buffer_keeps_raw_maps_for_wrapped_agent_centric_policy() -> None:
    batch = _batch(max_agents=4)
    batch["map_token_features"] = torch.zeros(
        *batch["map_polylines_center"].shape[:2],
        11,
        dtype=batch["map_polylines"].dtype,
    )
    env = ClosedLoopEnv(ClosedLoopEnvConfig(device="cpu", num_steps=1, history_steps=5))
    policy = AgentCentricModel(
        d_model=32,
        num_heads=4,
        max_context_agents=4,
        max_lanes=4,
        history_steps=5,
    )
    trainer = PPOTrainer(env, _ModuleWrapper(policy), config=PPOConfig(num_steps=1, optimization_epochs=1, minibatch_size=1))

    buffer, _, _, _ = trainer.collect_rollout(batch, sampling_method="mean")

    assert "map_polylines" in buffer.obs[0]
    assert "map_polylines_mask" in buffer.obs[0]
    assert "map_mask" in buffer.obs[0]
