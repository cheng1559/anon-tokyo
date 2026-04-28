from __future__ import annotations

import torch

from anon_tokyo.visualize.serialize import serialize_simulation_batch


def _minimal_batch() -> dict[str, torch.Tensor | list[str]]:
    obj_trajs = torch.zeros(1, 2, 1, 10)
    obj_trajs[:, :, :, 3] = 4.5
    obj_trajs[:, :, :, 4] = 2.0
    return {
        "scenario_id": ["serialize_test"],
        "obj_types": torch.ones(1, 2, dtype=torch.long),
        "agent_mask": torch.ones(1, 2, dtype=torch.bool),
        "obj_trajs": obj_trajs,
        "obj_trajs_mask": torch.ones(1, 2, 1, dtype=torch.bool),
        "obj_trajs_future": torch.zeros(1, 2, 0, 4),
        "obj_trajs_future_mask": torch.zeros(1, 2, 0, dtype=torch.bool),
        "obj_positions": torch.zeros(1, 2, 2),
        "obj_headings": torch.zeros(1, 2),
        "map_polylines": torch.zeros(1, 0, 0, 7),
        "map_polylines_mask": torch.zeros(1, 0, 0, dtype=torch.bool),
        "controlled_mask": torch.tensor([[False, True]]),
        "goal_positions": torch.zeros(1, 2, 2),
        "current_time_index": torch.tensor([0]),
    }


def test_serialize_simulation_keeps_invalid_rollout_frames_aligned() -> None:
    rollout_positions = torch.tensor(
        [
            [
                [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]],
                [[1.0, 0.0], [11.0, 0.0], [21.0, 0.0], [31.0, 0.0]],
            ]
        ]
    )
    rollout_valid = torch.tensor([[[True, False, True, False], [True, True, False, False]]])

    payload = serialize_simulation_batch(
        _minimal_batch(),
        rollout_positions=rollout_positions,
        rollout_headings=torch.zeros(1, 2, 4),
        rollout_valid=rollout_valid,
    )

    tracks = {track["agent_id"]: track for track in payload["scenarios"][0]["rollout"]}
    assert payload["scenarios"][0]["agents"][0]["history"] == []
    assert payload["scenarios"][0]["sim_start_frame"] == 1
    assert tracks[0]["valid"] == [1, 0, 1, 0]
    assert tracks[0]["points"] == [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]
    assert tracks[1]["valid"] == [1, 1, 0, 0]


def test_serialize_simulation_preprocessed_map_polylines() -> None:
    batch = _minimal_batch()
    batch.update(
        {
            "preprocessed_map_polylines": torch.tensor(
                [
                    [
                        [[1.0, 2.0], [3.0, 4.0], [9.0, 9.0]],
                        [[5.0, 6.0], [7.0, 8.0], [9.0, 9.0]],
                    ]
                ]
            ),
            "preprocessed_map_mask": torch.tensor([[[True, True, False], [False, False, False]]]),
            "preprocessed_map_types": torch.tensor([[[1.0, 1.0, 1.0], [7.0, 7.0, 7.0]]]),
            "preprocessed_map_batch_idx": torch.tensor([0]),
            "preprocessed_map_agent_idx": torch.tensor([1]),
            "preprocessed_map_frame": torch.tensor([1]),
        }
    )

    payload = serialize_simulation_batch(batch)

    records = payload["scenarios"][0]["preprocessed_map"]
    assert records == [{"frame": 1, "agent_id": 1, "polylines": [{"type": 1, "points": [[1.0, 2.0], [3.0, 4.0]]}]}]
