# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

import pytest

from isaaclab_hil_serl.data import (
    Transition,
    load_transition_dataset,
    save_transition_dataset,
    validate_episode_boundaries,
)


def _transition(
    episode_id: int,
    step_id: int,
    *,
    terminated: bool = False,
    truncated: bool = False,
    intervention: bool = False,
) -> Transition:
    policy_action = np.array([0.1, 0.2], dtype=np.float32)
    executed_action = np.array([-0.1, -0.2], dtype=np.float32) if intervention else policy_action
    return Transition(
        observation=np.array([episode_id, step_id, 0.0], dtype=np.float32),
        policy_action=policy_action,
        executed_action=executed_action,
        next_observation=np.array([episode_id, step_id + 1, 0.0], dtype=np.float32),
        reward=float(terminated),
        terminated=terminated,
        truncated=truncated,
        source="intervention" if intervention else "policy",
        intervention=intervention,
        episode_id=episode_id,
        step_id=step_id,
        success=terminated,
    )


def test_executed_action_is_used_for_serl_update():
    transition = _transition(0, 0, intervention=True)

    serl = transition.to_serl_dict()

    np.testing.assert_array_equal(serl["actions"], transition.executed_action)
    assert serl["masks"] == 1.0
    assert not serl["dones"]


def test_time_limit_bootstraps_but_terminal_success_does_not():
    truncated = _transition(0, 0, truncated=True).to_serl_dict()
    terminated = _transition(0, 0, terminated=True).to_serl_dict()

    assert truncated["masks"] == 1.0
    assert not truncated["dones"]
    assert terminated["masks"] == 0.0
    assert terminated["dones"]


def test_dataset_round_trip_and_episode_boundaries(tmp_path):
    original = [
        _transition(0, 0),
        _transition(0, 1, truncated=True),
        _transition(1, 0, intervention=True),
        _transition(1, 1, terminated=True),
    ]

    path = save_transition_dataset(tmp_path / "transitions.npz", original)
    restored = load_transition_dataset(path)

    assert len(restored) == len(original)
    for expected, actual in zip(original, restored, strict=True):
        assert expected.source == actual.source
        assert expected.episode_id == actual.episode_id
        assert expected.step_id == actual.step_id
        np.testing.assert_array_equal(expected.executed_action, actual.executed_action)


def test_episode_change_without_done_is_rejected():
    with pytest.raises(AssertionError, match="without termination"):
        validate_episode_boundaries([_transition(0, 0), _transition(1, 0)])


def test_non_intervention_cannot_change_action():
    with pytest.raises(AssertionError, match="changed the executed action"):
        Transition(
            observation=np.zeros(2, dtype=np.float32),
            policy_action=np.zeros(1, dtype=np.float32),
            executed_action=np.ones(1, dtype=np.float32),
            next_observation=np.zeros(2, dtype=np.float32),
            reward=0.0,
            terminated=False,
            truncated=False,
            source="policy",
            intervention=False,
            episode_id=0,
            step_id=0,
            success=False,
        )
