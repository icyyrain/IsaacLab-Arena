# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from isaaclab_hil_serl.teleop import SpaceMouseInterventionProvider


class _FakeSpaceMouse:
    def __init__(self, commands):
        self.commands = iter(commands)
        self.reset_count = 0

    def advance(self):
        return next(self.commands)

    def reset(self):
        self.reset_count += 1


def test_spacemouse_takeover_deadzone_release_and_scaling():
    device = _FakeSpaceMouse([
        np.zeros(6, dtype=np.float32),
        np.full(6, 0.01, dtype=np.float32),
        np.array([0.75, -0.75, 0.0, 0.6, -0.6, 0.0], dtype=np.float32),
        np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.zeros(6, dtype=np.float32),
        np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    ])
    provider = SpaceMouseInterventionProvider(
        device,
        deadzone=0.05,
        translation_scale=2.0,
        rotation_scale=0.5,
    )
    observation = np.zeros(43, dtype=np.float32)
    policy_action = np.zeros(6, dtype=np.float32)

    assert provider(observation, policy_action) is None
    assert provider(observation, policy_action) is None
    np.testing.assert_allclose(
        provider(observation, policy_action),
        np.array([1.0, -1.0, 0.0, 0.3, -0.3, 0.0], dtype=np.float32),
    )
    assert provider.event_count == 1
    assert provider.intervention_steps == 1
    assert provider.active

    np.testing.assert_allclose(
        provider(observation, policy_action),
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert provider.event_count == 1
    assert provider.intervention_steps == 2
    assert provider(observation, policy_action) is None
    assert not provider.active
    assert provider(observation, policy_action) is not None
    assert provider.event_count == 2

    provider.reset()
    assert device.reset_count == 1
    assert not provider.active


def test_spacemouse_rejects_invalid_command_shape():
    provider = SpaceMouseInterventionProvider(_FakeSpaceMouse([np.zeros(7)]))

    try:
        provider(np.zeros(43), np.zeros(6))
    except AssertionError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("Expected invalid SpaceMouse shape to fail")
