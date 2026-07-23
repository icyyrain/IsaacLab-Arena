# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SpaceMouse action takeover compatible with Factory's normalized action."""

from __future__ import annotations

import numpy as np
from typing import Any, Protocol

from isaaclab_hil_serl.envs.factory_adapter import FACTORY_ACTION_DIM


class SpaceMouseDevice(Protocol):
    """Minimal Isaac Lab teleoperation device interface used by the provider."""

    def advance(self) -> Any:
        """Return the latest six-dimensional device command."""

    def reset(self) -> None:
        """Clear the current device command."""


class SpaceMouseInterventionProvider:
    """Use non-deadzone SpaceMouse motion as a human action override."""

    source = "spacemouse"

    def __init__(
        self,
        device: SpaceMouseDevice,
        *,
        deadzone: float = 0.05,
        translation_scale: float = 1.0,
        rotation_scale: float = 1.0,
    ):
        assert 0.0 <= deadzone < 1.0, f"deadzone must be in [0, 1): {deadzone}"
        assert translation_scale > 0.0, "translation_scale must be positive"
        assert rotation_scale > 0.0, "rotation_scale must be positive"
        self._device = device
        self.deadzone = float(deadzone)
        self._scale = np.asarray(
            [translation_scale] * 3 + [rotation_scale] * 3,
            dtype=np.float32,
        )
        self.active = False
        self.event_count = 0
        self.intervention_steps = 0

    @classmethod
    def create_isaaclab(
        cls,
        *,
        deadzone: float = 0.05,
        translation_scale: float = 1.0,
        rotation_scale: float = 1.0,
    ) -> SpaceMouseInterventionProvider:
        """Create the provider with Isaac Lab's native HID implementation."""
        from isaaclab.devices import Se3SpaceMouse, Se3SpaceMouseCfg

        device = Se3SpaceMouse(
            Se3SpaceMouseCfg(
                gripper_term=False,
                pos_sensitivity=1.0,
                rot_sensitivity=1.0,
                sim_device="cpu",
            )
        )
        return cls(
            device,
            deadzone=deadzone,
            translation_scale=translation_scale,
            rotation_scale=rotation_scale,
        )

    def reset(self) -> None:
        """Release takeover and clear stale device motion at episode reset."""
        self._device.reset()
        self.active = False

    def __call__(self, observation: np.ndarray, policy_action: np.ndarray) -> np.ndarray | None:
        del observation, policy_action
        command = self._to_numpy(self._device.advance())
        assert command.shape == (
            FACTORY_ACTION_DIM,
        ), f"SpaceMouse command must have shape {(FACTORY_ACTION_DIM,)}, received {command.shape}"
        assert np.isfinite(command).all(), "SpaceMouse command contains NaN or Inf"
        command = np.clip(command * self._scale, -1.0, 1.0)
        takeover = np.linalg.norm(command) > self.deadzone
        if takeover and not self.active:
            self.event_count += 1
        self.active = takeover
        if not takeover:
            return None
        self.intervention_steps += 1
        return command

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().to(device="cpu").numpy()
        return np.asarray(value, dtype=np.float32).copy()
