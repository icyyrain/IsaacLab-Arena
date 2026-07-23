# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run Isaac Lab's RL-Games player with an explicit offscreen Factory camera."""

from __future__ import annotations

import gymnasium as gym
import runpy
import sys
from pathlib import Path
from types import MethodType

CAMERA_PATH = "/World/FactoryDemoCamera"
CAMERA_EYE = (1.3, 0.9, 0.75)
CAMERA_TARGET = (0.58, 0.0, 0.15)


def create_factory_demo_camera() -> None:
    """Create a USD camera aimed at the first Factory environment."""
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Gf, UsdGeom

    stage = get_current_stage()
    camera = UsdGeom.Camera.Define(stage, CAMERA_PATH)
    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.ClearXformOpOrder()
    camera_to_world = (
        Gf.Matrix4d().SetLookAt(Gf.Vec3d(*CAMERA_EYE), Gf.Vec3d(*CAMERA_TARGET), Gf.Vec3d(0.0, 0.0, 1.0)).GetInverse()
    )
    camera_xform.AddTransformOp().Set(camera_to_world)
    camera.GetFocalLengthAttr().Set(24.0)
    camera.GetHorizontalApertureAttr().Set(20.955)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
    print(f"[INFO] Factory demo camera: {CAMERA_PATH}", flush=True)


def install_camera_hook() -> None:
    """Create the demo camera immediately after the upstream environment."""
    original_make = gym.make

    def make_with_camera(*args, **kwargs):
        cfg = kwargs.get("cfg")
        assert cfg is not None, "Factory player did not pass an environment configuration"
        cfg.viewer.cam_prim_path = CAMERA_PATH
        cfg.viewer.resolution = (1280, 720)
        env = original_make(*args, **kwargs)
        create_factory_demo_camera()
        base_env = env.unwrapped
        original_render = base_env.render
        original_step = base_env.step
        step_count = 0
        was_successful = False

        def render_with_kit_update(self, *render_args, **render_kwargs):
            import omni.kit.app
            from isaaclab.app.settings_manager import get_settings_manager

            settings = get_settings_manager()
            self.sim.forward()
            settings.set_bool("/app/player/playSimulations", False)
            app = omni.kit.app.get_app()
            app.update()
            app.update()
            settings.set_bool("/app/player/playSimulations", True)
            return original_render(*render_args, **render_kwargs)

        base_env.render = MethodType(render_with_kit_update, base_env)

        def step_with_success_log(self, *step_args, **step_kwargs):
            nonlocal step_count, was_successful

            result = original_step(*step_args, **step_kwargs)
            step_count += 1
            successes = self._get_curr_successes(
                success_threshold=self.cfg_task.success_threshold,
                check_rot=False,
            )
            is_successful = bool(successes[0].item())
            if is_successful and not was_successful:
                print(f"[INFO] Factory peg insertion succeeded at demo step {step_count}", flush=True)
            was_successful = is_successful
            return result

        base_env.step = MethodType(step_with_success_log, base_env)
        for _ in range(3):
            base_env.render()
        return env

    gym.make = make_with_camera


def main() -> None:
    """Delegate PPO playback to Isaac Lab's unmodified RL-Games script."""
    repo_root = Path(__file__).resolve().parents[1]
    player = repo_root / "submodules" / "IsaacLab" / "scripts" / "reinforcement_learning" / "rl_games" / "play.py"
    assert player.is_file(), f"Isaac Lab RL-Games player not found: {player}"
    install_camera_hook()
    sys.argv[0] = str(player)
    runpy.run_path(str(player), run_name="__main__")


if __name__ == "__main__":
    main()
