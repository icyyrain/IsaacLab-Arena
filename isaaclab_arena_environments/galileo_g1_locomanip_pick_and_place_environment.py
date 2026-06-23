# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math
import warnings
from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


# The v0.2 brown-box-to-blue-bin workflow was SQA'd against this exact task description, Mimic
# datagen name, and (object, destination) pair. We preserve them verbatim for that specific pair so
# the pretrained gr00t model and existing Mimic datasets / policy checkpoints keyed on them keep
# resolving bit-identically. Any other pair -- including ``brown_box`` against a non-default
# destination -- uses the templated behavior from the base task / environment.
_LEGACY_PICK_UP_OBJECT_NAME = "brown_box"
_LEGACY_DESTINATION_NAME = "blue_sorting_bin"
_LEGACY_DATAGEN_NAME = "locomanip_pick_and_place_D0"
_LEGACY_BROWN_BOX_TO_BLUE_BIN_DESCRIPTION = (
    "Pick up the brown box from the shelf, and place it into the blue bin on the table located at the"
    " right of the shelf."
)

_CLEAN_OVERVIEW_PRIMS_TO_HIDE: tuple[str, ...] = (
    "galileo_locomanip/Structure/walls",
    "galileo_locomanip/Structure/doors",
)


def _is_legacy_pair(pick_up_object_name: str, destination_name: str) -> bool:
    return pick_up_object_name == _LEGACY_PICK_UP_OBJECT_NAME and destination_name == _LEGACY_DESTINATION_NAME


def _apply_legacy_datagen_name_override(
    env_cfg: Any,
    pick_up_object_name: str,
    destination_name: str,
) -> Any:
    """Rewrite the Mimic ``datagen_config.name`` to the legacy value for the v0.2 workflow.

    Only applies to Mimic configs (where ``datagen_config`` exists) and only to the exact
    ``(brown_box, blue_sorting_bin)`` pair that was SQA'd against this datagen key. All other
    pairs keep the templated name produced by ``G1PickAndPlaceMimicEnvCfg``.
    """
    if not _is_legacy_pair(pick_up_object_name, destination_name):
        return env_cfg

    datagen_config = getattr(env_cfg, "datagen_config", None)
    if datagen_config is None:
        return env_cfg

    print(
        f"Overriding Mimic datagen_config.name from {datagen_config.name} to the legacy {_LEGACY_DATAGEN_NAME}"
        "This preserves identical behavior with existing Mimic datasets"
        "Remove this in the future when checkpoints are retrained."
    )
    datagen_config.name = _LEGACY_DATAGEN_NAME
    return env_cfg


def _apply_overview_camera(env_cfg: Any) -> Any:
    """Frame the default multi-env Galileo grid from a fixed world-space camera."""
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.eye = (0.0, -55.0, 38.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    return env_cfg


def _use_clean_overview(args_cli: Any) -> bool:
    """Hide visual walls for either overview recording mode."""
    return bool(getattr(args_cli, "clean_overview", False) or getattr(args_cli, "mosaic_video", False))


def _apply_mosaic_camera(
    env_cfg: Any,
    width: int,
    height: int,
    camera_eye: tuple[float, float, float],
    camera_target: tuple[float, float, float],
) -> Any:
    """Add an optional batched third-person camera for visualization-only recording."""
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.sensors import TiledCameraCfg
    from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

    assert width > 0 and height > 0, "mosaic camera dimensions must be positive"
    eyes = torch.tensor([camera_eye], dtype=torch.float32)
    targets = torch.tensor([camera_target], dtype=torch.float32)
    rotation = tuple(quat_from_matrix(create_rotation_matrix_from_view(eyes, targets, up_axis="Z"))[0].tolist())
    env_cfg.scene.third_person_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis/ThirdPersonCamera",
        update_period=0.0,
        width=width,
        height=height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.0,
            clipping_range=(0.1, 100.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=tuple(camera_eye),
            rot=rotation,
            convention="opengl",
        ),
    )
    return env_cfg


def _hide_background_prims(env, env_ids, prim_relative_paths: tuple[str, ...]) -> None:
    """Hide selected background prims while preserving their physics state."""
    from pxr import UsdGeom

    del env_ids
    stage = env.sim.stage
    for env_prim_path in env.scene.env_prim_paths:
        for prim_relative_path in prim_relative_paths:
            prim_path = f"{env_prim_path}/{prim_relative_path}"
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                UsdGeom.Imageable(stage.OverridePrim(prim_path)).MakeInvisible()
            else:
                warnings.warn(
                    f"_hide_background_prims: prim not found at '{prim_path}'; walls may remain visible.",
                    stacklevel=1,
                )


@register_environment
class GalileoG1LocomanipPickAndPlaceEnvironment(ExampleEnvironmentBase):

    name: str = "galileo_g1_locomanip_pick_and_place"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        import isaaclab_arena.embodiments.g1.g1  # noqa: F401
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import G1PickAndPlaceMimicEnvCfg, PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose, PoseRange

        background = self.asset_registry.get_asset_by_name("galileo_locomanip")()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        destination = self.asset_registry.get_asset_by_name(args_cli.destination)()
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        XY_RANGE_M = 0.025
        pick_up_object.set_initial_pose(
            PoseRange(
                position_xyz_min=(0.5785 - XY_RANGE_M, 0.18 - XY_RANGE_M, 0.0707),
                position_xyz_max=(0.5785 + XY_RANGE_M, 0.18 + XY_RANGE_M, 0.0707),
                rpy_min=(math.pi, 0.0, math.pi),
                rpy_max=(math.pi, 0.0, math.pi),
            )
        )

        destination.set_initial_pose(
            Pose(
                position_xyz=(-0.2450, -1.6272, -0.2641),
                rotation_xyzw=(0.0, 0.0, 1.0, 0.0),
            )
        )
        embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.18, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        if (
            args_cli.embodiment == "g1_wbc_pink"
            and hasattr(args_cli, "mimic")
            and args_cli.mimic
            and not hasattr(args_cli, "auto")
        ):
            # Set navigation p-controller for locomanip use case
            action_cfg = embodiment.get_action_cfg()
            action_cfg.g1_action.use_p_control = True
            # Set nav subgoals (x,y,heading) and turning_in_place flag for G1 WBC Pink navigation p-controller
            action_cfg.g1_action.navigation_subgoals = [
                ([0.18, 0.18, 0.0], False),
                ([0.18, 0.18, -1.78], True),
                ([-0.0955, -1.1070, -1.78], False),
                ([-0.0955, -1.1070, -1.78], False),
            ]

        if args_cli.task_description is not None:
            task_description = args_cli.task_description
        elif _is_legacy_pair(args_cli.object, args_cli.destination):
            task_description = _LEGACY_BROWN_BOX_TO_BLUE_BIN_DESCRIPTION
        else:
            object_label = args_cli.object.replace("_", " ")
            destination_label = args_cli.destination.replace("_", " ")
            task_description = (
                f"Pick up the {object_label} from the shelf, and place it on the {destination_label} on the table"
                " located at the right of the shelf."
            )

        def env_cfg_callback(env_cfg):
            env_cfg = _apply_legacy_datagen_name_override(
                env_cfg,
                pick_up_object_name=pick_up_object.name,
                destination_name=destination.name,
            )
            if args_cli.overview_camera:
                env_cfg = _apply_overview_camera(env_cfg)
            if _use_clean_overview(args_cli):
                from isaaclab.managers import EventTermCfg

                env_cfg.events.hide_clean_overview_prims = EventTermCfg(
                    func=_hide_background_prims,
                    mode="prestartup",
                    params={"prim_relative_paths": _CLEAN_OVERVIEW_PRIMS_TO_HIDE},
                )
            if getattr(args_cli, "mosaic_video", False):
                env_cfg = _apply_mosaic_camera(
                    env_cfg,
                    width=args_cli.mosaic_camera_width,
                    height=args_cli.mosaic_camera_height,
                    camera_eye=tuple(args_cli.mosaic_camera_eye),
                    camera_target=tuple(args_cli.mosaic_camera_target),
                )
            return env_cfg

        def _build_g1_pick_and_place_mimic_cfg(arm_mode):
            return G1PickAndPlaceMimicEnvCfg(
                pick_up_object_name=pick_up_object.name,
                destination_location_name=destination.name,
                arm_mode=arm_mode,
            )

        scene = Scene(assets=[background, pick_up_object, destination])
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=PickAndPlaceTask(
                pick_up_object,
                destination,
                background,
                episode_length_s=30.0,
                task_description=task_description,
                force_threshold=0.5,
                velocity_threshold=0.1,
                mimic_env_cfg_factory=_build_g1_pick_and_place_mimic_cfg,
            ),
            teleop_device=teleop_device,
            env_cfg_callback=env_cfg_callback,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="brown_box")
        parser.add_argument("--destination", type=str, default="blue_sorting_bin")
        parser.add_argument("--embodiment", type=str, default="g1_wbc_pink")
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument(
            "--overview_camera",
            action="store_true",
            help="Use a world-space viewer camera that frames the multi-environment grid",
        )
        parser.add_argument(
            "--clean_overview",
            action="store_true",
            help="Hide the Galileo wall geometry in the overview without changing collisions",
        )
        parser.add_argument(
            "--task_description",
            type=str,
            default=None,
            help=(
                "Override the natural-language task description. Defaults to a template derived from --object "
                "and --destination."
            ),
        )
