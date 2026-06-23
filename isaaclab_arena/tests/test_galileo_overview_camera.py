# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the multi-environment Galileo overview camera callback."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from isaaclab_arena_environments.galileo_g1_locomanip_pick_and_place_environment import (
    _CLEAN_OVERVIEW_PRIMS_TO_HIDE,
    _apply_mosaic_camera,
    _apply_overview_camera,
    _hide_background_prims,
    _use_clean_overview,
)


def test_apply_overview_camera_uses_world_frame() -> None:
    cfg = SimpleNamespace(viewer=SimpleNamespace(origin_type="env", eye=(1.0, 2.0, 3.0), lookat=(0.0, 0.0, 0.0)))

    result = _apply_overview_camera(cfg)

    assert result is cfg
    assert cfg.viewer.origin_type == "world"
    assert cfg.viewer.eye == (0.0, -55.0, 38.0)
    assert cfg.viewer.lookat == (0.0, 0.0, 0.0)


def test_hide_background_prims_authors_visibility_without_deactivation(monkeypatch) -> None:
    hidden_prims = []

    class FakePrim:
        def IsValid(self):
            return True

        def SetActive(self, active):
            raise AssertionError(f"clean overview must not change active state: {active}")

    class FakeStage:
        def GetPrimAtPath(self, path):
            assert path == "/World/envs/env_0/galileo_locomanip/Structure/walls"
            return FakePrim()

        def OverridePrim(self, path):
            assert path == "/World/envs/env_0/galileo_locomanip/Structure/walls"
            return FakePrim()

    class FakeImageable:
        def __init__(self, prim):
            self.prim = prim

        def MakeInvisible(self):
            hidden_prims.append(self.prim)

    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(UsdGeom=SimpleNamespace(Imageable=FakeImageable)))
    env = SimpleNamespace(
        sim=SimpleNamespace(stage=FakeStage()),
        scene=SimpleNamespace(env_prim_paths=["/World/envs/env_0"]),
    )

    _hide_background_prims(
        env,
        env_ids=None,
        prim_relative_paths=("galileo_locomanip/Structure/walls",),
    )

    assert len(hidden_prims) == 1


def test_clean_overview_hides_walls_and_doors() -> None:
    assert _CLEAN_OVERVIEW_PRIMS_TO_HIDE == (
        "galileo_locomanip/Structure/walls",
        "galileo_locomanip/Structure/doors",
    )


def test_mosaic_video_uses_clean_overview_automatically() -> None:
    assert _use_clean_overview(SimpleNamespace(clean_overview=False, mosaic_video=True))
    assert _use_clean_overview(SimpleNamespace(clean_overview=True, mosaic_video=False))
    assert not _use_clean_overview(SimpleNamespace(clean_overview=False, mosaic_video=False))


def test_apply_mosaic_camera_adds_requested_tiled_rgb_sensor() -> None:
    cfg = SimpleNamespace(scene=SimpleNamespace())

    result = _apply_mosaic_camera(
        cfg,
        width=480,
        height=360,
        camera_eye=(-2.8, -2.8, 2.0),
        camera_target=(0.0, 0.0, 0.6),
        camera_mode="pelvis",
    )

    camera = cfg.scene.third_person_camera
    assert result is cfg
    assert type(camera).__name__ == "TiledCameraCfg"
    assert camera.prim_path == "{ENV_REGEX_NS}/Robot/pelvis/ThirdPersonCamera"
    assert camera.width == 480
    assert camera.height == 360
    assert camera.data_types == ["rgb"]
    assert camera.spawn.focal_length == 20.0
    assert camera.offset.pos == (-2.8, -2.8, 2.0)
    assert camera.offset.convention == "opengl"
    assert camera.offset.rot != (0.0, 0.0, 0.0, 1.0)


@pytest.mark.parametrize(
    ("mode", "expected_path"),
    [
        ("fixed", "{ENV_REGEX_NS}/ThirdPersonCamera"),
        ("planar", "{ENV_REGEX_NS}/ThirdPersonCamera"),
        ("pelvis", "{ENV_REGEX_NS}/Robot/pelvis/ThirdPersonCamera"),
    ],
)
def test_apply_mosaic_camera_selects_mode_specific_prim_path(mode: str, expected_path: str) -> None:
    cfg = SimpleNamespace(scene=SimpleNamespace())

    _apply_mosaic_camera(
        cfg,
        width=480,
        height=360,
        camera_eye=(-2.2, -2.2, 1.7),
        camera_target=(0.0, 0.0, 0.6),
        camera_mode=mode,
    )

    assert cfg.scene.third_person_camera.prim_path == expected_path
