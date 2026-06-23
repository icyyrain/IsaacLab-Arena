# Mosaic Camera Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fixed, planar-smoothed, and pelvis-mounted mosaic camera modes, with planar tracking as the stable default.

**Architecture:** The Galileo environment owns camera prim placement while the recorder owns runtime planar tracking. A pure tensor helper performs deadband and exponential smoothing so camera behavior is testable without Isaac Sim; fixed and pelvis modes do not execute runtime pose updates.

**Tech Stack:** Python, PyTorch, Gymnasium wrappers, Isaac Lab `TiledCameraCfg`, OpenCV, pytest.

---

### Task 1: CLI and mode-specific camera configuration

**Files:**
- Modify: `isaaclab_arena/evaluation/policy_runner_cli.py`
- Modify: `isaaclab_arena/tests/test_policy_runner_mosaic_cli.py`
- Modify: `isaaclab_arena_environments/galileo_g1_locomanip_pick_and_place_environment.py`
- Modify: `isaaclab_arena/tests/test_galileo_overview_camera.py`

- [ ] **Step 1: Write failing CLI tests**

Extend `test_policy_runner_mosaic_cli_defaults` to assert:

```python
assert args.mosaic_camera_mode == "planar"
assert args.mosaic_camera_follow_tau == 0.25
assert args.mosaic_camera_follow_deadband == 0.02
```

Add a parser rejection test:

```python
def test_policy_runner_rejects_unknown_mosaic_camera_mode() -> None:
    parser = argparse.ArgumentParser()
    add_policy_runner_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--policy_type", "example.Policy", "--mosaic_camera_mode", "orbit"])
```

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```powershell
& C:\Isaac\envs\arena-py311\python.exe -m pytest isaaclab_arena/tests/test_policy_runner_mosaic_cli.py -q
```

Expected: missing camera-mode attributes and no parser rejection.

- [ ] **Step 3: Add CLI arguments**

Add to `add_policy_runner_arguments`:

```python
parser.add_argument(
    "--mosaic_camera_mode",
    choices=("fixed", "planar", "pelvis"),
    default="planar",
)
parser.add_argument("--mosaic_camera_follow_tau", type=float, default=0.25)
parser.add_argument("--mosaic_camera_follow_deadband", type=float, default=0.02)
```

- [ ] **Step 4: Write failing camera-prim tests**

Parameterize the Galileo camera test:

```python
@pytest.mark.parametrize(
    ("mode", "expected_path"),
    [
        ("fixed", "{ENV_REGEX_NS}/ThirdPersonCamera"),
        ("planar", "{ENV_REGEX_NS}/ThirdPersonCamera"),
        ("pelvis", "{ENV_REGEX_NS}/Robot/pelvis/ThirdPersonCamera"),
    ],
)
def test_apply_mosaic_camera_selects_mode_specific_prim_path(mode, expected_path):
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
```

- [ ] **Step 5: Run camera config test and verify RED**

Run:

```powershell
& C:\Isaac\envs\arena-py311\python.exe -m pytest isaaclab_arena/tests/test_galileo_overview_camera.py -q
```

Expected: `_apply_mosaic_camera` does not accept `camera_mode`.

- [ ] **Step 6: Implement mode-specific camera prims**

Add a `camera_mode: str` parameter, validate it, and select:

```python
assert camera_mode in ("fixed", "planar", "pelvis"), f"unsupported mosaic camera mode: {camera_mode}"
prim_path = (
    "{ENV_REGEX_NS}/Robot/pelvis/ThirdPersonCamera"
    if camera_mode == "pelvis"
    else "{ENV_REGEX_NS}/ThirdPersonCamera"
)
```

Pass `args_cli.mosaic_camera_mode` from the environment callback.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run both test files. Expected: all tests pass.

### Task 2: Pure planar tracking math

**Files:**
- Modify: `isaaclab_arena/evaluation/mosaic_video.py`
- Modify: `isaaclab_arena/tests/test_mosaic_video.py`

- [ ] **Step 1: Write deadband and smoothing tests**

Import `_smooth_planar_xy` and add:

```python
def test_smooth_planar_xy_holds_inside_deadband() -> None:
    current = torch.tensor([[1.0, 2.0]])
    desired = torch.tensor([[1.01, 2.0]])
    actual = _smooth_planar_xy(current, desired, dt=0.02, tau=0.25, deadband=0.02)
    torch.testing.assert_close(actual, current)


def test_smooth_planar_xy_uses_exponential_filter_outside_deadband() -> None:
    current = torch.zeros((1, 2))
    desired = torch.tensor([[1.0, 0.0]])
    actual = _smooth_planar_xy(current, desired, dt=0.25, tau=0.25, deadband=0.02)
    torch.testing.assert_close(actual, torch.tensor([[1.0 - math.exp(-1.0), 0.0]]))
```

- [ ] **Step 2: Run tests and verify RED**

Expected: `_smooth_planar_xy` cannot be imported.

- [ ] **Step 3: Implement the pure helper**

Add:

```python
def _smooth_planar_xy(
    current_xy: torch.Tensor,
    desired_xy: torch.Tensor,
    dt: float,
    tau: float,
    deadband: float,
) -> torch.Tensor:
    assert current_xy.shape == desired_xy.shape and current_xy.shape[-1] == 2
    assert dt > 0.0 and tau > 0.0 and deadband >= 0.0
    delta = desired_xy - current_xy
    should_move = torch.linalg.vector_norm(delta, dim=-1, keepdim=True) > deadband
    alpha = 1.0 - math.exp(-dt / tau)
    return torch.where(should_move, current_xy + alpha * delta, current_xy)
```

- [ ] **Step 4: Run mosaic unit tests and verify GREEN**

Expected: helper and existing tiling/encoding tests pass.

### Task 3: Wire fixed and planar recorder behavior

**Files:**
- Modify: `isaaclab_arena/evaluation/mosaic_video.py`
- Modify: `isaaclab_arena/evaluation/policy_runner.py`
- Modify: `isaaclab_arena/tests/test_mosaic_video.py`

- [ ] **Step 1: Extend test fakes**

Give `_FakeSensor` pose capture fields, `_FakeScene` environment origins and a fake robot with a pelvis body state, and `_FakeEnv.step` a valid Gym return. The robot state must expose:

```python
body_names = ["pelvis"]
data.body_link_state_w = torch.tensor(
    [
        [[0.0, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0]],
        [[10.0, 20.2, 0.8, 0.0, 0.0, 0.0, 1.0]],
    ]
)
```

- [ ] **Step 2: Write failing fixed and planar recorder tests**

Fixed mode must never call `set_world_poses_from_view`. Planar construction must place each camera from pelvis X/Y plus eye/target offsets while preserving fixed Z. After changing pelvis X/Y and stepping, assert both eye and target receive the same smoothed translation. After reset, assert the filter snaps to the reset pelvis position rather than retaining the previous episode's value.

- [ ] **Step 3: Run recorder tests and verify RED**

Expected: constructor rejects new mode/tracking arguments and planar poses are absent.

- [ ] **Step 4: Implement recorder mode state**

Extend `TiledCameraMosaicRecorder.__init__` with:

```python
camera_mode: str = "pelvis",
camera_eye: tuple[float, float, float] | None = None,
camera_target: tuple[float, float, float] | None = None,
follow_tau: float = 0.25,
follow_deadband: float = 0.02,
```

For planar mode, resolve `scene["robot"]`, find the `pelvis` body index, cache `scene.env_origins`, eye/target tensors, `env.unwrapped.step_dt`, and initialize filtered world X/Y from `body_link_state_w`.

- [ ] **Step 5: Implement planar pose updates**

Add private methods that:

1. Read pelvis world X/Y.
2. Reset or smooth filtered X/Y.
3. Build eyes and targets by adding identical filtered X/Y translations to the configured offsets.
4. Keep eye and target Z relative to each environment origin.
5. Call `sensor.set_world_poses_from_view(eyes, targets)`.

Call the planar update before `env.step`. Override `reset` to call the wrapped reset and snap planar tracking to the new pelvis pose. Fixed and pelvis modes skip all pose updates.

- [ ] **Step 6: Pass runner arguments**

In `policy_runner.py`, pass:

```python
camera_mode=args_cli.mosaic_camera_mode,
camera_eye=tuple(args_cli.mosaic_camera_eye),
camera_target=tuple(args_cli.mosaic_camera_target),
follow_tau=args_cli.mosaic_camera_follow_tau,
follow_deadband=args_cli.mosaic_camera_follow_deadband,
```

- [ ] **Step 7: Run focused unit tests and verify GREEN**

Run mosaic, CLI, Galileo camera, scheduler, metrics, policy, and overlay tests. Expected: zero failures.

### Task 4: Documentation and runtime validation

**Files:**
- Modify: `LOCAL_RUNTIME_SETUP.md`
- Runtime outputs: `eval/videos/mosaic_camera_fixed_smoke/`
- Runtime outputs: `eval/videos/mosaic_camera_planar_smoke/`
- Runtime outputs: `eval/videos/async_vla_n6_mosaic_planar/`
- Runtime outputs: `eval/async_vla_demo/n6_mosaic_planar/`

- [ ] **Step 1: Document camera modes**

Document that planar is the default, fixed is completely stationary, and pelvis preserves the previous full-follow view. Include the two tracking parameters and a fixed-mode command override.

- [ ] **Step 2: Run N=1 fixed and planar camera smoke tests**

Use zero action for scene/framing validation. Verify both videos are nonblack, fixed mode has invariant camera pose, and planar mode uses an environment-root camera.

- [ ] **Step 3: Run N=6 planar GR00T validation**

Run 1500 steps with the existing async EDF configuration and `--mosaic_camera_mode planar`. Generate the status video and GPU timeline from the trace.

- [ ] **Step 4: Inspect beginning, first miss, midpoint, and final frames**

Verify all six workcells remain visible, robot bodies are not clipped, camera roll/yaw does not inherit gait motion, and the final frame still follows robots that have moved from the shelf.

- [ ] **Step 5: Final automated verification**

Run focused pytest, `compileall`, `git diff --check`, exact raw/status frame counts, nonblack tile checks, and GPU service interval non-overlap checks.

- [ ] **Step 6: Commit implementation**

Stage only camera-mode code, tests, docs, and this plan. Exclude `eval/`, models, datasets, submodule changes, and research reports. Commit with DCO sign-off.
