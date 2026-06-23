# GPU Timeline and Third-Person Mosaic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an exact control-time GPU service lane and a clear 2x3 third-person camera mosaic for the six-robot asynchronous GR00T demo.

**Architecture:** The scheduler will persist non-overlapping virtual GPU service intervals using measured inference wall durations mapped 1:1 into control time. A new optional batched `TiledCameraCfg` will capture one fixed third-person view per independent environment; a Gym wrapper will tile the batched sensor output into one MP4 without changing policy observations or task physics.

**Tech Stack:** Python 3.11, PyTorch, NumPy, OpenCV, MoviePy, Isaac Lab `TiledCameraCfg`, Gymnasium, pytest.

---

## File Map

- Modify `isaaclab_arena/policy/action_scheduling/async_deadline_action_scheduler.py`: retain exact virtual GPU service intervals.
- Modify `isaaclab_arena/tests/test_async_deadline_action_scheduler.py`: verify interval ordering, duration, and reset behavior.
- Modify `isaaclab_arena_gr00t/policy/async_metrics.py`: include GPU intervals in the trace schema.
- Modify `isaaclab_arena_gr00t/policy/gr00t_remote_closedloop_policy.py`: pass scheduler intervals into the final trace.
- Modify `isaaclab_arena_gr00t/utils/async_video_overlay.py`: render a categorical GPU lane and gray idle periods.
- Modify `isaaclab_arena_gr00t/tests/test_async_metrics.py`: verify trace serialization.
- Modify `isaaclab_arena_gr00t/tests/test_async_video_overlay.py`: verify the GPU lane and idle color.
- Create `isaaclab_arena/evaluation/mosaic_video.py`: tile batched camera tensors and record the mosaic MP4.
- Create `isaaclab_arena/tests/test_mosaic_video.py`: test tiling, labels, non-square batches, and sensor errors.
- Modify `isaaclab_arena/evaluation/policy_runner_cli.py`: add optional mosaic recording and camera framing arguments.
- Modify `isaaclab_arena/evaluation/policy_runner.py`: wrap the environment with the mosaic recorder.
- Modify `isaaclab_arena_environments/galileo_g1_locomanip_pick_and_place_environment.py`: add the observer `TiledCameraCfg` only when requested.
- Modify `isaaclab_arena/tests/test_galileo_overview_camera.py`: test observer camera configuration.
- Modify `LOCAL_RUNTIME_SETUP.md`: document capacity-safe and visualization-only commands.

### Task 1: Exact Virtual GPU Service Intervals

**Files:**
- Modify: `isaaclab_arena/policy/action_scheduling/async_deadline_action_scheduler.py`
- Test: `isaaclab_arena/tests/test_async_deadline_action_scheduler.py`

- [ ] **Step 1: Write the failing interval test**

Create two same-time requests and accept results with `0.30 s` and `0.20 s` inference durations. Require exact serialized intervals:

```python
assert scheduler.metrics()["gpu_service_intervals"] == [
    {
        "env_id": 0,
        "generation": 0,
        "sequence": 0,
        "submit_sim_time_s": 0.5,
        "start_sim_time_s": 0.5,
        "finish_sim_time_s": 0.8,
        "inference_wall_s": 0.3,
    },
    {
        "env_id": 1,
        "generation": 0,
        "sequence": 1,
        "submit_sim_time_s": 0.5,
        "start_sim_time_s": 0.8,
        "finish_sim_time_s": 1.0,
        "inference_wall_s": 0.2,
    },
]
```

- [ ] **Step 2: Run the scheduler test and verify RED**

Run:

```powershell
& C:\Isaac\envs\arena-py311\python.exe -m pytest `
  isaaclab_arena/tests/test_async_deadline_action_scheduler.py `
  -k gpu_service_intervals -v
```

Expected: failure because `gpu_service_intervals` is absent.

- [ ] **Step 3: Record accepted service intervals**

In `accept_result`, append a strict-JSON dictionary after calculating `virtual_start_s` and `virtual_finish_s`. Round public values to nine decimal places; do not record bootstrap or stale-generation results.

```python
self._gpu_service_intervals.append(
    {
        "env_id": env_id,
        "generation": request.generation,
        "sequence": request.sequence,
        "submit_sim_time_s": round(request.submit_sim_time_s, 9),
        "start_sim_time_s": round(virtual_start_s, 9),
        "finish_sim_time_s": round(virtual_finish_s, 9),
        "inference_wall_s": round(inference_wall_s, 9),
    }
)
```

Expose a copy from `metrics()` so callers cannot mutate scheduler state.

- [ ] **Step 4: Run scheduler tests and verify GREEN**

Expected: all scheduler tests pass.

### Task 2: GPU Lane in Trace and Timeline

**Files:**
- Modify: `isaaclab_arena_gr00t/policy/async_metrics.py`
- Modify: `isaaclab_arena_gr00t/policy/gr00t_remote_closedloop_policy.py`
- Modify: `isaaclab_arena_gr00t/utils/async_video_overlay.py`
- Test: `isaaclab_arena_gr00t/tests/test_async_metrics.py`
- Test: `isaaclab_arena_gr00t/tests/test_async_video_overlay.py`

- [ ] **Step 1: Write failing trace and rendering tests**

Require `build_async_trace` to retain `gpu_service_intervals`. Build a two-second synthetic trace with R0 active from `0.2–0.5 s`, R1 active from `0.5–0.8 s`, and idle elsewhere. Assert the GPU lane contains both robot colors and exact gray idle pixels.

```python
trace = build_async_trace(
    num_envs=2,
    step_dt=0.02,
    deadline_window_s=0.5,
    frames=frames,
    gpu_service_intervals=intervals,
)
assert trace["gpu_service_intervals"] == intervals
```

- [ ] **Step 2: Run tests and verify RED**

Run the two test files with `-k "trace or timeline"`. Expected: signature or pixel assertion failures.

- [ ] **Step 3: Extend the trace schema**

Add `gpu_service_intervals` to `build_async_trace` and pass `scheduler_metrics["gpu_service_intervals"]` from policy close. Keep `active_inference_env_id` in frame records for live diagnostics, but document that the rendered GPU lane uses exact virtual intervals.

- [ ] **Step 4: Render one categorical GPU lane**

Add a six-color robot palette distinct from status colors. Draw a gray lane first, then clip each interval to `[0, trace_duration]` and fill its x-range. Add compact `R0`, `R1`, ... labels inside segments wide enough for text and include a palette legend.

```python
cv2.rectangle(canvas, (plot_left, gpu_y0), (plot_right, gpu_y1), GPU_IDLE_BGR, -1)
for interval in trace["gpu_service_intervals"]:
    x0 = time_to_x(interval["start_sim_time_s"])
    x1 = time_to_x(interval["finish_sim_time_s"])
    cv2.rectangle(canvas, (x0, gpu_y0), (max(x0 + 1, x1), gpu_y1), robot_color(interval["env_id"]), -1)
```

- [ ] **Step 5: Run trace and overlay tests**

Expected: strict JSON round-trip and timeline pixel tests pass.

### Task 3: Batched Third-Person Mosaic Recorder

**Files:**
- Create: `isaaclab_arena/evaluation/mosaic_video.py`
- Create: `isaaclab_arena/tests/test_mosaic_video.py`

- [ ] **Step 1: Write failing pure tiling tests**

Test a six-frame `[6, 4, 5, 3]` batch with three columns, a five-frame batch requiring one black padding tile, float-to-uint8 conversion, and visible R0–R5 labels.

```python
mosaic = tile_camera_batch(frames, columns=3, draw_labels=True)
assert mosaic.shape == (8, 15, 3)
np.testing.assert_array_equal(mosaic[0:4, 0:5], expected_env_0_with_label)
```

- [ ] **Step 2: Run tests and verify RED**

Expected: import failure for `isaaclab_arena.evaluation.mosaic_video`.

- [ ] **Step 3: Implement pure tiling**

Implement `tile_camera_batch(frames, columns, draw_labels)` using NumPy and OpenCV. Accept Torch or NumPy `[N,H,W,C]`, convert to uint8, preserve env order row-major, and pad unused tiles black.

- [ ] **Step 4: Implement `TiledCameraMosaicRecorder`**

The Gym wrapper receives `sensor_name`, `camera_eye`, `camera_target`, `columns`, and standard video arguments. In `__init__`, resolve `env.unwrapped.scene[sensor_name]`, position every camera relative to `scene.env_origins` with `set_world_poses_from_view`, and fail with a clear assertion if the sensor is missing. In `step`, read `sensor.data.output["rgb"]`, tile it, buffer frames, and encode one `third-person-mosaic-step-0.mp4` on completion or close.

- [ ] **Step 5: Run mosaic unit tests**

Expected: all pure tiling and fake-sensor wrapper tests pass without launching Isaac Sim.

### Task 4: Optional Observer Camera and Runner Wiring

**Files:**
- Modify: `isaaclab_arena/evaluation/policy_runner_cli.py`
- Modify: `isaaclab_arena/evaluation/policy_runner.py`
- Modify: `isaaclab_arena_environments/galileo_g1_locomanip_pick_and_place_environment.py`
- Modify: `isaaclab_arena/tests/test_galileo_overview_camera.py`

- [ ] **Step 1: Write failing CLI and camera-config tests**

Assert defaults: mosaic disabled, `480x360`, three columns, eye `(3.6, -4.2, 2.8)`, target `(0.0, -0.7, 1.0)`. With mosaic enabled, require a scene field named `third_person_camera` whose prim path is `{ENV_REGEX_NS}/ThirdPersonCamera` and whose type is tiled.

- [ ] **Step 2: Run tests and verify RED**

Expected: missing CLI attributes and missing scene sensor.

- [ ] **Step 3: Add CLI arguments**

Add:

```text
--mosaic_video / --mosaic-video
--mosaic_camera_width 480
--mosaic_camera_height 360
--mosaic_columns 3
--mosaic_camera_eye 3.6 -4.2 2.8
--mosaic_camera_target 0.0 -0.7 1.0
```

Validate positive dimensions and columns in the recorder.

- [ ] **Step 4: Add the Galileo observer camera only when requested**

In the environment callback, assign a `TiledCameraCfg` to `env_cfg.scene.third_person_camera` with RGB output, requested resolution, `focal_length=20.0`, and `clipping_range=(0.1, 100.0)`. The recorder sets final world poses after scene construction.

- [ ] **Step 5: Wire the recorder in `policy_runner.py`**

Treat mosaic video like other recording modes when calculating `video_length` and creating the output directory. Wrap after `RecordVideo` and `CameraObsVideoRecorder`, passing the local eye/target and output settings. Do not change render mode solely for mosaic because the sensor supplies its own RGB frames.

- [ ] **Step 6: Run focused runner and environment tests**

Expected: CLI parsing and camera config tests pass; existing video modes remain unchanged.

### Task 5: Documentation and Runtime Verification

**Files:**
- Modify: `LOCAL_RUNTIME_SETUP.md`
- Runtime outputs: `eval/async_vla_demo/n6_mosaic/`, `eval/videos/async_vla_n6_mosaic/`

- [ ] **Step 1: Run all focused unit tests**

Run scheduler, metrics, overlay, mosaic, Galileo camera, GR00T policy, worker, and summary tests. Expected: zero failures.

- [ ] **Step 2: Run a 120-step N=6 mosaic smoke test**

Use the existing GR00T server, `--mosaic_video`, no viewport `--video`, and trace output. Verify the MP4 is `1440x720`, nonblack, 50 FPS, and shows all six labeled robots.

- [ ] **Step 3: Tune only camera eye/target if framing is poor**

Use the smoke-test middle frame. The accepted frame must show the whole G1 body, source shelf, and destination table/bin in every tile without clipping. Keep camera resolution and policy input unchanged.

- [ ] **Step 4: Generate the status video and timeline**

Run `render_async_status_video.py` on `third-person-mosaic-step-0.mp4`. Verify the output preserves mosaic resolution and the timeline contains a GPU lane with colored service segments plus gray idle.

- [ ] **Step 5: Run the full 1500-step N=6 visualization demo**

Write metrics and trace under `eval/async_vla_demo/n6_mosaic/`. This visualization run is not the capacity measurement because the extra observer camera changes rendering load.

- [ ] **Step 6: Update the runbook**

Document the exact smoke/full commands, the visualization overhead warning, output paths, and the distinction between the control-time GPU lane and sampled live worker state.

- [ ] **Step 7: Final verification**

Run focused pytest, compileall, pre-commit on changed files, `git diff --check`, video metadata assertions, nonblack canvas checks, trace interval non-overlap assertions, and a visual inspection of the mosaic midpoint and first deadline miss frame. Do not commit until the user requests it.
