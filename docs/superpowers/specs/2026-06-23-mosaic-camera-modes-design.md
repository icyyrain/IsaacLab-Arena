# Mosaic Camera Modes Design

## Goal

Make multi-robot mosaic recordings stable enough for motion and scheduling analysis while preserving the existing close pelvis-mounted view for comparison.

## Modes

`--mosaic_camera_mode` accepts three values and defaults to `planar`:

- `fixed`: one static camera per environment. Its eye and target are relative to the environment origin and never change after scene creation.
- `planar`: one environment-root camera follows the robot pelvis in world X/Y only. Eye height and orientation remain fixed.
- `pelvis`: the current camera remains parented to the pelvis and inherits its full translation and rotation.

The existing `--mosaic_camera_eye` and `--mosaic_camera_target` values remain the camera offsets. In planar mode, the same filtered pelvis X/Y translation is added to both values, so the look direction never changes.

## Planar Tracking

The recorder reads each robot's pelvis world position before stepping the environment. It tracks only X/Y and keeps Z unchanged. Tracking uses an exponential low-pass filter with a configurable time constant and a small deadband:

- `--mosaic_camera_follow_tau`: default `0.25` seconds.
- `--mosaic_camera_follow_deadband`: default `0.02` meters.

When displacement from the filtered position is within the deadband, the camera does not move. Otherwise the filter coefficient is derived from the environment control step duration. Reset reinitializes the filtered position immediately so camera state does not leak between episodes.

## Scene Integration

The Galileo environment selects the observer camera prim path from the mode:

- `pelvis`: `{ENV_REGEX_NS}/Robot/pelvis/ThirdPersonCamera`
- `fixed` and `planar`: `{ENV_REGEX_NS}/ThirdPersonCamera`

All modes keep the existing batched RGB sensor, close framing, automatic visual wall hiding, 3x2 tiling, and policy-camera inputs. Camera modes affect visualization only; physics, observations consumed by GR00T, scheduling, and metrics remain unchanged.

## Validation

Unit tests cover CLI defaults and choices, mode-specific prim paths, planar filter/deadband behavior, fixed-camera immobility, planar pose updates, and reset behavior. Runtime validation records short N=1 fixed and planar videos followed by an N=6 planar GR00T smoke test. The accepted planar video must keep orientation fixed, retain all six robots in frame, and preserve scheduler HUD and GPU timeline output.
