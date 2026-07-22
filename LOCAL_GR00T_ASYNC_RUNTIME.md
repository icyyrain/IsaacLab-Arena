# Local GR00T Async Runtime

This file documents the known-good local setup for the Windows-native Isaac
Lab-Arena client plus Docker-hosted GR00T server used by the multi-robot async
demo on this workstation.

Use this path first. Do not rebuild the GR00T server image or create a new
sm120/PyTorch image unless the known-good server below fails with a CUDA error
such as `no kernel image is available for execution on the device`.

## Verified Runtime

- Repository: `C:\Projects\isaac`
- Isaac/Arena Python: `C:\Isaac\envs\arena-py311\python.exe`
- Isaac Lab experience: `C:\Projects\isaac\submodules\IsaacLab\apps\isaaclab.python.rendering.kit`
- Isaac-GR00T submodule: `C:\Projects\isaac\submodules\Isaac-GR00T`
- Real GR00T server image: `isaaclab_arena_gr00t-server:fast`
- Real GR00T server port: `127.0.0.1:5555`
- G1 loco-manipulation checkpoint on host:
  `C:\Projects\isaac\models\isaaclab_arena\locomanipulation_tutorial\checkpoint-20000`
- Same checkpoint inside the GR00T server container:
  `/models/isaaclab_arena/locomanipulation_tutorial/checkpoint-20000`

The latest successful real-model rerun was:

- Run directory: `C:\Projects\isaac\eval\async_vla_demo\n5_real_rerun_20260722`
- Video directory: `C:\Projects\isaac\eval\videos\async_vla_n5_real_rerun_20260722`
- `num_envs`: 5
- simulated time: 30.0 s
- `inference_wall_s.p95`: 0.207 s
- `deadline_miss_rate`: 0.388

That inference latency is in the real-model range. Replay/fake servers usually
show `inference_wall_s.p95` around `0.01-0.03 s` and must not be used for GPU
capacity measurements.

## Check Current State

Use these before starting a server:

```powershell
netstat -ano | Select-String ':5555'
docker ps -a --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" |
  Select-String -Pattern '5555|gr00t|replay'
Test-Path 'C:\Projects\isaac\models\isaaclab_arena\locomanipulation_tutorial\checkpoint-20000'
```

If another server owns port `5555`, stop it before switching between real and
replay modes:

```powershell
docker rm -f isaaclab_arena_gr00t_server 2>$null
docker rm -f isaaclab_arena_gr00t_replay_server 2>$null
```

## Start Real GR00T Policy Server

This is the server to use for throughput, GPU timeline, queueing, and deadline
miss experiments.

```powershell
docker rm -f isaaclab_arena_gr00t_server isaaclab_arena_gr00t_replay_server 2>$null

docker run -d --name isaaclab_arena_gr00t_server --gpus all `
  -p 5555:5555 `
  -v C:\Projects\isaac:/workspaces/isaac:ro `
  -v C:\Projects\isaac\models\isaaclab_arena\locomanipulation_tutorial\checkpoint-20000:/models/isaaclab_arena/locomanipulation_tutorial/checkpoint-20000:ro `
  -w /workspaces/isaac/submodules/Isaac-GR00T `
  isaaclab_arena_gr00t-server:fast `
  python gr00t/eval/run_gr00t_server.py `
    --modality-config-path /workspaces/isaac/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py `
    --model-path /models/isaaclab_arena/locomanipulation_tutorial/checkpoint-20000 `
    --embodiment-tag NEW_EMBODIMENT `
    --device cuda `
    --host 0.0.0.0 `
    --port 5555
```

Wait for the server to respond:

```powershell
$env:PYTHONPATH = 'C:\Projects\isaac\submodules\Isaac-GR00T;C:\Projects\isaac'

& C:\Isaac\envs\arena-py311\python.exe `
  C:\Projects\isaac\isaaclab_arena_gr00t\utils\wait_for_gr00t_server.py `
  --host 127.0.0.1 --port 5555 --timeout 300
```

Expected success line:

```text
GR00T server ready (ping={'status': 'ok', 'message': 'Server is running'})
```

Inspect logs if needed:

```powershell
docker logs -f isaaclab_arena_gr00t_server
```

## Start Replay Server

The replay server implements the same policy-server API but returns actions from
a small LeRobot dataset instead of running the neural model. Use it only for
protocol, traffic capture, or scheduler smoke tests.

```powershell
docker rm -f isaaclab_arena_gr00t_server isaaclab_arena_gr00t_replay_server 2>$null

docker run -d --name isaaclab_arena_gr00t_replay_server `
  -p 5555:5555 `
  -v C:\Projects\isaac:/workspaces/isaac:ro `
  -w /workspaces/isaac `
  isaaclab_arena_gr00t-server:fast `
  python tools/run_gr00t_replay_server.py `
    --dataset-path /workspaces/isaac/isaaclab_arena_gr00t/tests/test_data/test_g1_locomanip_lerobot `
    --modality-config-path /workspaces/isaac/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py `
    --execution-horizon 50 `
    --video-backend torchcodec `
    --host 0.0.0.0 `
    --port 5555
```

Check logs:

```powershell
docker logs -f isaaclab_arena_gr00t_replay_server
```

The replay log prints `Starting GR00T replay server`.

## Run N=5 Real-Policy Async Demo

The client command is the same for real and replay servers; the server listening
on `127.0.0.1:5555` determines which policy is used.

```powershell
$env:OMNI_KIT_ACCEPT_EULA = 'Y'
$env:ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE = '1'
$env:PYTHONPATH = 'C:\Projects\isaac\submodules\Isaac-GR00T;C:\Projects\isaac'

$runName = 'n5_real_rerun'
New-Item -ItemType Directory -Force `
  -Path "eval\async_vla_demo\$runName", "eval\videos\async_vla_$runName" | Out-Null

& C:\Isaac\envs\arena-py311\python.exe `
  C:\Projects\isaac\isaaclab_arena\evaluation\policy_runner.py `
  --headless `
  --experience C:\Projects\isaac\submodules\IsaacLab\apps\isaaclab.python.rendering.kit `
  --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy `
  --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/g1_locomanip_gr00t_closedloop_config.yaml `
  --remote_host 127.0.0.1 --remote_port 5555 `
  --scheduler async_edf `
  --async_time_aligned `
  --async_prefetch_lead_steps 10 `
  --async_step_dt 0.02 `
  --async_metrics_path "eval/async_vla_demo/$runName/metrics.json" `
  --async_trace_path "eval/async_vla_demo/$runName/trace.json" `
  --num_steps 1500 --num_envs 5 --env_spacing 20 --enable_cameras `
  --no-async_status_ui `
  --mosaic_video --video_dir "eval/videos/async_vla_$runName" `
  --mosaic_camera_mode planar `
  galileo_g1_locomanip_pick_and_place `
  --object brown_box --embodiment g1_wbc_joint 2>&1 |
  Tee-Object -FilePath "eval\async_vla_demo\$runName\policy_runner.log"
```

Render the status-overlay video and timeline:

```powershell
& C:\Isaac\envs\arena-py311\python.exe `
  C:\Projects\isaac\isaaclab_arena_gr00t\scripts\render_async_status_video.py `
  --input-video "C:\Projects\isaac\eval\videos\async_vla_$runName\third-person-mosaic-step-0.mp4" `
  --trace "C:\Projects\isaac\eval\async_vla_demo\$runName\trace.json" `
  --output-video "C:\Projects\isaac\eval\videos\async_vla_$runName\third-person-mosaic-status.mp4" `
  --timeline "C:\Projects\isaac\eval\async_vla_demo\$runName\timeline.png"
```

## Verify Real vs Replay

After a run, inspect:

```powershell
Get-Content "eval\async_vla_demo\$runName\metrics.json" -Raw
```

Use these rules of thumb:

- Real GR00T: `inference_wall_s.p95` is typically around `0.14-0.22 s` on this workstation.
- Replay/fake: `inference_wall_s.p95` is typically around `0.01-0.03 s`.
- Real capacity experiments should use the real GR00T server only.
- Replay is acceptable for protocol smoke tests, traffic capture, or scheduler tests.

## Expected Startup Time

The full chain has two slow phases:

- GR00T server model load: usually tens of seconds to a few minutes.
- Isaac/Kit/Galileo/camera/mosaic initialization: can be several minutes. If
  Galileo USD cache or remote dependencies are cold, this can look stuck for much
  longer before the control loop starts.

Once the control loop starts, progress is shown as `Steps: ...`. The verified
N=5 rerun completed the 1500-step control loop in about 3 minutes 21 seconds,
while the full command wall time was much longer due to initialization.

## Stop Servers

```powershell
docker rm -f isaaclab_arena_gr00t_server 2>$null
docker rm -f isaaclab_arena_gr00t_replay_server 2>$null
```
