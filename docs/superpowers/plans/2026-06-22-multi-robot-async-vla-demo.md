# Multi-Robot Async VLA Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run multiple G1 robots in one Isaac Stage against one serial GR00T GPU worker, visibly demonstrate independent motion and deadline-miss pauses, and measure the largest robot count that meets control-time deadlines.

**Architecture:** Keep the simulation vectorized as Arena `num_envs` instances in one Stage. Add a per-environment asynchronous chunk scheduler driven by simulation/control time, while a dedicated thread owns the ZeroMQ client and serializes B=1 requests with EDF ordering. Actual measured inference durations feed a virtual GPU timeline, so slow Isaac Sim wall time cannot create artificial deadline slack.

**Tech Stack:** Python 3.11, PyTorch, ZeroMQ GR00T `PolicyClient`, Isaac Lab/Arena vectorized environments, `omni.ui`, pytest.

---

## File Map

- Create `isaaclab_arena/policy/action_scheduling/async_deadline_action_scheduler.py`: pure per-env chunk/deadline state machine with no Isaac Sim or GR00T dependency.
- Modify `isaaclab_arena/policy/action_scheduling/__init__.py`: export the scheduler and its request/result/status types.
- Create `isaaclab_arena/tests/test_async_deadline_action_scheduler.py`: deterministic control-time tests.
- Create `isaaclab_arena_gr00t/policy/gr00t_async_worker.py`: EDF priority queue and the one thread that owns the GR00T ZMQ client.
- Create `isaaclab_arena_gr00t/tests/test_gr00t_async_worker.py`: worker ordering, timing, shutdown, and stale-generation tests.
- Modify `isaaclab_arena_gr00t/policy/gr00t_remote_closedloop_policy.py`: CLI selection, observation splitting, bootstrap, async step integration, metrics, and cleanup.
- Modify `isaaclab_arena_gr00t/tests/test_gr00t_remote_closedloop_policy.py`: async policy integration tests using fake clients.
- Create `isaaclab_arena_gr00t/policy/async_status_ui.py`: optional Kit status window, imported lazily.
- Create `isaaclab_arena_gr00t/policy/async_metrics.py`: JSON summary and per-env capacity metrics.
- Create `isaaclab_arena_gr00t/tests/test_async_metrics.py`: aggregation and zero-miss capacity checks.
- Create `isaaclab_arena_gr00t/scripts/summarize_async_capacity.py`: summarize multiple JSON run files.
- Modify `LOCAL_RUNTIME_SETUP.md`: exact Windows commands for demo and capacity sweep.

### Task 1: Control-Time Async Scheduler

**Files:**
- Create: `isaaclab_arena/policy/action_scheduling/async_deadline_action_scheduler.py`
- Modify: `isaaclab_arena/policy/action_scheduling/__init__.py`
- Test: `isaaclab_arena/tests/test_async_deadline_action_scheduler.py`

- [ ] **Step 1: Write failing scheduler tests**

Cover bootstrap, prefetch at `chunk_length - lead_steps`, EDF metadata, virtual completion gating, hold-pose on miss, late-result resume, per-env reset generations, and metrics. Use `step_dt=0.02`, `chunk_length=50`, and `lead_steps=25` so the deadline window is exactly `0.5` simulated seconds.

```python
def test_holds_at_boundary_until_virtual_completion():
    scheduler = AsyncDeadlineActionScheduler(
        num_envs=1, action_chunk_length=50, action_horizon=50,
        action_dim=2, step_dt=0.02, prefetch_lead_steps=25,
        device="cpu",
    )
    scheduler.bootstrap(torch.ones(1, 50, 2))
    for _ in range(25):
        scheduler.step(torch.zeros(1, 2))
    request = scheduler.take_pending_requests()[0]
    scheduler.accept_result(request, torch.full((50, 2), 2.0), inference_wall_s=0.6)
    for _ in range(25):
        scheduler.step(torch.zeros(1, 2))
    assert scheduler.statuses[0] == AsyncEnvStatus.DEADLINE_MISS
    torch.testing.assert_close(scheduler.step(torch.full((1, 2), 9.0))[0], torch.tensor([9.0, 9.0]))
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
& C:\Isaac\envs\arena-py311\python.exe -m pytest isaaclab_arena/tests/test_async_deadline_action_scheduler.py -v
```

Expected: collection/import failure because `AsyncDeadlineActionScheduler` does not exist.

- [ ] **Step 3: Implement the state machine**

Define these public methods with the behavior specified above: `bootstrap(chunks)`,
`step(hold_action)`, `take_pending_requests()`, `mark_inference_started(env_id, generation)`,
`accept_result(request, chunk, inference_wall_s, network_delay_s=0.0)`, `reset(env_ids=None)`,
and `metrics()`.

```python
class AsyncEnvStatus(StrEnum):
    BOOTSTRAP = "bootstrap"
    EXECUTING = "executing"
    QUEUED = "queued"
    INFERENCE = "inference"
    GATED = "gated"
    DEADLINE_MISS = "deadline_miss"

@dataclass(frozen=True)
class AsyncChunkRequest:
    env_id: int
    generation: int
    submit_sim_time_s: float
    deadline_sim_time_s: float
    sequence: int
```

The scheduler advances only by `step_dt` per `step()` call. Compute virtual service completion as:

```python
virtual_start = max(request.submit_sim_time_s, self._virtual_gpu_available_s)
virtual_finish = virtual_start + inference_wall_s + network_delay_s
self._virtual_gpu_available_s = virtual_finish
```

Never use Isaac Sim wall-step duration in deadline calculations.

- [ ] **Step 4: Run scheduler tests**

Expected: all scheduler tests pass.

- [ ] **Step 5: Review diff checkpoint**

Inspect only the scheduler, export, and test files. Do not commit unless the user requests a commit.

### Task 2: Single-Owner EDF GR00T Worker

**Files:**
- Create: `isaaclab_arena_gr00t/policy/gr00t_async_worker.py`
- Test: `isaaclab_arena_gr00t/tests/test_gr00t_async_worker.py`

- [ ] **Step 1: Write failing worker tests**

Use a fake client factory that records the thread ID of construction and every `get_action` call. Assert one owner thread, earliest-deadline-first ordering, per-request queue/inference timing, exception propagation, and bounded shutdown.

```python
worker.submit(WorkerRequest(deadline_sim_time_s=1.0, sequence=1, payload="late"))
worker.submit(WorkerRequest(deadline_sim_time_s=0.5, sequence=2, payload="early"))
results = worker.wait_for_results(2, timeout_s=2.0)
assert [result.request.payload for result in results] == ["early", "late"]
assert len({client.owner_thread_id, *client.call_thread_ids}) == 1
```

- [ ] **Step 2: Run tests and verify failure**

Expected: import failure for `gr00t_async_worker`.

- [ ] **Step 3: Implement the worker**

Use `queue.PriorityQueue`, a monotonic sequence tie-breaker, `threading.Event`, and a result `SimpleQueue`. Construct and ping `Gr00tPolicyClient` inside the worker thread. Expose `submit`, `poll`, `wait_for_results`, `active_env_id`, and idempotent `close`. The worker calls only `client.get_action(payload)`; action translation stays on the main thread.

- [ ] **Step 4: Run worker tests**

Expected: all worker tests pass without Isaac Sim or a real server.

- [ ] **Step 5: Review diff checkpoint**

Confirm no ZeroMQ socket created on the main thread and no changes under `submodules/`.

### Task 3: GR00T Policy Async Integration

**Files:**
- Modify: `isaaclab_arena_gr00t/policy/gr00t_remote_closedloop_policy.py`
- Modify: `isaaclab_arena_gr00t/tests/test_gr00t_remote_closedloop_policy.py`

- [ ] **Step 1: Add failing integration tests**

Test CLI/config defaults and an async policy lifecycle:

```python
args = Gr00tRemoteClosedloopPolicyArgs(
    policy_config_yaml_path=policy_config_yaml,
    policy_device="cpu",
    num_envs=NUM_ENVS,
    remote_host="unused",
    remote_port=0,
    remote_api_token=None,
    async_prefetch_lead_steps=25,
    async_step_dt=0.02,
    async_network_delay_s=0.0, async_status_ui=False,
)
policy = Gr00tRemoteClosedloopPolicy(args, scheduler_mode="async_edf", client_factory=fake_factory)
first = policy.get_action(env=None, observation=synthetic_observation)
assert first.shape == (NUM_ENVS, EXPECTED_ACTION_DIM)
assert policy.async_metrics()["num_envs"] == NUM_ENVS
policy.close()
assert fake_factory.worker_stopped
```

Also verify that nested policy observations split into B=1 payloads and that reset generations discard stale results.

- [ ] **Step 2: Run tests and verify failure**

Expected: constructor/CLI failures for missing async arguments and scheduler mode.

- [ ] **Step 3: Add async CLI and configuration**

Extend `--scheduler` with `async_edf` and add:

```text
--async_prefetch_lead_steps 25
--async_step_dt 0.02
--async_network_delay_s 0.0
--async_metrics_path <optional JSON>
--async_status_ui / --no-async_status_ui
```

Validate `1 <= lead_steps <= action_chunk_length` and `step_dt > 0`.

- [ ] **Step 4: Integrate bootstrap and per-step flow**

On first observation, build the full policy observation once, split it into B=1 payloads, bootstrap all env chunks through the serial worker, and start simulation-time accounting only after all chunks are loaded. On each later step: poll results, translate each B=1 action dict, accept/gate results, enqueue newly triggered requests from the current observation snapshot, then return scheduler actions.

- [ ] **Step 5: Preserve synchronous modes**

Keep `chunk` and `synced_batch` behavior unchanged. Keep the existing `GR00T_TIMING` diagnostics for synchronous mode and add structured async metrics separately.

- [ ] **Step 6: Run policy tests**

Expected: existing sync tests and new async tests all pass.

- [ ] **Step 7: Review diff checkpoint**

Confirm close/reset are idempotent and stale async results cannot enter a reset environment.

### Task 4: Visible Status HUD and Metrics

**Files:**
- Create: `isaaclab_arena_gr00t/policy/async_status_ui.py`
- Create: `isaaclab_arena_gr00t/policy/async_metrics.py`
- Create: `isaaclab_arena_gr00t/tests/test_async_metrics.py`
- Modify: `isaaclab_arena_gr00t/policy/gr00t_remote_closedloop_policy.py`

- [ ] **Step 1: Write failing metric tests**

Given per-env request counts, inference samples, queue samples, and hold steps, assert p50/p95/p99, miss rate, hold duration, real-time factor metadata, and `zero_miss`.

- [ ] **Step 2: Implement pure metric aggregation**

Return JSON-serializable values only. Write atomically using a temporary file plus `Path.replace`.

- [ ] **Step 3: Implement lazy Kit HUD**

Import `omni.ui` only when enabled and SimulationApp is already running. Show one row per robot with status text and color: executing green, queued yellow, inference/gated blue, deadline miss red. Include simulated time, queue depth, total misses, and measured inference p95.

- [ ] **Step 4: Wire metrics and HUD into async policy**

Update after each action step, write the final JSON on close, and print one concise `[ASYNC_VLA_SUMMARY]` JSON line for machine parsing.

- [ ] **Step 5: Run unit tests**

Expected: metric tests and all GR00T policy tests pass with HUD disabled.

### Task 5: Capacity Summary Tool and Runbook

**Files:**
- Create: `isaaclab_arena_gr00t/scripts/summarize_async_capacity.py`
- Create: `isaaclab_arena_gr00t/tests/test_summarize_async_capacity.py`
- Modify: `LOCAL_RUNTIME_SETUP.md`

- [ ] **Step 1: Write failing summary tests**

Feed JSON fixtures for N=1,2,3,4 and assert the tool reports the largest N with `deadline_miss_rate == 0`, plus the first overloaded N.

- [ ] **Step 2: Implement summary CLI**

Accept one or more metric JSON paths and print a compact table sorted by `num_envs` with p95 inference, p95 virtual queue wait, misses, hold time, and zero-miss status.

- [ ] **Step 3: Add exact Windows commands to the runbook**

Document server health check and `policy_runner.py` commands for N=2 visible baseline, N=4 saturation demo, and a headless N sweep. Explicitly state that capacity deadlines use `prefetch_lead_steps * 0.02`, while `sim_time / wall_time` is reported separately.

- [ ] **Step 4: Run summary tests and `--help`**

Expected: tests pass and CLI help exits zero.

### Task 6: Runtime Verification and Capacity Sweep

**Files:**
- Runtime outputs only: `eval/async_vla_demo/`

- [ ] **Step 1: Verify the existing GR00T server**

Confirm `isaaclab_arena_gr00t_server` is running and port 5555 responds to ping. Do not rebuild Docker or modify `docker/`.

- [ ] **Step 2: Run focused unit tests**

Run scheduler, worker, policy, metrics, and summary tests with the Windows Arena Python environment.

- [ ] **Step 3: Run N=2 smoke test**

Use the G1 brown-box task, `--num_envs 2`, tiled cameras, `async_edf`, `lead_steps=25`, metrics output, and a short rollout. Verify both robots move and no unhandled worker errors occur.

- [ ] **Step 4: Run visible saturation demo**

Run N=4 or the smallest N that produces at least one virtual deadline miss. Use Kit visualization and HUD. Record a viewport video. If viewport encoding excludes the HUD, also save a screenshot of the live HUD; always preserve the metrics JSON next to the visual evidence.

- [ ] **Step 5: Sweep capacity**

Run N=1 upward until two consecutive counts show misses or GPU/VRAM prevents startup. Keep `lead_steps`, scene, cameras, and rollout length fixed. Summarize with `summarize_async_capacity.py`.

- [ ] **Step 6: Verify scientific invariants**

Check that deadline windows equal `lead_steps * 0.02`, no metric uses 5.70-second wall execution as slack, actual inference latency remains wall-clock measured, and simulation real-time factor is reported independently.

- [ ] **Step 7: Final verification**

Run `git diff --check`, inspect the focused diff, and report tests, demo paths, measured zero-miss capacity, first overloaded count, and any remaining hardware-bound limitation.
