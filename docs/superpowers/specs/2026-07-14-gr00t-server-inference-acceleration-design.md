# GR00T Server Inference Acceleration Design

## Goal

Add optional `torch.compile` and DiT-only TensorRT acceleration to an Arena-owned
GR00T inference server, while preserving the existing server CLI and wire protocol.
Measure the acceleration on the local GR00T N1.6 G1 locomanipulation checkpoint before
considering N1.7 full-pipeline TensorRT.

## Success Criteria

- The original `submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py` remains
  unchanged and usable.
- Existing clients, including synchronous and asynchronous Arena policies, require no
  changes.
- The Arena-owned server accepts the original server arguments with the same defaults.
- Omitting all acceleration arguments produces eager PyTorch behavior equivalent to
  the current server.
- The accelerated modes fail clearly during startup when their requirements are not
  met; they never silently fall back to eager mode.
- On the local N1.6 `checkpoint-20000`, benchmark 50 post-warmup inferences with the
  same input and report median latency, p95 latency, and frequency. A reduction of at
  least 20 percent in median server-side end-to-end inference latency is sufficient to
  keep this phase-one approach.

## Non-Goals

- Do not modify `submodules/Isaac-GR00T`.
- Do not update the Isaac-GR00T submodule or add N1.7 full-pipeline TensorRT.
- Do not change action horizons, action chunks, scheduling, client transport, or
  simulation control timing.
- Do not alter the in-progress multi-robot asynchronous inference implementation.
- Do not silently recover from compilation, CUDA, model-layout, or TensorRT errors.

## Interface Compatibility

The new Arena entry point exposes the same fields as the upstream `ServerConfig`:

- `model_path`
- `embodiment_tag`
- `device`
- `dataset_path`
- `modality_config_path`
- `execution_horizon`
- `host`
- `port`
- `strict`
- `use_sim_policy_wrapper`

It adds only these optional fields:

- `inference_mode`: `eager`, `compile`, or `tensorrt`; default `eager`.
- `compile_mode`: the `torch.compile` mode; default `max-autotune`.
- `trt_engine_path`: path to a serialized DiT TensorRT engine; required only for
  `tensorrt`.

The server continues to use Isaac-GR00T's `PolicyServer`, so request serialization,
response serialization, health checks, ports, and Arena client behavior remain
unchanged. Existing commands can continue to invoke the upstream entry point. To use
acceleration, operators invoke the Arena entry point with the same arguments plus an
acceleration mode.

## Architecture

### Inference backend

A focused Arena module owns acceleration setup. It provides one public operation that
receives a loaded `Gr00tPolicy` and the selected mode, validates requirements, applies
the backend, and returns the same policy object.

- `eager` leaves the policy unchanged.
- `compile` replaces only `policy.model.action_head.model.forward` with
  `torch.compile(..., mode=compile_mode)`.
- `tensorrt` replaces the same DiT `forward` method with a lightweight wrapper around
  a deserialized TensorRT engine.

TensorRT imports are lazy so eager and compile modes do not require TensorRT to be
installed. The TensorRT wrapper retains references to its runtime, engine, and
execution context for the lifetime of the policy. Inputs are made contiguous on the
selected CUDA device, named engine bindings receive their addresses, and output is
allocated as BF16 to match the existing N1.6 export pipeline.

The implementation is adapted from the existing N1.6 deployment helpers, but is kept
inside Arena and contains no dataset, plotting, or standalone-evaluation dependencies.

### Server entry point

The Arena server mirrors the upstream server construction flow:

1. Select a model policy when `model_path` is provided; otherwise select a replay
   policy when `dataset_path` is provided, matching the upstream precedence.
2. Construct `Gr00tPolicy` or `ReplayPolicy` using the original arguments.
3. Apply the selected backend only to a `Gr00tPolicy`.
4. Apply `Gr00tSimPolicyWrapper` when requested.
5. Start the unchanged Isaac-GR00T `PolicyServer`.

Replay policies support only `eager`, because they do not contain a GR00T action head.
The server startup log prints the selected mode and, for TensorRT, the engine path.

## Validation And Errors

Startup validation covers the following cases:

- At least one of `model_path` and `dataset_path` must be provided; `model_path` takes
  precedence when both are present, matching the upstream server.
- `compile` and `tensorrt` require a GR00T model policy.
- Accelerated modes require the expected `model.action_head.model.forward` structure.
- TensorRT requires CUDA, a valid CUDA device, an existing engine file, the TensorRT
  Python package, a deserializable engine, and all expected named bindings.
- `trt_engine_path` is accepted only with `tensorrt`; eager and compile modes reject
  it with a clear configuration error.

Errors identify the failed requirement and stop server startup. Runtime TensorRT
execution failures raise an error rather than returning a stale or empty action.

## Testing

Unit tests use small fake policy objects and injected compile/TensorRT dependencies so
they do not require a model, CUDA, Isaac Sim, or a TensorRT installation. They verify:

- Eager mode preserves the original `forward` callable.
- Compile mode wraps exactly the DiT `forward` callable with the configured mode.
- TensorRT mode validates its engine path and installs a signature-compatible forward
  callable.
- Invalid mode/backend combinations fail with specific messages.
- The Arena server configuration contains all upstream fields with matching defaults.
- Existing server arguments construct the same policy and `PolicyServer` inputs.
- New arguments default to eager and do not affect replay mode.

Tests are written and observed failing before production implementation. Package tests
run inside the clone's Arena development container; host `pre-commit` checks only the
files changed for this feature so unrelated user work is not modified.

## Benchmark

The primary benchmark measures server-side policy inference, from entering
`Gr00tPolicy.get_action` until the action result is available. Network transfer and
simulation time are excluded because this phase does not change them.

Use the local G1 locomanipulation N1.6 checkpoint at
`models/isaaclab_arena/locomanipulation_tutorial/checkpoint-20000` and one fixed,
valid observation for every mode. For each mode:

1. Load a fresh policy.
2. Perform enough warmup calls to finish CUDA initialization and, for compile mode,
   graph compilation.
3. Record 50 synchronized CUDA inference samples.
4. Report median, p95, mean, standard deviation, and equivalent Hz.

Run eager and compile first. If the container already has the TensorRT dependencies,
export and build the N1.6 DiT engine using the existing upstream deployment scripts,
then run the same measurement for TensorRT. Engine generation artifacts stay outside
git. If dependencies or GPU runtime are unavailable, report that separately rather
than treating it as a performance result.

## Decision Rule

Keep phase one when either compile or DiT-only TensorRT reduces median server-side
end-to-end latency by at least 20 percent without changing output shape or producing
runtime errors. If neither mode reaches that threshold, the next design phase will
evaluate an Isaac-GR00T update and N1.7 full-pipeline TensorRT.
