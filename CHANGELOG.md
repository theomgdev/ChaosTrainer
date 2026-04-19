# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-19

### Fixed
- Decoupled weight decay now applied before the momentum update (AdamW
  ordering per Loshchilov & Hutter 2019). The previous order scaled the
  preconditioned update itself by `(1 − lr·wd)` — numerically `O(lr²·wd)` and
  harmless at typical `lr`, but not the rigorous decoupled form.

### Changed
- Unified `Chaos` and the experimental `ChaosStream` into a single optimizer
  class. `Chaos` now dispatches internally on parameter device: CUDA uses the
  stream-parallel + `CUDAGraph`-captured path (racy, non-deterministic); CPU
  and other devices use the deterministic `torch.func.vmap` path. Both paths
  share the same public API, hyperparameters, and return value (the clean
  baseline `L(θ)` taken before the step's update).
- New default `weight_decay=0.01` on `Chaos`. Decoupled weight decay
  (`θ ← (1 − lr·wd)·θ`) counteracts the radial drift of `‖θ‖` under
  LARS-driven random-direction updates. Without it, scale-sensitive losses
  (e.g. cross-entropy) stop reflecting accuracy as training progresses.
- `num_perturbations` default raised from `1` to `8`. The CUDA path amortizes
  perturbations across concurrent streams, and `K=1` is a poor ES estimator.
- `use_cuda_graph: bool = True` is now a top-level `Chaos` flag, active on the
  CUDA path. Disable when composing with `torch.compile(mode=
  "reduce-overhead")` so the two graph managers do not conflict.
- `perturbation_chunk_size` is now CPU-only; on CUDA the stream fan-out
  already controls concurrency and the flag is ignored.

### Removed
- `chaostrainer.optim.ChaosStream` as a separate class; its behavior is the
  CUDA branch of `Chaos`. **Breaking:** the `ChaosStream` import is gone.
- `examples/xor_stream.py` and `examples/mnist_stream.py`; `examples/xor.py`
  and `examples/mnist.py` cover both paths via `--device`.
- `tests/test_chaos_stream.py`; CUDA-path coverage lives in
  `tests/test_chaos.py` behind `@pytest.mark.skipif(not torch.cuda.is_available())`.

## [0.1.3] - 2026-04-19

### Added
- `ChaosStream` now pre-allocates per-stream noise buffers, the gradient
  accumulator, and the loss scalar once and fills them in place each step,
  eliminating `num_perturbations × len(params)` per-step allocator traffic.
- `ChaosStream` now captures the full multi-stream step into a
  `torch.cuda.CUDAGraph` on the first call and replays it thereafter,
  collapsing thousands of kernel launches into a single replay. Graph is
  rebuilt automatically when input shapes or the parameter set change.
- `use_cuda_graph: bool = True` constructor flag on `ChaosStream` to opt out
  (e.g. when composing with `torch.compile(mode="reduce-overhead")`).
- `--no-cuda-graph` flag on the `ChaosStream` examples.

### Fixed
- `ChaosStream.step` now returns the clean loss `L(θ)` from a single baseline
  forward taken before any stream forks, rather than the mean of per-stream
  `loss_plus` values. Those are evaluated on a racy superposition of in-flight
  perturbations — wildly biased in magnitude and misleading as a training
  signal. One extra forward per step (captured in the same graph).

## [0.1.2] - 2026-04-19

### Added
- `perturbation_chunk_size` hyperparameter on `Chaos`. When set, perturbations
  are evaluated in sequential chunks of this size instead of a single vmap
  over all `num_perturbations`, capping peak activation VRAM while preserving
  vmap amortization.

### Changed
- Plus / minus vmap forward passes are now sequenced with eager release, so
  only one side's batched parameter tuple is live at a time (~33% parameter-
  space VRAM reduction).
- Per-parameter gradient reduction uses a matmul over the flattened
  perturbation axis instead of broadcast-multiply-then-mean, avoiding a
  `[K, *p.shape]` intermediate allocation.

## [0.1.1] - 2026-04-19

### Changed
- Migrated `Chaos.step` to a `torch.func.vmap` architecture: all
  `num_perturbations` samples are evaluated in two fused forward passes
  (antithetic plus/minus) instead of a Python loop.
- Perturbations and updates are now driven exclusively from the optimizer's
  `param_groups`, so partial-parameter optimizers and reordered param groups
  behave correctly. Parameters on the model that are not registered with the
  optimizer are no longer perturbed.
- Trust-ratio step now applied without forcing a per-step GPU→CPU sync.
- `num_perturbations` is stored in `defaults` and therefore round-trips
  through `state_dict()`.

### Fixed
- Gradient accumulator could misalign with `param_groups` when group order
  differed from `model.named_parameters()` order.

## [0.1.0] - 2026-04-18

### Added
- Initial release of the `Chaos` optimizer.
- Antithetic (central-difference) Evolution-Strategy gradient estimator.
- LARS-style global trust-ratio adaptive learning rate.
- Per-parameter-group hyperparameters (`lr`, `beta`).
- Optimizer-level `num_perturbations` control.
- `state_dict` / `load_state_dict` support inherited from
  `torch.optim.Optimizer`.
- CPU and CUDA test coverage, XOR example.
