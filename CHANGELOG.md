# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - Unreleased

### Changed
- `perturbation_std` now defaults to `None`. When `None`, the standard deviation `ε` is dynamically computed per parameter based on its Root Mean Square (RMS) scaling to adapt to fine-tuning layers with vastly different scales. This dynamic calculation avoids GPU-to-CPU synchronization, preserving training performance.

## [0.4.0] - 2026-04-20

### Added
- `estimate_grad(model, criterion, *args, **kwargs)` public method on `Chaos`.
  Runs the identical ES forward passes as `step()` but writes the resulting
  per-parameter gradient estimate into `param.grad` (accumulating, like
  `loss.backward()`) instead of applying the LARS momentum update. Enables
  three new usage patterns without any changes to existing code:
  - **Pair with any standard optimizer** — `chaos.estimate_grad(...)` then
    `adamw.step()`: AdamW (or any optimizer) owns the update rule while Chaos
    supplies gradient-free gradient estimates.
  - **Gradient-free RL / black-box objectives** — reward signal goes through
    `estimate_grad`; no `backward()` call needed at any point.
  - **ES + backprop hybrid** — call `estimate_grad` then `loss.backward()` to
    accumulate both signal sources into `param.grad` before a single
    `optimizer.step()`.
- `_run_es(model, criterion, *args, **kwargs)` private method that encapsulates
  the entire ES forward-pass loop, returning `(optim_params, grad_acc,
  group_slices, mean_loss)`. Both `step()` and `estimate_grad()` delegate to
  `_run_es`, eliminating all duplicate ES logic.
- `examples/xor_adamw.py` — new example demonstrating `estimate_grad` paired
  with `torch.optim.AdamW`.

### Changed
- `step()` refactored to call `_run_es()` internally; behaviour is identical
  to prior releases.
- README expanded with four usage sections: standalone ES, `estimate_grad` +
  external optimizer, gradient-free RL, and ES+backprop hybrid.

### Notes
- **Why not replace LARS with AdamW by default?** Benchmarks (8 seeds, XOR)
  show LARS converges in ~294 steps on average vs ~530 for ES+AdamW at the
  same `num_perturbations=8`. AdamW's second moment accumulates the ES
  estimator variance and shrinks steps disproportionately; LARS's norm ratio
  is noise-invariant by design. `estimate_grad` lets users opt in to AdamW (or
  any other rule) when they prefer its properties.

## [0.3.1] - 2026-04-20

### Fixed
- `vmap` was constructed without `randomness="different"`, causing a hard crash
  when the model contains stochastic layers (e.g. `nn.Dropout`). Changed to
  `randomness="different"` so each of the K vectorized evaluations receives an
  independent RNG stream. Models without stochastic layers are unaffected.

### Changed
- `examples/xor.py` default `--num-perturbations` corrected from `1` to `8`
  to match the library default and produce representative out-of-the-box
  results. Default `--epochs` reduced from `15000` to `3000` (with K=8 and
  the default fitness-shaping + orthogonal settings, convergence is reliably
  faster). Log interval adjusted to every 500 steps.
- README AMP section corrected: `torch.autocast` contexts do not wrap the
  optimizer's internal `vmap` forward passes. The section now advises converting
  the model dtype directly (`model.to(torch.bfloat16)`) rather than using an
  autocast wrapper. Usage example loop count updated from `15_000` to `3_000`.
- Added a docstring note clarifying that only `lr`, `beta`, and `weight_decay`
  are per-param-group overridable; all other hyperparameters are optimizer-wide.

## [0.3.0] - 2026-04-20

### Changed
- `num_perturbations` default changed from `1` to `8`. Single-sample estimates
  are extremely noisy in practice; 8 samples give reliable convergence on most
  objectives at a compute cost that remains acceptable for ES-sized models.
- `fitness_shaping` default changed from `False` to `True`. Centered rank
  transform is strictly more robust than raw loss differences — it costs no
  additional forward passes and makes convergence invariant to loss scale and
  monotonic transformations. Disable only to reproduce raw-estimator baselines.
- `orthogonal_perturbations` default changed from `False` to `True`.
  Orthogonal directions explore the loss landscape more systematically than
  i.i.d. Gaussian at the same sample count, with negligible QR overhead for
  ES-sized models. The i.i.d. fallback remains automatic when
  `num_perturbations > param.numel()`.

## [0.2.0] - 2026-04-19

### Added
- `weight_decay` hyperparameter on `Chaos`. Decoupled L2 regularization
  applied per parameter-group as `θ ← θ · (1 − lr · λ)` after each ES step,
  in the spirit of AdamW. Default `0.0` (disabled). Stored in `defaults` and
  therefore per-group-overridable.
- `grad_clip` hyperparameter on `Chaos`. Clips the global L2 norm of the ES
  gradient estimate to this threshold before the momentum update, preventing
  runaway steps on noisy or sparse objectives. Implemented via a tensor
  multiply so no GPU→CPU sync is required. Default `None` (disabled).
- `fitness_shaping` flag on `Chaos`. When enabled, replaces raw loss
  differences with centered rank scores computed over all `2 · num_perturbations`
  evaluations before forming the gradient estimate. Makes the optimizer
  invariant to monotonic transformations of the loss (e.g. shifted or scaled
  rewards), which is especially valuable for RL-style or highly non-stationary
  objectives. Default `False`.
- `orthogonal_perturbations` flag on `Chaos`. When enabled, generates
  perturbation directions via QR orthogonalization (in float32 for numerical
  stability) per parameter, so the `num_perturbations` noise vectors span
  orthogonal subspaces. Reduces estimator variance compared to i.i.d. Gaussian
  at the same sample count. Falls back to i.i.d. when
  `num_perturbations > param.numel()`. Default `False`.
- `_generate_noises` private helper encapsulates noise generation (i.i.d. or
  orthogonal) for the two-phase code path.
- `_centered_rank_coef` module-level helper implements the centered rank
  transform over the `2K` antithetic losses.

### Changed
- `step()` now branches on `fitness_shaping or orthogonal_perturbations`: the
  enhanced path pre-generates all `K` noise tensors and collects all `2K`
  losses before computing coefficients (`O(K · |θ|)` additional noise memory;
  `perturbation_chunk_size` still governs peak activation VRAM). The standard
  path is unchanged.

## [0.1.3] - 2026-04-19

### Added
- `perturbation_std` hyperparameter on `Chaos`. Controls the standard deviation
  `ε` of the Gaussian perturbation `δ ~ N(0, ε² I)`. Useful for fp16 training
  (lower toward `1e-4` to stay above the FP noise floor) or noisy loss surfaces
  (raise toward `1e-2`). Default `1e-3` is unchanged from prior releases.
  Stored in `defaults` and round-trips through `state_dict()`.

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
