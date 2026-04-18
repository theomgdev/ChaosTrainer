# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
