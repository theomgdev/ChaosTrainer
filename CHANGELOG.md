# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
