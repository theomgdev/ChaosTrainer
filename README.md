# ChaosTrainer

Gradient-free PyTorch optimizer using Evolution-Strategy style stochastic
perturbations combined with LARS-style global trust-ratio adaptive step sizing.

`chaostrainer.Chaos` is a `torch.optim.Optimizer` subclass that works
transparently on CPU / CUDA / MPS, vectorizes perturbation evaluation via
`torch.func.vmap`, and supports parameter groups, `state_dict()` checkpointing,
and per-group hyperparameters.

Unlike standard PyTorch optimizers, `Chaos.step` takes `(model, criterion,
*args, **kwargs)` rather than an optional `closure`: the optimizer drives the
forward pass itself through `torch.func.functional_call`, so `loss.backward()`
is never needed.

## Why gradient-free?

Chaos is useful when first-order optimization is inconvenient or impossible:

- **Non-differentiable losses** — discrete outputs, hard-argmax, reward signals.
- **Black-box simulators** — physics engines, emulators, external programs.
- **Autograd-unavailable graphs** — frozen compiled modules, C extensions.
- **Memory-constrained training** — no activation graph is kept across the step.
- **Sanity baselines** — validate that your loss is actually informative before
  committing to a more elaborate optimizer.

It is **not** a replacement for Adam / SGD on standard differentiable deep
learning workloads; variance of the gradient estimate scales poorly with
parameter count. Use it where gradients cost more than ~`num_params / 100`
forward passes, or where they simply don't exist.

## Install

```bash
pip install chaostrainer
```

From source:

```bash
git clone https://github.com/theomgdev/chaostrainer
cd chaostrainer
pip install -e ".[dev]"
pytest
```

## Usage

```python
import torch
from torch import nn
from chaostrainer import Chaos

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.Tanh(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([[0.], [1.], [1.], [0.]])

loss_fn = nn.MSELoss()
optimizer = Chaos(model.parameters(), lr=1e-2)

for step in range(15_000):
    loss = optimizer.step(model, loss_fn, X, Y).item()
    if loss < 1e-4:
        break
```

## Algorithm

Each step:

1. Sample `δ_k ~ N(0, ε² I)` independently for every parameter, for
   `k = 1 … num_perturbations`.
2. Evaluate `L(θ + δ_k)` and `L(θ − δ_k)` in parallel via `vmap` over the
   perturbation dimension — one vmapped forward pass for the plus batch and
   one for the minus batch, regardless of `num_perturbations`.
3. Form the antithetic Evolution-Strategy estimator and average across samples:

   ```
   ĝ = mean_k [ (L(θ+δ_k) − L(θ−δ_k)) · δ_k / (2 ε²) ]
   ```

4. Update the momentum buffer `m ← β · m + ĝ`.
5. Take a LARS-inspired step rescaled by the global weight-to-momentum norm
   ratio

   ```
   η = lr · ‖θ‖_global / ‖m‖_global
   θ ← θ − η · m
   ```

   The effective per-step displacement is approximately `lr · ‖θ‖`, so
   `lr ≈ 1e-3` implements the "per-step change ≈ 0.1% of weights" heuristic.

## Hyperparameters

| Name                | Default | Notes |
|---------------------|---------|-------|
| `lr`                | `1e-3`  | Effective per-step displacement as a fraction of `‖θ‖`. |
| `beta`              | `0.9`   | Momentum decay on the gradient estimate. |
| `num_perturbations` | `1`     | Samples averaged per step. |

`lr` and `beta` are per-parameter-group and can be overridden via the standard
PyTorch `param_groups` mechanism. `num_perturbations` is an optimizer-level flag.

The perturbation std `ε = 1e-3` is a fixed internal constant. The variance of
the central-difference ES estimator is independent of `ε`, and its bias
vanishes as `O(ε²)`, so this value works across model scales for fp32 and
mixed-precision (AMP) training without tuning.

### Tuning tips

- Start with defaults. Raise `lr` (e.g. `1e-2`) on small, well-conditioned
  problems; lower it for fine-tuning or noisy objectives.
- Lower `beta` (e.g. `0.5`) when rapid adaptation matters; raise it
  (`0.95 – 0.99`) for smoother trajectories in flat regions.
- Increase `num_perturbations` when the gradient estimate is too noisy and
  convergence stalls — cost scales linearly.

### Performance & VRAM Optimization (Pro Tips)

- **Vectorized perturbations via `vmap`:** All `num_perturbations` samples are
  evaluated in two fused forward passes rather than a Python loop, so Python
  dispatch overhead is amortized across the entire batch of perturbations.
  Peak VRAM scales with `num_perturbations × batch_size × activation_size` —
  reduce `num_perturbations` if you OOM.
- **Multi-tensor `_foreach` momentum/update:** Parameter-level bookkeeping
  (momentum step, norm reduction, weight update) uses PyTorch's C++
  multi-tensor kernels, avoiding per-parameter Python overhead.
- **`torch.compile()` is your best friend:** Wrapping `model = torch.compile(model)`
  fuses the vmapped forward into a single kernel, dramatically speeding up
  large `num_perturbations` runs.
- **Native Automatic Mixed Precision (AMP):** Since no autograd graph is kept,
  running forward passes under `torch.autocast` in `fp16` / `bf16` halves the
  memory footprint and unlocks TensorCore acceleration.

## Running the example

```bash
python examples/xor.py
python examples/xor.py --device cuda
```

## License

MIT — see [LICENSE](LICENSE).

## References

- Salimans et al. 2017, *Evolution Strategies as a Scalable Alternative to
  Reinforcement Learning.* <https://arxiv.org/abs/1703.03864>
- You et al. 2017, *Large Batch Training of Convolutional Networks* (LARS).
  <https://arxiv.org/abs/1708.03888>
