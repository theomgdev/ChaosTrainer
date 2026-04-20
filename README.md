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

### Standalone ES optimizer

The simplest path: `Chaos.step` drives every forward pass itself and applies
the LARS-style update — no `loss.backward()` needed.

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

for step in range(3_000):
    loss = optimizer.step(model, loss_fn, X, Y).item()
    if loss < 1e-4:
        break
```

### Pair with any standard optimizer (`estimate_grad`)

`estimate_grad` runs the same ES forward passes but writes the gradient
estimate into `param.grad` instead of applying the update — exactly like
`loss.backward()`. This lets any PyTorch optimizer consume the ES gradient:

```python
chaos = Chaos(model.parameters(), num_perturbations=16)
adamw = torch.optim.AdamW(model.parameters(), lr=1e-2)

for data, target in dataloader:
    adamw.zero_grad()
    loss = chaos.estimate_grad(model, loss_fn, data, target)
    adamw.step()
```

### Gradient-free RL / black-box objectives

No `backward()` needed at all — the reward signal never touches autograd:

```python
chaos = Chaos(model.parameters())
adamw = torch.optim.AdamW(model.parameters(), lr=1e-2)

for state in env:
    adamw.zero_grad()
    mean_return = chaos.estimate_grad(model, rollout_fn, state)
    adamw.step()
```

### Mix ES gradients with backprop

Both signal sources accumulate into `param.grad` before the update step:

```python
adamw.zero_grad()
es_loss = chaos.estimate_grad(model, loss_fn, data, target)
# compute loss outside no_grad for backward:
bp_loss = loss_fn(model(data), target)
bp_loss.backward()  # adds onto the ES estimate already in .grad
adamw.step()
```

## Algorithm

Each step:

1. Sample `δ_k` for every parameter, `k = 1 … num_perturbations`. By default
   (`orthogonal_perturbations=True`) the `K` directions per parameter are
   orthogonalized via QR (in float32), so they span orthogonal subspaces and
   cover the loss landscape more systematically than i.i.d. Gaussian noise.
2. Evaluate `L(θ + δ_k)` and `L(θ − δ_k)` in parallel via `vmap` — one
   vmapped forward pass per direction per chunk, amortizing Python dispatch
   across all `perturbation_chunk_size` samples at once.
3. Form the antithetic Evolution-Strategy gradient estimate and average across
   samples. By default (`fitness_shaping=True`) raw losses are replaced by
   centered rank scores over all `2K` evaluations before weighting:

   ```
   rank all 2K losses jointly → center to [−0.5, 0.5]
   ĝ = mean_k [ (rank(L(θ+δ_k)) − rank(L(θ−δ_k))) · δ_k / (2 ε²) ]
   ```

   This makes the update invariant to any monotonic transformation of the
   loss (rescaling, shifting, non-linear reward shaping). With raw losses
   (`fitness_shaping=False`) the estimator is the standard:

   ```
   ĝ = mean_k [ (L(θ+δ_k) − L(θ−δ_k)) · δ_k / (2 ε²) ]
   ```

4. Update the momentum buffer `m ← β · m + ĝ`.
5. Take a LARS-inspired step rescaled by the global weight-to-momentum norm
   ratio:

   ```
   η = lr · ‖θ‖_global / ‖m‖_global
   θ ← θ − η · m
   ```

   The effective per-step displacement is approximately `lr · ‖θ‖`, so
   `lr ≈ 1e-2` implements the "per-step change ≈ 1% of weights" heuristic.

## Hyperparameters

| Name                          | Default | Notes |
|-------------------------------|---------|-------|
| `lr`                          | `1e-2`  | Effective per-step displacement as a fraction of `‖θ‖`. |
| `beta`                        | `0.9`   | Momentum decay on the gradient estimate. |
| `weight_decay`                | `0.0`   | Decoupled L2 regularization coefficient (AdamW-style). Applied per-group as `θ ← θ · (1 − lr · λ)` after the ES step. |
| `num_perturbations`           | `8`     | Samples averaged per step. More samples reduce variance linearly at proportional forward-pass cost. |
| `perturbation_chunk_size`     | `None`  | Micro-batch size for the vmap forward (caps peak activation VRAM). `None` ⇒ one chunk of size `num_perturbations`. |
| `perturbation_std`            | `None`  | Standard deviation `ε` of `δ ~ N(0, ε² I)`. If `None`, computed dynamically per parameter via RMS. The estimator variance is independent of `ε`; bias vanishes as `O(ε²)`. |
| `grad_clip`                   | `None`  | Global L2 gradient norm clip threshold, applied before the momentum update. No GPU→CPU sync. |
| `fitness_shaping`             | `True`  | Replace raw loss differences with centered rank scores over all `2K` evaluations. Makes the update invariant to monotonic loss transformations. |
| `orthogonal_perturbations`    | `True`  | Orthogonalize the `K` noise directions per parameter via QR (in float32). Reduces estimator variance at the same sample count. Falls back to i.i.d. when `K > param.numel()`. |

`lr`, `beta`, and `weight_decay` are per-parameter-group and can be overridden
via the standard PyTorch `param_groups` mechanism. `num_perturbations`,
`perturbation_chunk_size`, `perturbation_std`, `grad_clip`, `fitness_shaping`,
and `orthogonal_perturbations` are optimizer-level flags.

> **Memory note:** `fitness_shaping` and `orthogonal_perturbations` require all
> `K` noise tensors to be held in memory simultaneously (`O(K · |θ|)` extra),
> because fitness shaping needs all `2K` losses before forming any coefficient
> and orthogonal perturbations need `K` jointly-generated directions.
> `perturbation_chunk_size` still controls peak *activation* VRAM in both modes.

### Tuning tips

- Start with defaults — they are calibrated for reliable convergence across
  a broad range of tasks without any tuning.
- Raise `lr` (e.g. `1e-2`) on small, well-conditioned problems; lower it
  for fine-tuning or noisy objectives.
- Lower `beta` (e.g. `0.5`) when rapid adaptation matters; raise it
  (`0.95 – 0.99`) for smoother trajectories in flat regions.
- Lower `num_perturbations` (e.g. `2–4`) only when compute is very tight.
  Raise it (e.g. `16–64`) on high-variance or high-dimensional objectives
  where the default 8 samples leave the estimate noisy.
- Adjust `perturbation_std` only when the dynamic parameter scaling fails: explicitly set to
  `1e-4` for fp16 training (to stay above the FP noise floor), or raise toward
  `1e-2` for very noisy or flat loss surfaces.
- Use `weight_decay` (e.g. `1e-4`) as a regularizer for large models; it
  costs nothing extra and applies per-group.
- Add `grad_clip` (e.g. `1.0` or `10.0`) for objectives with occasional
  extreme loss spikes or very high-variance gradient estimates.
- Set `fitness_shaping=False` only when reproducing results from code that
  used raw loss differences, or when the fixed-magnitude coefficient at
  `num_perturbations=1` is undesirable.
- Set `orthogonal_perturbations=False` only when reproducing i.i.d.-Gaussian
  baselines, or when the QR cost is significant at very large
  `num_perturbations`. The fallback to i.i.d. is automatic when
  `num_perturbations > param.numel()`.

### Performance & VRAM Optimization (Pro Tips)

- **Vectorized perturbations via `vmap`:** Each chunk of `num_perturbations`
  samples is evaluated in two fused forward passes rather than a Python loop,
  so Python dispatch overhead is amortized across the entire chunk. The plus
  and minus forward passes are sequenced so their working sets do not overlap
  in VRAM.
- **Cap peak VRAM with `perturbation_chunk_size`:** Activation memory scales
  with the chunk size, not `num_perturbations`. For large `K` (e.g. 1000),
  setting `perturbation_chunk_size=64` keeps vmap amortization intact while
  cutting peak activation VRAM by `K/chunk_size`. Use this when OOM risk
  forces you to choose between batch size and sample count.
- **Multi-tensor `_foreach` momentum/update:** Parameter-level bookkeeping
  (momentum step, norm reduction, weight update) uses PyTorch's C++
  multi-tensor kernels, avoiding per-parameter Python overhead.
- **`torch.compile()` is your best friend:** Wrapping `model = torch.compile(model)`
  fuses the vmapped forward into a single kernel, dramatically speeding up
  large `num_perturbations` runs.
- **Automatic Mixed Precision (AMP):** Since Chaos drives its own forward passes
  internally, user-level `torch.autocast` contexts do not wrap them. To use
  AMP, convert the model to `fp16` or `bf16` directly (e.g.
  `model.to(torch.bfloat16)`) before passing it to `optimizer.step`. This
  halves activation memory and unlocks TensorCore acceleration without any
  autocast wrapper.

## Running the examples

```bash
# Standalone ES (LARS update)
python examples/xor.py
python examples/xor.py --device cuda

# estimate_grad + AdamW
python examples/xor_adamw.py
python examples/xor_adamw.py --device cuda
```

## License

MIT — see [LICENSE](LICENSE).

## References

- Salimans et al. 2017, *Evolution Strategies as a Scalable Alternative to
  Reinforcement Learning.* <https://arxiv.org/abs/1703.03864>
- You et al. 2017, *Large Batch Training of Convolutional Networks* (LARS).
  <https://arxiv.org/abs/1708.03888>
