# ChaosTrainer

Gradient-free PyTorch optimizer using Evolution-Strategy style stochastic
perturbations combined with LARS-style global trust-ratio adaptive step
sizing and decoupled weight decay.

`chaostrainer.Chaos` is a `torch.optim.Optimizer` subclass that transparently
dispatches on parameter device:

- **CUDA** — each perturbation is launched on its own `torch.cuda.Stream`, and
  the full multi-stream step is captured into a `torch.cuda.CUDAGraph` and
  replayed each iteration, collapsing thousands of kernel launches into a
  single replay. Streams mutate the shared parameters in place without
  synchronization: the race-by-design dynamic behaves like lock-free
  evolutionary crossover. Trajectories are non-deterministic even with a
  fixed seed.
- **CPU / other** — perturbations are evaluated through a single vectorized
  `torch.func.vmap` over the perturbation axis. Deterministic under a fixed
  seed.

Both paths share the same public API, hyperparameters, and return value
(the clean baseline loss `L(θ)` taken before the step's update), support
parameter groups, per-group hyperparameters, and `state_dict()`
checkpointing.

Unlike standard PyTorch optimizers, `Chaos.step` takes `(model, criterion,
*args, **kwargs)` rather than an optional `closure`: the optimizer drives the
forward pass itself, so `loss.backward()` is never needed.

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
optimizer = Chaos(model.parameters(), lr=1e-2, weight_decay=0.0)

for step in range(15_000):
    loss = optimizer.step(model, loss_fn, X, Y).item()
    if loss < 1e-4:
        break
```

## Algorithm

Each step:

1. Sample `δ_k ~ N(0, ε² I)` independently for every parameter, for
   `k = 1 … num_perturbations`.
2. Evaluate `L(θ + δ_k)` and `L(θ − δ_k)` in parallel — on CUDA via one
   stream per perturbation, on CPU via a single `vmap` over the perturbation
   axis.
3. Form the antithetic Evolution-Strategy estimator and average across samples:

   ```
   ĝ = mean_k [ (L(θ+δ_k) − L(θ−δ_k)) · δ_k / (2 ε²) ]
   ```

4. Update the momentum buffer `m ← β · m + ĝ`.
5. Take a LARS-inspired step rescaled by the global weight-to-momentum norm
   ratio and apply decoupled weight decay:

   ```
   η = lr · ‖θ‖_global / ‖m‖_global
   θ ← (1 − lr · wd) · θ − η · m
   ```

   The effective per-step displacement is approximately `lr · ‖θ‖`, so
   `lr ≈ 1e-3` implements the "per-step change ≈ 0.1% of weights" heuristic.
   Decoupled weight decay counteracts the radial drift of `‖θ‖` that LARS
   induces under random-direction updates — without it, `‖θ‖` grows
   unboundedly and scale-sensitive losses (e.g. cross-entropy) stop reflecting
   accuracy.

## Hyperparameters

| Name                        | Default | Notes |
|-----------------------------|---------|-------|
| `lr`                        | `1e-3`  | Effective per-step displacement as a fraction of `‖θ‖`. |
| `beta`                      | `0.9`   | Momentum decay on the gradient estimate. |
| `weight_decay`              | `0.01`  | Decoupled shrink `θ ← (1 − lr·wd)·θ`. Set `0.0` on tasks where non-zero weights are essential (e.g. XOR with tiny models). |
| `num_perturbations`         | `8`     | Samples averaged per step. On CUDA, also the number of concurrent streams. |
| `perturbation_chunk_size`   | `None`  | CPU path only. Micro-batch size for the vmap forward (caps peak VRAM). `None` ⇒ one chunk. Ignored on CUDA. |
| `use_cuda_graph`            | `True`  | CUDA path only. Capture the full step into a `CUDAGraph`. Disable when composing with `torch.compile(mode="reduce-overhead")`. |

`lr`, `beta`, and `weight_decay` are per-parameter-group and can be overridden
via the standard PyTorch `param_groups` mechanism. The remaining flags are
optimizer-level.

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
  convergence stalls — cost scales linearly on CPU, near-free on CUDA up to
  the stream-parallelism limit of the device.
- If reported loss is high but accuracy looks fine, you are probably seeing
  `‖θ‖` drift. Increase `weight_decay`.

### Performance & VRAM optimization

- **CUDA stream + graph path (automatic):** all `num_perturbations` samples run
  concurrently on dedicated streams; the entire step is captured once into a
  `torch.cuda.CUDAGraph` and replayed every iteration. Per-stream noise
  buffers, the gradient accumulator, and the loss scalar are pre-allocated
  and filled in place — no per-step allocator traffic. Graphs rebuild
  transparently when input shapes or the parameter set change.
- **Constant input shapes help graph reuse.** On CUDA, pass a stable batch
  shape (e.g. `DataLoader(..., drop_last=True)`) to avoid re-capture on the
  trailing batch.
- **CPU vmap path (automatic):** plus / minus vmap forwards are sequenced with
  eager release so only one side's batched parameter tuple is live at a time.
  Per-parameter gradient reduction uses a matmul over the flattened
  perturbation axis, avoiding a `[K, *p.shape]` intermediate allocation.
- **Cap peak VRAM with `perturbation_chunk_size`** on the CPU path when `K` is
  large — activation memory scales with the chunk size, not `K`.
- **Multi-tensor `_foreach` momentum/update:** parameter-level bookkeeping
  uses PyTorch's C++ multi-tensor kernels, avoiding per-parameter Python
  overhead on both paths.
- **`torch.compile()` is your friend.** Wrapping the model fuses the forward
  into a single kernel, dramatically speeding up large `num_perturbations`
  runs on CPU. On CUDA, keep `use_cuda_graph=True` with plain eager `model`;
  combine with `torch.compile(mode="reduce-overhead")` only by passing
  `use_cuda_graph=False` so the two graph managers don't conflict.
- **Native Automatic Mixed Precision (AMP):** since no autograd graph is kept,
  running forward passes under `torch.autocast` in `fp16` / `bf16` halves the
  memory footprint and unlocks TensorCore acceleration.

## Running the examples

```bash
python examples/xor.py
python examples/xor.py --device cuda
python examples/mnist.py --device cuda --compile --num-perturbations 10000
```

## License

MIT — see [LICENSE](LICENSE).

## References

- Salimans et al. 2017, *Evolution Strategies as a Scalable Alternative to
  Reinforcement Learning.* <https://arxiv.org/abs/1703.03864>
- You et al. 2017, *Large Batch Training of Convolutional Networks* (LARS).
  <https://arxiv.org/abs/1708.03888>
- Loshchilov & Hutter 2019, *Decoupled Weight Decay Regularization.*
  <https://arxiv.org/abs/1711.05101>
