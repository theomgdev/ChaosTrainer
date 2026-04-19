"""ChaosStream: experimental CUDA-stream-parallel Chaos optimizer.

WARNING — EXPERIMENTAL
    This variant replaces the vmap-batched perturbation evaluation in
    :class:`chaostrainer.Chaos` with a Python loop over ``num_perturbations``,
    each iteration dispatched on its own ``torch.cuda.Stream``. All streams
    mutate the *same* parameter tensors in-place and accumulate into the
    *same* gradient buffer, so **the algorithm is racy by design**.

    This is a research variant exploring a past empirical observation:
    lock-free concurrent writes on shared parameters behave somewhat like
    asynchronous SGD (Hogwild!) or evolutionary crossover — consistent search
    directions reinforce each other through statistical overwrite, while
    inconsistent contributions wash out. It is not a well-posed Evolution-
    Strategy estimator and is not a drop-in for the stable :class:`Chaos`.

    - Non-deterministic, even with seeded RNG (stream interleaving is uncontrolled).
    - CUDA only — the race dynamics rely on concurrent stream execution.
    - Peak VRAM does not scale with ``num_perturbations × activation_size`` as
      in the vmap path; it scales with ``num_perturbations × parameter_size``
      (per-stream noise) plus a single forward's activations per live stream.

Use at your own risk; expect surprises.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

__all__ = ["ChaosStream"]


_PERTURBATION_STD: float = 1e-3
_NORM_FLOOR: float = 1e-8


class ChaosStream(Optimizer):
    r"""Stream-parallel, race-condition-tolerant variant of :class:`Chaos`.

    Each of the ``num_perturbations`` perturbations is evaluated on its own
    ``torch.cuda.Stream``. Streams concurrently:

    1. Write ``θ += δ_k`` in place on the shared parameter tensors.
    2. Run a forward pass to obtain ``L(θ+δ_k)``.
    3. Write ``θ -= 2·δ_k`` in place.
    4. Run a forward pass to obtain ``L(θ-δ_k)``.
    5. Write ``θ += δ_k`` in place (nominal restore).
    6. Accumulate ``(L+ - L-) · δ_k / (2 ε²)`` into the shared gradient buffer.

    Because steps 1, 3, 5 and the forward pass happen without synchronization
    across streams, the value of ``θ`` seen by any given forward is a random
    superposition of several streams' perturbations. The gradient accumulator
    is also written concurrently. Both effects are intentional: consistent
    directions statistically dominate the racy average, producing
    crossover-like dynamics. See module-level docstring for the full caveat.

    Args:
        params: iterable of parameters or dicts defining parameter groups.
        lr: scalar applied on top of the weight-to-momentum norm ratio.
        beta: momentum decay on the gradient estimate. Default: ``0.9``.
        num_perturbations: number of concurrent streams. Default: ``8``.

    Note:
        Requires CUDA. Raises ``RuntimeError`` on CPU/MPS devices.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta: float = 0.9,
        *,
        num_perturbations: int = 8,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta: {beta} (must be in [0, 1))")
        if num_perturbations < 1:
            raise ValueError(f"Invalid num_perturbations: {num_perturbations}")

        defaults = dict(lr=lr, beta=beta, num_perturbations=int(num_perturbations))
        super().__init__(params, defaults)
        self.num_perturbations = int(num_perturbations)
        self._streams: list[torch.cuda.Stream] | None = None

    @torch.no_grad()
    def step(self, model: nn.Module, criterion: Callable[..., Tensor], *args, **kwargs) -> Tensor:  # type: ignore[override]
        """Perform a single racy stream-parallel optimization step.

        Only parameters registered in ``param_groups`` are perturbed and
        updated. The set of perturbed parameters must live on a single CUDA
        device.
        """
        param_to_name = {p: name for name, p in model.named_parameters()}

        optim_params: list[Tensor] = []
        group_slices: list[tuple[int, int]] = []
        for group in self.param_groups:
            start = len(optim_params)
            for p in group["params"]:
                if p is None or not p.requires_grad:
                    continue
                if p not in param_to_name:
                    raise ValueError(
                        "ChaosStream received a parameter that is not "
                        "registered on the supplied model."
                    )
                optim_params.append(p)
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            group_slices.append((start, len(optim_params)))

        if not optim_params:
            return torch.zeros((), dtype=torch.float32)

        device = optim_params[0].device
        if device.type != "cuda":
            raise RuntimeError(
                "ChaosStream requires CUDA parameters; its race-condition "
                "dynamics rely on concurrent CUDA stream execution."
            )

        eps = _PERTURBATION_STD
        inv_2eps_sq = 1.0 / (2.0 * eps * eps)
        K = self.num_perturbations

        # Lazy-allocate the stream pool so we can recover gracefully if the
        # optimizer was moved to a fresh device mid-training.
        if (
            self._streams is None
            or len(self._streams) != K
            or self._streams[0].device != device
        ):
            self._streams = [torch.cuda.Stream(device=device) for _ in range(K)]

        grad_acc: list[Tensor] = [torch.zeros_like(p) for p in optim_params]
        loss_plus_acc = torch.zeros((), device=device, dtype=optim_params[0].dtype)

        current_stream = torch.cuda.current_stream(device)
        # Each worker must see the most recent state produced on the current
        # stream (e.g. data.to(device) copies, prior optimizer updates).
        for s in self._streams:
            s.wait_stream(current_stream)

        fwd_arg = args[:1]
        extra_args = args[1:]

        for stream in self._streams:
            with torch.cuda.stream(stream):
                noises: list[Tensor] = [
                    torch.randn_like(p).mul_(eps) for p in optim_params
                ]

                # θ += δ — RACED in-place on shared parameters.
                torch._foreach_add_(optim_params, noises)
                out = model(*fwd_arg)
                loss_plus = criterion(out, *extra_args, **kwargs)

                # θ -= 2δ
                torch._foreach_add_(optim_params, noises, alpha=-2.0)
                out = model(*fwd_arg)
                loss_minus = criterion(out, *extra_args, **kwargs)

                # θ += δ (nominal restore — other streams may have mutated θ
                # arbitrarily in between; this is the point of the experiment).
                torch._foreach_add_(optim_params, noises)

                coef = (loss_plus - loss_minus) * inv_2eps_sq

                # grad += coef * noise, fused via addcmul_ (no temporary, no sync).
                # Accumulator is also shared across streams → RACED adds.
                for g, n in zip(grad_acc, noises):
                    g.addcmul_(n, coef.expand_as(n))

                loss_plus_acc.add_(loss_plus)

        # Publish the worker streams' writes back to the default stream before
        # the momentum / LARS update reads them.
        for s in self._streams:
            current_stream.wait_stream(s)

        torch._foreach_div_(grad_acc, float(K))
        mean_loss = loss_plus_acc / K

        momentums = [self.state[p]["momentum"] for p in optim_params]
        for group, (start, end) in zip(self.param_groups, group_slices):
            if start == end:
                continue
            group_mems = momentums[start:end]
            group_grads = grad_acc[start:end]
            torch._foreach_mul_(group_mems, group["beta"])
            torch._foreach_add_(group_mems, group_grads)

        p_norms = torch._foreach_norm(optim_params, 2.0)
        m_norms = torch._foreach_norm(momentums, 2.0)
        weight_sq = torch.stack([n.to(torch.float32) for n in p_norms]).pow(2).sum()
        moment_sq = torch.stack([n.to(torch.float32) for n in m_norms]).pow(2).sum()
        weight_norm = torch.sqrt(weight_sq + _NORM_FLOOR)
        moment_norm = torch.sqrt(moment_sq + _NORM_FLOOR)
        ratio = weight_norm / moment_norm

        for group, (start, end) in zip(self.param_groups, group_slices):
            if start == end:
                continue
            group_params = optim_params[start:end]
            group_mems = momentums[start:end]
            scale = (-group["lr"] * ratio).to(group_mems[0].dtype)
            scaled = torch._foreach_mul(group_mems, scale)
            torch._foreach_add_(group_params, scaled)

        return mean_loss
