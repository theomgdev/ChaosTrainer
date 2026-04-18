"""Chaos: gradient-free PyTorch optimizer.

Zero-order optimization via Evolution-Strategy style antithetic stochastic
perturbation combined with LARS-style global trust-ratio adaptive step sizing.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

__all__ = ["Chaos"]


# Perturbation scale for the antithetic finite-difference estimator.
# Fixed at ~sqrt(fp32 machine epsilon); variance of the estimator is
# independent of this value and bias vanishes as O(ε²), so exposing it
# as a hyperparameter adds no useful knob for fp32/AMP training.
_PERTURBATION_STD: float = 1e-3

# Numerical-stability floor inside the global norms. Only matters on the very
# first step, before momentum is populated, where ‖m‖ is exactly zero. Outside
# that cold-start case the value is irrelevant, so there is no knob to tune.
_NORM_FLOOR: float = 1e-8


class Chaos(Optimizer):
    r"""Gradient-free optimizer using antithetic ES perturbations with LARS trust ratio.

    Each :meth:`step` samples a random perturbation :math:`\delta\sim\mathcal{N}(0,
    \varepsilon^2 I)` for every parameter and evaluates the loss at both
    :math:`\theta+\delta` and :math:`\theta-\delta`. The central-difference
    Evolution-Strategy estimator

    .. math::
        \hat{g} \;=\; \frac{\bigl(L(\theta+\delta) - L(\theta-\delta)\bigr)\,\delta}{2\varepsilon^2}

    then feeds a momentum buffer, and the actual step is rescaled by the
    global ratio of the weight norm to the momentum norm, in the spirit of LARS

    .. math::
        \eta \;=\; \text{lr} \cdot \frac{\lVert \theta \rVert}{\lVert m \rVert}

    so the update magnitude tracks the parameter magnitude, giving robust
    behavior across layers of very different scale.

    Because no autograd graph is required, this optimizer is useful when:
    gradients are unavailable (non-differentiable losses, black-box simulators,
    discrete ops), gradients are prohibitively expensive, or you want an
    autograd-free sanity baseline. It wraps each closure call in
    :func:`torch.no_grad`, saving activations memory relative to first-order
    methods.

    Args:
        params: iterable of parameters or dicts defining parameter groups.
        lr: scalar applied on top of the weight-to-momentum norm ratio. The
            effective per-step displacement is approximately ``lr · ‖θ‖``, so
            values near ``1e-3`` match the "per-step change ≈ 0.1% of weights"
            heuristic. Default: ``1e-3``.
        beta: momentum decay on the gradient estimate. Default: ``0.9``.
        num_perturbations: number of perturbation samples averaged per step.
            More samples reduce variance linearly at proportional cost.
            Default: ``1``.

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from chaostrainer import Chaos
        >>> model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
        >>> optimizer = Chaos(model.parameters(), lr=1.0)
        >>> X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        >>> Y = torch.tensor([[0.], [1.], [1.], [0.]])
        >>> loss_fn = nn.MSELoss()
        >>> def closure():
        ...     return loss_fn(torch.sigmoid(model(X)), Y)
        >>> for _ in range(2000):
        ...     loss = optimizer.step(closure)

    Note:
        This optimizer **requires** a closure that returns the scalar loss. The
        closure does *not* need to call ``loss.backward()``; Chaos ignores any
        gradients. :meth:`step` returns :math:`L(\theta+\delta)` of the final
        perturbation sample, consistent with the torch optimizer convention of
        returning an objective value observed during the step.

    References:
        Salimans et al. 2017, *Evolution Strategies as a Scalable Alternative
        to Reinforcement Learning*. https://arxiv.org/abs/1703.03864

        You et al. 2017, *Large Batch Training of Convolutional Networks* (LARS).
        https://arxiv.org/abs/1708.03888
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta: float = 0.9,
        *,
        num_perturbations: int = 1,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta: {beta} (must be in [0, 1))")
        if num_perturbations < 1:
            raise ValueError(f"Invalid num_perturbations: {num_perturbations}")

        defaults = dict(
            lr=lr,
            beta=beta,
        )
        super().__init__(params, defaults)
        self.num_perturbations = int(num_perturbations)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Tensor]] = None) -> Tensor:  # type: ignore[override]
        """Perform a single optimization step.

        Args:
            closure: zero-argument callable that re-evaluates the model and
                returns the scalar loss tensor. Required.
        """
        if closure is None:
            raise RuntimeError(
                "Chaos requires a closure that recomputes and returns the loss."
            )

        # Collect trainable params and lazy-initialize momentum state.
        params: list[Tensor] = []
        for group in self.param_groups:
            for p in group["params"]:
                if p is None or not p.requires_grad:
                    continue
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                params.append(p)

        if not params:
            return torch.zeros((), dtype=torch.float32)

        device = params[0].device
        eps_std = _PERTURBATION_STD
        inv_2eps_sq = 1.0 / (2.0 * eps_std * eps_std)

        # Gradient-estimate accumulator (same shape/dtype/device as each param).
        grad_acc: list[Tensor] = [torch.zeros_like(p) for p in params]
        
        # Pre-allocate perturbation buffers to prevent VRAM fragmentation
        # and allocation overhead during the num_perturbations loop.
        deltas: list[Tensor] = [torch.empty_like(p) for p in params]

        last_loss: Optional[Tensor] = None

        for _ in range(self.num_perturbations):
            # Sample δ ~ N(0, ε² I) in-place.
            for d in deltas:
                d.normal_(mean=0.0, std=eps_std)

            # Evaluate L(θ + δ).
            torch._foreach_add_(params, deltas)
            loss_plus = _as_scalar(closure()).detach()

            # Move θ + δ → θ − δ (subtract 2δ), then evaluate L(θ − δ).
            torch._foreach_add_(params, deltas, alpha=-2.0)
            loss_minus = _as_scalar(closure()).detach()

            # Restore θ.
            torch._foreach_add_(params, deltas)

            # Central-difference ES estimator: ĝ = (L+ − L−) · δ / (2ε²)
            coef = float((loss_plus - loss_minus) * inv_2eps_sq)
            torch._foreach_add_(grad_acc, deltas, alpha=coef)

            last_loss = loss_plus

        if self.num_perturbations > 1:
            inv_n = 1.0 / float(self.num_perturbations)
            for acc in grad_acc:
                acc.mul_(inv_n)

        # Momentum update (per-group β).
        idx = 0
        for group in self.param_groups:
            beta = group["beta"]
            group_params = [p for p in group["params"] if p is not None and p.requires_grad]
            if not group_params:
                continue
            
            group_mems = [self.state[p]["momentum"] for p in group_params]
            group_grads = grad_acc[idx : idx + len(group_params)]
            
            # Multi-tensor momentum update
            torch._foreach_mul_(group_mems, beta)
            torch._foreach_add_(group_mems, group_grads)
            idx += len(group_params)

        # Global trust-ratio norms (in fp32 to avoid low-precision overflow).
        momentums = [self.state[p]["momentum"] for p in params]
        
        # Use highly-optimized C++ multi-tensor operations for L2 norms
        p_norms = torch._foreach_norm(params, 2.0)
        m_norms = torch._foreach_norm(momentums, 2.0)

        # Stack individual norms, cast to fp32, and compute the global squared norm
        weight_sq = torch.stack([n.to(torch.float32) for n in p_norms]).pow(2).sum()
        moment_sq = torch.stack([n.to(torch.float32) for n in m_norms]).pow(2).sum()

        weight_norm = torch.sqrt(weight_sq + _NORM_FLOOR)
        moment_norm = torch.sqrt(moment_sq + _NORM_FLOOR)
        global_ratio = float((weight_norm / moment_norm).detach())

        # Apply per-group update using group-local lr.
        for group in self.param_groups:
            lr = group["lr"]
            step_scale = -lr * global_ratio
            group_params = [p for p in group["params"] if p is not None and p.requires_grad]
            if not group_params:
                continue
                
            group_mems = [self.state[p]["momentum"] for p in group_params]
            
            # Multi-tensor weight update
            torch._foreach_add_(group_params, group_mems, alpha=step_scale)

        return last_loss if last_loss is not None else torch.zeros((), device=device)

def _as_scalar(value) -> Tensor:
    """Coerce a closure return value into a 0-dim scalar tensor."""
    if isinstance(value, Tensor):
        if value.ndim != 0:
            value = value.mean()
        return value
    return torch.as_tensor(float(value))
