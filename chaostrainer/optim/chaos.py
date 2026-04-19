"""Chaos: gradient-free PyTorch optimizer.

Zero-order optimization via Evolution-Strategy style antithetic stochastic
perturbation combined with LARS-style global trust-ratio adaptive step sizing.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn
from torch.func import functional_call, vmap
from torch.optim.optimizer import Optimizer

__all__ = ["Chaos"]


_PERTURBATION_STD: float = 1e-3  # default perturbation scale

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
    autograd-free sanity baseline. It runs model passes within
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
        perturbation_chunk_size: if set, evaluates perturbations in chunks of
            this size instead of a single vmap over all ``num_perturbations``.
            Caps peak VRAM (activations scale with the chunk size, not
            ``num_perturbations``) while keeping vmap amortization. ``None``
            means one chunk of size ``num_perturbations`` (no chunking).
            Default: ``None``.
        perturbation_std: standard deviation ``ε`` of the Gaussian perturbation
            ``δ ~ N(0, ε² I)``. The central-difference estimator's variance is
            independent of this value and its bias vanishes as ``O(ε²)``, so
            ``1e-3`` works well for fp32 and AMP training. Decrease toward
            ``1e-4`` for fp16 to stay above the FP noise floor; increase toward
            ``1e-2`` for very noisy loss surfaces. Must be positive.
            Default: ``1e-3``.

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from chaostrainer import Chaos
        >>> model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
        >>> optimizer = Chaos(model.parameters(), lr=1e-2)
        >>> X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        >>> Y = torch.tensor([[0.], [1.], [1.], [0.]])
        >>> loss_fn = nn.MSELoss()
        >>> for _ in range(2000):
        ...     loss = optimizer.step(model, loss_fn, X, Y)

    Note:
        This optimizer **requires** a module and a callable ``criterion`` that
        takes the model output (and any additional args/kwargs passed to
        :meth:`step`) and returns a scalar loss. ``loss.backward()`` is never
        called; Chaos ignores autograd entirely. :meth:`step` returns the mean
        of :math:`L(\theta+\delta_k)` across the ``num_perturbations`` samples
        used in the step.

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
        perturbation_chunk_size: int | None = None,
        perturbation_std: float = _PERTURBATION_STD,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta: {beta} (must be in [0, 1))")
        if num_perturbations < 1:
            raise ValueError(f"Invalid num_perturbations: {num_perturbations}")
        if perturbation_chunk_size is not None and perturbation_chunk_size < 1:
            raise ValueError(
                f"Invalid perturbation_chunk_size: {perturbation_chunk_size}"
            )
        if perturbation_std <= 0.0:
            raise ValueError(f"Invalid perturbation_std: {perturbation_std} (must be > 0)")

        defaults = dict(
            lr=lr,
            beta=beta,
            num_perturbations=int(num_perturbations),
            perturbation_chunk_size=(
                int(perturbation_chunk_size)
                if perturbation_chunk_size is not None
                else None
            ),
            perturbation_std=float(perturbation_std),
        )
        super().__init__(params, defaults)
        self.num_perturbations = int(num_perturbations)
        self.perturbation_chunk_size = (
            int(perturbation_chunk_size)
            if perturbation_chunk_size is not None
            else None
        )
        self.perturbation_std = float(perturbation_std)

    @torch.no_grad()
    def step(self, model: nn.Module, criterion: Callable[..., Tensor], *args, **kwargs) -> Tensor:  # type: ignore[override]
        """Perform a single optimization step using vectorized map (vmap).

        Only parameters registered in this optimizer's ``param_groups`` are
        perturbed and updated; any other ``requires_grad=True`` parameters on
        the model are held at their current values during the forward pass.

        Args:
            model: PyTorch module whose parameters are being optimized.
            criterion: A callable computing the scalar loss from model outputs.
            *args: Positional arguments passed to the model and criterion.
            **kwargs: Keyword arguments passed to the criterion.

        Returns:
            Mean of ``L(θ+δ)`` across the perturbation samples used in the step.
        """
        # Build the inverse map from Parameter -> name so we can drive the
        # forward pass exclusively from param_groups (not named_parameters).
        param_to_name = {p: name for name, p in model.named_parameters()}

        optim_params: list[Tensor] = []
        optim_names: list[str] = []
        group_slices: list[tuple[int, int]] = []
        for group in self.param_groups:
            start = len(optim_params)
            for p in group["params"]:
                if p is None or not p.requires_grad:
                    continue
                name = param_to_name.get(p)
                if name is None:
                    raise ValueError(
                        "Chaos optimizer received a parameter that is not "
                        "registered on the supplied model. Ensure the optimizer "
                        "was constructed from model.parameters()."
                    )
                optim_params.append(p)
                optim_names.append(name)
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            group_slices.append((start, len(optim_params)))

        if not optim_params:
            return torch.zeros((), dtype=torch.float32)

        device = optim_params[0].device
        eps_std = self.perturbation_std
        inv_2eps_sq = 1.0 / (2.0 * eps_std * eps_std)

        K = self.num_perturbations
        M = self.perturbation_chunk_size if self.perturbation_chunk_size is not None else K

        def compute_loss(params_tuple):
            params_mapping = dict(zip(optim_names, params_tuple))
            out = functional_call(model, params_mapping, args[:1])
            return criterion(out, *args[1:], **kwargs)

        vmap_loss = vmap(compute_loss, in_dims=(0,))

        # Accumulate sum(noise_k · coef_k) across chunks, then divide by K at
        # the end. Using matmul over the flattened perturbation axis avoids
        # materializing the [k_c, *p.shape] broadcast product.
        grad_acc: list[Tensor] = [torch.zeros_like(p) for p in optim_params]
        loss_plus_sum = torch.zeros((), device=device, dtype=optim_params[0].dtype)

        offset = 0
        while offset < K:
            k_c = min(M, K - offset)
            noises_c: list[Tensor] = [
                torch.randn(k_c, *p.shape, device=device, dtype=p.dtype) * eps_std
                for p in optim_params
            ]

            # Plus pass — free the batched tuple before allocating minus so the
            # two forward-pass working sets do not overlap in VRAM.
            plus = tuple(p.unsqueeze(0) + n for p, n in zip(optim_params, noises_c))
            loss_plus = vmap_loss(plus)
            del plus

            minus = tuple(p.unsqueeze(0) - n for p, n in zip(optim_params, noises_c))
            loss_minus = vmap_loss(minus)
            del minus

            coef = (loss_plus - loss_minus) * inv_2eps_sq  # [k_c]
            loss_plus_sum = loss_plus_sum + loss_plus.sum()
            del loss_minus

            # grad_i += sum_k coef_k * noise_{k,i} via matmul on the flat axis.
            for g, noise in zip(grad_acc, noises_c):
                g.add_((coef @ noise.view(k_c, -1)).view(g.shape))

            del noises_c, coef, loss_plus
            offset += k_c

        torch._foreach_div_(grad_acc, float(K))
        mean_loss = loss_plus_sum / K

        # Momentum update per group; slices keep optim_params aligned with
        # param_groups order by construction.
        momentums = [self.state[p]["momentum"] for p in optim_params]
        for group, (start, end) in zip(self.param_groups, group_slices):
            if start == end:
                continue
            group_mems = momentums[start:end]
            group_grads = grad_acc[start:end]
            torch._foreach_mul_(group_mems, group["beta"])
            torch._foreach_add_(group_mems, group_grads)

        # Global trust-ratio norms (fp32 accumulation for mixed-precision safety).
        p_norms = torch._foreach_norm(optim_params, 2.0)
        m_norms = torch._foreach_norm(momentums, 2.0)
        weight_sq = torch.stack([n.to(torch.float32) for n in p_norms]).pow(2).sum()
        moment_sq = torch.stack([n.to(torch.float32) for n in m_norms]).pow(2).sum()
        weight_norm = torch.sqrt(weight_sq + _NORM_FLOOR)
        moment_norm = torch.sqrt(moment_sq + _NORM_FLOOR)
        ratio = weight_norm / moment_norm

        # Per-group step without forcing a GPU→CPU sync: multiply the momentum
        # list by the (tensor) scale once and _foreach_add_ the result.
        for group, (start, end) in zip(self.param_groups, group_slices):
            if start == end:
                continue
            group_params = optim_params[start:end]
            group_mems = momentums[start:end]
            scale = (-group["lr"] * ratio).to(group_mems[0].dtype)
            scaled = torch._foreach_mul(group_mems, scale)
            torch._foreach_add_(group_params, scaled)

        return mean_loss
