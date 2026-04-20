"""Chaos: gradient-free PyTorch optimizer.

Zero-order optimization via Evolution-Strategy style antithetic stochastic
perturbation combined with LARS-style global trust-ratio adaptive step sizing.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor, nn
from torch.func import functional_call, vmap
from torch.optim.optimizer import Optimizer

__all__ = ["Chaos"]

# Numerical-stability floor inside the global norms. Only matters on the very
# first step, before momentum is populated, where ‖m‖ is exactly zero. Outside
# that cold-start case the value is irrelevant, so there is no knob to tune.
_NORM_FLOOR: float = 1e-8


def _centered_rank_coef(loss_plus: Tensor, loss_minus: Tensor) -> Tensor:
    """Antithetic ES coefficients via centered rank transform over all 2K losses.

    Ranks all 2K fitness values jointly so the coefficient magnitude is
    invariant to the loss scale and any monotonic transformation of the loss.
    Lower loss → lower centered rank → negative contribution, consistent with
    the minimization sign convention of the raw ``(L+ − L−)``
    estimator.
    """
    K = loss_plus.shape[0]
    all_losses = torch.cat([loss_plus, loss_minus])                    # [2K]
    raw_ranks = torch.argsort(torch.argsort(all_losses)).float()       # 0 = smallest
    centered = raw_ranks / (2 * K - 1) - 0.5                          # [−0.5, 0.5]
    return centered[:K] - centered[K:]


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
    :func:`torch.no_grad`, saving activation memory relative to first-order
    methods.

    **Dual-mode usage.** In addition to the standalone :meth:`step` (which runs
    ES and applies the LARS update internally), :meth:`estimate_grad` exposes
    the ES gradient estimate as a standard ``param.grad`` tensor, enabling
    pairing with any PyTorch optimizer for the update step — see
    :meth:`estimate_grad` for details and examples.

    Args:
        params: iterable of parameters or dicts defining parameter groups.
        lr: scalar applied on top of the weight-to-momentum norm ratio. The
            effective per-step displacement is approximately ``lr · ‖θ‖``, so
            values near ``1e-2`` match the "per-step change ≈ 1% of weights"
            heuristic. Default: ``1e-2``.
        beta: momentum decay on the gradient estimate. Default: ``0.9``.
        weight_decay: decoupled L2 regularization coefficient. After each ES
            step, parameters are shrunk as ``θ ← θ · (1 − lr · λ)``, in the
            spirit of AdamW. Applied per-group. Default: ``0.0``.
        num_perturbations: number of perturbation samples averaged per step.
            More samples reduce estimator variance linearly at proportional
            forward-pass cost. Lower to 1–2 only when compute is extremely
            tight; raise to 16–64 for high-variance objectives. Default: ``8``.
        perturbation_chunk_size: if set, evaluates perturbations in chunks of
            this size instead of a single vmap over all ``num_perturbations``.
            Caps peak activation VRAM (activations scale with the chunk size,
            not ``num_perturbations``) while keeping vmap amortization. ``None``
            means one chunk of size ``num_perturbations`` (no chunking).
            Note: when ``fitness_shaping`` or ``orthogonal_perturbations`` is
            enabled, all ``num_perturbations`` noise tensors are held in memory
            simultaneously regardless of this setting. Default: ``None``.
        perturbation_std: standard deviation ``ε`` of the Gaussian perturbation
            ``δ ~ N(0, ε² I)``. If ``None``, it is dynamically computed per parameter
            based on its scaling to adapt to fine-tuning. The central-difference
            estimator's variance is independent of this value and its bias vanishes
            as ``O(ε²)``. Must be positive if provided. Default: ``None``.
        grad_clip: if set, clips the global L2 norm of the ES gradient estimate
            to this value before the momentum update, preventing runaway steps
            on noisy or sparse objectives. Applied without a GPU→CPU sync.
            Must be positive. Default: ``None``.
        fitness_shaping: if ``True``, replaces raw loss differences with
            centered rank scores computed over all ``2 · num_perturbations``
            evaluations before forming the gradient estimate. Makes the optimizer
            invariant to monotonic transformations of the loss (e.g. shifted or
            scaled rewards) — valuable for RL-style or non-stationary objectives.
            Set to ``False`` only when reproducing results that used raw loss
            differences, or when ``num_perturbations`` is very small and the
            fixed-magnitude coefficient at K=1 is undesirable. Default: ``True``.
        orthogonal_perturbations: if ``True``, generates perturbation directions
            via QR orthogonalization (per parameter, in float32 for numerical
            stability), so the ``num_perturbations`` noise vectors span orthogonal
            subspaces. Reduces estimator variance compared to i.i.d. Gaussian at
            the same sample count. Falls back to i.i.d. when
            ``num_perturbations > param.numel()``. Set to ``False`` only when
            reproducing i.i.d.-Gaussian baselines or when the QR cost matters
            at very large ``num_perturbations``. Default: ``True``.

    Example — standalone ES training::

        >>> import torch
        >>> from torch import nn
        >>> from chaostrainer import Chaos
        >>> model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
        >>> optimizer = Chaos(model.parameters(), lr=1e-2)
        >>> X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        >>> Y = torch.tensor([[0.], [1.], [1.], [0.]])
        >>> loss_fn = nn.MSELoss()
        >>> for _ in range(3000):
        ...     loss = optimizer.step(model, loss_fn, X, Y)

    Note:
        :meth:`step` **requires** a module and a callable ``criterion`` that
        returns a scalar loss. ``loss.backward()`` is never called; Chaos ignores
        autograd entirely. :meth:`step` returns the mean of
        :math:`L(\theta+\delta_k)` across the ``num_perturbations`` samples used.

        Only ``lr``, ``beta``, and ``weight_decay`` can be overridden per
        param-group; all other hyperparameters are optimizer-wide and shared
        across all groups.

    References:
        Salimans et al. 2017, *Evolution Strategies as a Scalable Alternative
        to Reinforcement Learning*. https://arxiv.org/abs/1703.03864

        You et al. 2017, *Large Batch Training of Convolutional Networks* (LARS).
        https://arxiv.org/abs/1708.03888
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        beta: float = 0.9,
        weight_decay: float = 0.0,
        *,
        num_perturbations: int = 8,
        perturbation_chunk_size: int | None = None,
        perturbation_std: float | None = None,
        grad_clip: float | None = None,
        fitness_shaping: bool = True,
        orthogonal_perturbations: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta: {beta} (must be in [0, 1))")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if num_perturbations < 1:
            raise ValueError(f"Invalid num_perturbations: {num_perturbations}")
        if perturbation_chunk_size is not None and perturbation_chunk_size < 1:
            raise ValueError(
                f"Invalid perturbation_chunk_size: {perturbation_chunk_size}"
            )
        if perturbation_std is not None and perturbation_std <= 0.0:
            raise ValueError(f"Invalid perturbation_std: {perturbation_std} (must be > 0)")
        if grad_clip is not None and grad_clip <= 0.0:
            raise ValueError(f"Invalid grad_clip: {grad_clip} (must be > 0)")

        defaults = dict(
            lr=lr,
            beta=beta,
            weight_decay=float(weight_decay),
            num_perturbations=int(num_perturbations),
            perturbation_chunk_size=(
                int(perturbation_chunk_size)
                if perturbation_chunk_size is not None
                else None
            ),
            perturbation_std=float(perturbation_std) if perturbation_std is not None else None,
            grad_clip=float(grad_clip) if grad_clip is not None else None,
            fitness_shaping=bool(fitness_shaping),
            orthogonal_perturbations=bool(orthogonal_perturbations),
        )
        super().__init__(params, defaults)
        self.num_perturbations = int(num_perturbations)
        self.perturbation_chunk_size = (
            int(perturbation_chunk_size)
            if perturbation_chunk_size is not None
            else None
        )
        self.perturbation_std = float(perturbation_std) if perturbation_std is not None else None
        self.grad_clip = float(grad_clip) if grad_clip is not None else None
        self.fitness_shaping = bool(fitness_shaping)
        self.orthogonal_perturbations = bool(orthogonal_perturbations)

    def _generate_noises(
        self,
        optim_params: list[Tensor],
        K: int,
        device: torch.device,
        eps_stds: list[Tensor],
    ) -> list[Tensor]:
        """Generate K perturbation noise tensors, one per parameter.

        When ``orthogonal_perturbations`` is enabled, each parameter's K noise
        vectors are orthogonalized via QR in float32 for numerical stability,
        then cast back to the parameter dtype. Falls back to i.i.d. Gaussian
        when ``K > param.numel()``.

        Returns a list of tensors shaped ``[K, *param.shape]``.
        """
        if self.orthogonal_perturbations and K > 1:
            noises = []
            for p, eps_std in zip(optim_params, eps_stds):
                d = p.numel()
                Z = torch.randn(K, d, device=device, dtype=torch.float32)
                if K <= d:
                    Q, _ = torch.linalg.qr(Z.T)              # [d, K] orthonormal
                    noise_flat = Q.T.contiguous() * (eps_std * math.sqrt(d))
                else:
                    noise_flat = Z * eps_std
                noises.append(noise_flat.to(p.dtype).view(K, *p.shape))
            return noises
        return [
            torch.randn(K, *p.shape, device=device, dtype=p.dtype) * eps_std
            for p, eps_std in zip(optim_params, eps_stds)
        ]

    @torch.no_grad()
    def _run_es(
        self,
        model: nn.Module,
        criterion: Callable[..., Tensor],
        *args,
        **kwargs,
    ) -> tuple[list[Tensor], list[Tensor], list[tuple[int, int]], Tensor]:
        """Run ES forward passes and return the raw gradient accumulator.

        Collects all ``num_perturbations`` antithetic evaluations, applies
        fitness shaping and orthogonal noise as configured, and returns the
        per-parameter gradient estimates (already divided by K) together with
        the structural information needed by callers.

        Returns:
            ``(optim_params, grad_acc, group_slices, mean_loss)``

            - *optim_params*: flat list of optimized parameter tensors.
            - *grad_acc*: ES gradient estimate for each parameter, same order.
            - *group_slices*: ``(start, end)`` index pairs mapping each
              ``param_group`` into *optim_params*.
            - *mean_loss*: mean of ``L(θ+δ_k)`` across the K samples.
        """
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
                    state["momentum"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
            group_slices.append((start, len(optim_params)))

        if not optim_params:
            empty: list[Tensor] = []
            return empty, empty, [], torch.zeros((), dtype=torch.float32)

        device = optim_params[0].device
        K = self.num_perturbations
        M = self.perturbation_chunk_size if self.perturbation_chunk_size is not None else K

        eps_stds: list[Tensor] = []
        inv_2eps_sqs: list[Tensor] = []
        for p in optim_params:
            if self.perturbation_std is None:
                scale = p.norm() / math.sqrt(p.numel())
                eps = scale.clamp(min=1e-8) * 1e-2
            else:
                eps = torch.tensor(self.perturbation_std, device=device, dtype=p.dtype)
            eps_stds.append(eps)
            inv_2eps_sqs.append(1.0 / (2.0 * eps * eps))

        def compute_loss(params_tuple):
            params_mapping = dict(zip(optim_names, params_tuple))
            out = functional_call(model, params_mapping, args[:1])
            return criterion(out, *args[1:], **kwargs)

        vmap_loss = vmap(compute_loss, in_dims=(0,), randomness="different")
        grad_acc: list[Tensor] = [torch.zeros_like(p) for p in optim_params]

        if self.fitness_shaping or self.orthogonal_perturbations:
            # Two-phase approach: pre-generate all K noise tensors, collect all
            # losses across chunks, then form coefficients and accumulate the
            # gradient. Required because rank shaping needs all 2K loss values
            # before any coefficient can be computed, and orthogonal perturbations
            # need K jointly-generated directions per parameter.
            # Memory: O(K · |θ|) for noise storage in addition to activations.
            all_noises = self._generate_noises(optim_params, K, device, eps_stds)
            loss_dtype = optim_params[0].dtype
            all_loss_plus = torch.empty(K, device=device, dtype=loss_dtype)
            all_loss_minus = torch.empty(K, device=device, dtype=loss_dtype)

            offset = 0
            while offset < K:
                k_c = min(M, K - offset)
                noises_c = [n[offset : offset + k_c] for n in all_noises]
                plus = tuple(p.unsqueeze(0) + n for p, n in zip(optim_params, noises_c))
                all_loss_plus[offset : offset + k_c] = vmap_loss(plus)
                del plus
                minus = tuple(p.unsqueeze(0) - n for p, n in zip(optim_params, noises_c))
                all_loss_minus[offset : offset + k_c] = vmap_loss(minus)
                del minus
                offset += k_c

            coefs = (
                _centered_rank_coef(all_loss_plus, all_loss_minus)
                if self.fitness_shaping
                else (all_loss_plus - all_loss_minus)
            )
            for g, noise, inv_sq in zip(grad_acc, all_noises, inv_2eps_sqs):
                g.add_(((coefs * inv_sq) @ noise.view(K, -1)).view(g.shape))
            mean_loss = all_loss_plus.sum() / K

        else:
            # Standard single-phase chunked loop: noise is generated and freed
            # chunk-by-chunk so peak memory scales with chunk size, not K.
            loss_plus_sum = torch.zeros((), device=device, dtype=optim_params[0].dtype)
            offset = 0
            while offset < K:
                k_c = min(M, K - offset)
                noises_c: list[Tensor] = [
                    torch.randn(k_c, *p.shape, device=device, dtype=p.dtype) * eps_std
                    for p, eps_std in zip(optim_params, eps_stds)
                ]
                plus = tuple(p.unsqueeze(0) + n for p, n in zip(optim_params, noises_c))
                loss_plus = vmap_loss(plus)
                del plus
                minus = tuple(p.unsqueeze(0) - n for p, n in zip(optim_params, noises_c))
                loss_minus = vmap_loss(minus)
                del minus
                coef = (loss_plus - loss_minus)
                loss_plus_sum = loss_plus_sum + loss_plus.sum()
                del loss_minus
                for g, noise, inv_sq in zip(grad_acc, noises_c, inv_2eps_sqs):
                    g.add_(((coef * inv_sq) @ noise.view(k_c, -1)).view(g.shape))
                del noises_c, coef, loss_plus
                offset += k_c
            mean_loss = loss_plus_sum / K

        torch._foreach_div_(grad_acc, float(K))
        return optim_params, grad_acc, group_slices, mean_loss

    @torch.no_grad()
    def estimate_grad(
        self,
        model: nn.Module,
        criterion: Callable[..., Tensor],
        *args,
        **kwargs,
    ) -> Tensor:
        """Compute the ES gradient estimate and accumulate it into ``param.grad``.

        Runs the same vmap ES forward passes as :meth:`step` but writes the
        resulting per-parameter gradient estimate into ``param.grad``
        (accumulating, like ``loss.backward()``) instead of applying the LARS
        momentum update. Call ``optimizer.zero_grad()`` before this method when
        you want a fresh estimate rather than accumulation.

        This unlocks three usage patterns beyond the standalone :meth:`step`:

        **1. Pair with any standard optimizer (recommended for Lightning):**

        .. code-block:: python

            chaos = Chaos(model.parameters(), num_perturbations=16)
            adamw = torch.optim.AdamW(model.parameters(), lr=1e-2)

            for data, target in dataloader:
                adamw.zero_grad()
                loss = chaos.estimate_grad(model, loss_fn, data, target)
                adamw.step()

        **2. Gradient-free RL / black-box objectives (no backward needed):**

        .. code-block:: python

            chaos = Chaos(model.parameters())
            adamw = torch.optim.AdamW(model.parameters(), lr=1e-2)

            for state in env:
                adamw.zero_grad()
                mean_return = chaos.estimate_grad(model, rollout_fn, state)
                adamw.step()

        **3. Mix ES gradients with backprop gradients:**

        .. code-block:: python

            adamw.zero_grad()
            loss = chaos.estimate_grad(model, loss_fn, data, target)
            # compute your own loss outside no_grad for backward():
            loss_bp = loss_fn(model(data), target)
            loss_bp.backward()   # accumulates on top of the ES estimate
            adamw.step()

        Args:
            model: PyTorch module whose parameters are being optimized.
            criterion: callable returning a scalar loss from the model output.
            *args: forwarded to the model and criterion (same as :meth:`step`).
            **kwargs: forwarded to the criterion.

        Returns:
            Mean of ``L(θ+δ_k)`` across the K perturbation samples (a detached
            scalar tensor — not differentiable; use your own forward pass if
            you need ``loss.backward()``).
        """
        optim_params, grad_acc, _, mean_loss = self._run_es(
            model, criterion, *args, **kwargs
        )
        for p, g in zip(optim_params, grad_acc):
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.add_(g)
        return mean_loss

    @torch.no_grad()
    def step(self, model: nn.Module, criterion: Callable[..., Tensor], *args, **kwargs) -> Tensor:  # type: ignore[override]
        """Perform a single optimization step using the LARS-style update rule.

        Runs ES forward passes via :meth:`_run_es`, applies optional gradient
        clipping, updates the per-parameter momentum buffers, and takes a
        trust-ratio-rescaled step. Weight decay is applied per-group after the
        ES update.

        Only parameters registered in this optimizer's ``param_groups`` are
        perturbed and updated; any other ``requires_grad=True`` parameters on
        the model are held at their current values during the forward pass.

        Args:
            model: PyTorch module whose parameters are being optimized.
            criterion: callable computing the scalar loss from model outputs.
            *args: positional arguments forwarded to the model and criterion.
            **kwargs: keyword arguments forwarded to the criterion.

        Returns:
            Mean of ``L(θ+δ)`` across the perturbation samples used in the step.
        """
        optim_params, grad_acc, group_slices, mean_loss = self._run_es(
            model, criterion, *args, **kwargs
        )

        if not optim_params:
            return mean_loss

        # Gradient clipping: scale the ES gradient to at most grad_clip in L2
        # norm. Implemented as a tensor multiply to avoid a GPU→CPU sync.
        if self.grad_clip is not None:
            g_norms = torch._foreach_norm(grad_acc, 2.0)
            total_gnorm = torch.sqrt(
                torch.stack([n.to(torch.float32).pow(2) for n in g_norms]).sum()
            )
            clip_coef = (self.grad_clip / total_gnorm.clamp(min=1e-6)).clamp(max=1.0)
            torch._foreach_mul_(grad_acc, clip_coef.to(grad_acc[0].dtype))

        # Momentum update per group; slices keep optim_params aligned with
        # param_groups order by construction.
        momentums = [self.state[p]["momentum"] for p in optim_params]
        for group, (start, end) in zip(self.param_groups, group_slices):
            if start == end:
                continue
            torch._foreach_mul_(momentums[start:end], group["beta"])
            torch._foreach_add_(momentums[start:end], grad_acc[start:end])

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
        # Decoupled weight decay is applied after the ES step.
        for group, (start, end) in zip(self.param_groups, group_slices):
            if start == end:
                continue
            group_params = optim_params[start:end]
            group_mems = momentums[start:end]
            scale = (-group["lr"] * ratio).to(group_mems[0].dtype)
            scaled = torch._foreach_mul(group_mems, scale)
            torch._foreach_add_(group_params, scaled)
            if group["weight_decay"] > 0.0:
                torch._foreach_mul_(
                    group_params, 1.0 - group["lr"] * group["weight_decay"]
                )

        return mean_loss
