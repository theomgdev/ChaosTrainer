"""Chaos: gradient-free PyTorch optimizer.

Zero-order optimization via antithetic Evolution-Strategy perturbations with
LARS-style global trust-ratio step sizing and decoupled weight decay.

The execution path is selected by the device of the optimized parameters:

* **CUDA** — each of ``num_perturbations`` perturbations is dispatched on its
  own ``torch.cuda.Stream``. All streams mutate the shared parameters in
  place and accumulate into the shared gradient buffer without
  synchronization. The full multi-stream step is captured into a
  ``torch.cuda.CUDAGraph`` on the first call and replayed thereafter,
  collapsing thousands of kernel launches into a single replay. The
  race-by-design dynamic behaves like lock-free evolutionary crossover:
  consistent search directions statistically dominate the racy average.
  Trajectories are non-deterministic even with a fixed seed.

* **CPU / other** — perturbations are evaluated through a single vectorized
  ``torch.func.vmap`` over the perturbation axis. Deterministic under a
  fixed seed.

Both paths share the same public API, hyperparameters, and return value
(the clean baseline loss ``L(θ)`` taken before the step's update).
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn
from torch.func import functional_call, vmap
from torch.optim.optimizer import Optimizer

__all__ = ["Chaos"]


_PERTURBATION_STD: float = 1e-3
_NORM_FLOOR: float = 1e-8
_WARMUP_ITERS: int = 3


class Chaos(Optimizer):
    r"""Gradient-free optimizer using antithetic ES perturbations, LARS trust
    ratio, and decoupled weight decay.

    Each :meth:`step` samples per-parameter noise
    :math:`\delta \sim \mathcal{N}(0, \varepsilon^2 I)` and forms the
    antithetic central-difference Evolution-Strategy estimator

    .. math::
        \hat{g} \;=\; \mathrm{mean}_k \frac{\bigl(L(\theta+\delta_k)
                        - L(\theta-\delta_k)\bigr)\,\delta_k}{2\varepsilon^2}

    which feeds a momentum buffer. The update is rescaled globally by the
    weight-to-momentum norm ratio (LARS), then shrunk by a decoupled weight
    decay term:

    .. math::
        m \;&\leftarrow\; \beta\,m + \hat{g} \\
        \eta \;&=\; \mathrm{lr} \cdot \frac{\lVert\theta\rVert}{\lVert m\rVert} \\
        \theta \;&\leftarrow\; (1 - \mathrm{lr}\cdot\mathrm{wd})\,\theta - \eta\,m

    Args:
        params: iterable of parameters or dicts defining parameter groups.
        lr: scalar applied on top of the trust-ratio rescaling. Effective
            per-step displacement is approximately ``lr · ‖θ‖``. Default:
            ``1e-3``.
        beta: momentum decay. Default: ``0.9``.
        weight_decay: decoupled shrink ``θ ← (1 − lr·wd)·θ``. Counteracts the
            radial drift of ``‖θ‖`` under LARS-driven random-direction updates
            and sets the equilibrium weight scale. Default: ``0.01`` (AdamW
            convention). Raise toward ``0.1`` for LLM-scale training where
            explicit regularization matters; lower toward ``1e-4`` if the
            task needs large logit margins (e.g. classification with
            cross-entropy).
        num_perturbations: perturbation samples per step. On CUDA this is the
            number of concurrent streams; on CPU it is the size of the vmap
            batch. Default: ``8``.
        perturbation_chunk_size: CPU path only. If set, the vmap evaluates
            ``num_perturbations`` in sequential chunks of this size, capping
            peak activation memory at chunk-size cost. Ignored on CUDA.
            Default: ``None`` (one chunk).
        use_cuda_graph: CUDA path only. Capture the full multi-stream step
            into a ``torch.cuda.CUDAGraph`` on first call and replay
            thereafter. Set to ``False`` when composing with a wrapper that
            manages its own CUDA graphs (e.g. ``torch.compile(mode=
            "reduce-overhead")``). Default: ``True``.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta: float = 0.9,
        weight_decay: float = 0.01,
        *,
        num_perturbations: int = 8,
        perturbation_chunk_size: int | None = None,
        use_cuda_graph: bool = True,
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
        )
        super().__init__(params, defaults)
        self.num_perturbations = int(num_perturbations)
        self.perturbation_chunk_size = (
            int(perturbation_chunk_size)
            if perturbation_chunk_size is not None
            else None
        )
        self.use_cuda_graph = bool(use_cuda_graph)

        # CUDA-path state — lazily allocated on first CUDA step.
        self._streams: list[torch.cuda.Stream] | None = None
        self._capture_stream: torch.cuda.Stream | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_inputs: list | None = None
        self._static_loss: Tensor | None = None
        self._noise_buffers: list[list[Tensor]] | None = None
        self._grad_acc: list[Tensor] | None = None
        self._captured_shapes: tuple | None = None
        self._captured_param_ids: tuple | None = None

    # ------------------------------------------------------------------ step

    @torch.no_grad()
    def step(self, model: nn.Module, criterion: Callable[..., Tensor], *args, **kwargs) -> Tensor:  # type: ignore[override]
        """Perform one optimization step.

        Only parameters registered in ``param_groups`` are perturbed and
        updated. The registered parameters must share a single device; the
        execution path (stream-parallel + CUDA graph, or vmap) is chosen
        from that device.

        Returns the clean loss ``L(θ)`` evaluated at the parameters as they
        stood at the start of the step.
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
                        "Chaos received a parameter that is not registered on "
                        "the supplied model."
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
            return torch.zeros((), dtype=torch.float32)

        device = optim_params[0].device
        if device.type == "cuda":
            return self._step_cuda(
                optim_params, group_slices, model, criterion, args, kwargs
            )
        return self._step_vmap(
            optim_params, optim_names, group_slices, model, criterion, args, kwargs
        )

    # ---------------------------------------------------------------- vmap

    def _step_vmap(
        self,
        optim_params: list[Tensor],
        optim_names: list[str],
        group_slices: list[tuple[int, int]],
        model: nn.Module,
        criterion: Callable[..., Tensor],
        args: tuple,
        kwargs: dict,
    ) -> Tensor:
        device = optim_params[0].device
        eps = _PERTURBATION_STD
        inv_2eps_sq = 1.0 / (2.0 * eps * eps)
        K = self.num_perturbations
        M = self.perturbation_chunk_size if self.perturbation_chunk_size is not None else K

        def compute_loss(params_tuple):
            params_mapping = dict(zip(optim_names, params_tuple))
            out = functional_call(model, params_mapping, args[:1])
            return criterion(out, *args[1:], **kwargs)

        vmap_loss = vmap(compute_loss, in_dims=(0,))

        grad_acc: list[Tensor] = [torch.zeros_like(p) for p in optim_params]

        offset = 0
        while offset < K:
            k_c = min(M, K - offset)
            noises_c: list[Tensor] = [
                torch.randn(k_c, *p.shape, device=device, dtype=p.dtype) * eps
                for p in optim_params
            ]

            plus = tuple(p.unsqueeze(0) + n for p, n in zip(optim_params, noises_c))
            loss_plus = vmap_loss(plus)
            del plus

            minus = tuple(p.unsqueeze(0) - n for p, n in zip(optim_params, noises_c))
            loss_minus = vmap_loss(minus)
            del minus

            coef = (loss_plus - loss_minus) * inv_2eps_sq  # [k_c]
            del loss_minus, loss_plus

            for g, noise in zip(grad_acc, noises_c):
                g.add_((coef @ noise.view(k_c, -1)).view(g.shape))

            del noises_c, coef
            offset += k_c

        torch._foreach_div_(grad_acc, float(K))

        # Clean baseline loss at current θ — reported to the caller.
        out = model(*args[:1])
        clean_loss = criterion(out, *args[1:], **kwargs).detach()

        self._apply_update(optim_params, grad_acc, group_slices)
        return clean_loss

    # ---------------------------------------------------------------- cuda

    def _step_cuda(
        self,
        optim_params: list[Tensor],
        group_slices: list[tuple[int, int]],
        model: nn.Module,
        criterion: Callable[..., Tensor],
        args: tuple,
        kwargs: dict,
    ) -> Tensor:
        device = optim_params[0].device
        K = self.num_perturbations

        self._ensure_streams(K, device)
        self._ensure_buffers(optim_params, K, device)

        input_shapes = tuple(
            tuple(a.shape) if isinstance(a, Tensor) else None for a in args
        )
        param_ids = tuple(id(p) for p in optim_params)

        if not self.use_cuda_graph:
            self._run_cuda_inner(
                optim_params, group_slices, model, criterion, args, kwargs
            )
            return self._static_loss.clone()

        need_rebuild = (
            self._graph is None
            or self._captured_shapes != input_shapes
            or self._captured_param_ids != param_ids
        )
        if need_rebuild:
            self._build_graph(
                optim_params, group_slices, model, criterion, args, kwargs
            )
            self._captured_shapes = input_shapes
            self._captured_param_ids = param_ids

        for static, runtime in zip(self._static_inputs, args):
            if isinstance(runtime, Tensor):
                static.copy_(runtime, non_blocking=True)
        self._graph.replay()
        return self._static_loss.clone()

    def _ensure_streams(self, K: int, device: torch.device) -> None:
        if (
            self._streams is None
            or len(self._streams) != K
            or self._streams[0].device != device
        ):
            self._streams = [torch.cuda.Stream(device=device) for _ in range(K)]
            self._graph = None

    def _ensure_buffers(
        self, optim_params: list[Tensor], K: int, device: torch.device
    ) -> None:
        dtype = optim_params[0].dtype
        stale = (
            self._noise_buffers is None
            or len(self._noise_buffers) != K
            or len(self._noise_buffers[0]) != len(optim_params)
            or any(
                nb.shape != p.shape
                or nb.dtype != p.dtype
                or nb.device != p.device
                for nb, p in zip(self._noise_buffers[0], optim_params)
            )
        )
        if stale:
            self._noise_buffers = [
                [torch.empty_like(p) for p in optim_params] for _ in range(K)
            ]
            self._grad_acc = [torch.zeros_like(p) for p in optim_params]
            self._static_loss = torch.zeros((), device=device, dtype=dtype)
            self._graph = None

    def _build_graph(
        self,
        optim_params: list[Tensor],
        group_slices: list[tuple[int, int]],
        model: nn.Module,
        criterion: Callable[..., Tensor],
        args: tuple,
        kwargs: dict,
    ) -> None:
        device = optim_params[0].device

        self._static_inputs = [
            torch.empty_like(a) if isinstance(a, Tensor) else a for a in args
        ]
        for static, runtime in zip(self._static_inputs, args):
            if isinstance(runtime, Tensor):
                static.copy_(runtime)

        saved_params = [p.detach().clone() for p in optim_params]
        saved_moms = [
            self.state[p]["momentum"].detach().clone() for p in optim_params
        ]

        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(_WARMUP_ITERS):
                self._run_cuda_inner(
                    optim_params, group_slices, model, criterion,
                    self._static_inputs, kwargs,
                )
        torch.cuda.current_stream(device).wait_stream(warmup_stream)
        torch.cuda.synchronize(device)

        self._capture_stream = torch.cuda.Stream(device=device)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, stream=self._capture_stream):
            self._run_cuda_inner(
                optim_params, group_slices, model, criterion,
                self._static_inputs, kwargs,
            )

        torch.cuda.synchronize(device)
        for p, orig in zip(optim_params, saved_params):
            p.copy_(orig)
        for p, orig in zip(optim_params, saved_moms):
            self.state[p]["momentum"].copy_(orig)
        torch._foreach_zero_(self._grad_acc)

    def _run_cuda_inner(
        self,
        optim_params: list[Tensor],
        group_slices: list[tuple[int, int]],
        model: nn.Module,
        criterion: Callable[..., Tensor],
        inputs,
        kwargs: dict,
    ) -> None:
        eps = _PERTURBATION_STD
        inv_2eps_sq = 1.0 / (2.0 * eps * eps)
        device = optim_params[0].device

        fwd_arg = inputs[:1]
        extra_args = inputs[1:]

        torch._foreach_zero_(self._grad_acc)

        # Clean baseline forward before any stream forks — the racy sub-stream
        # forwards see random superpositions of in-flight perturbations and
        # do not reflect L(θ).
        out = model(*fwd_arg)
        self._static_loss.copy_(criterion(out, *extra_args, **kwargs))

        current_stream = torch.cuda.current_stream(device)
        for s in self._streams:
            s.wait_stream(current_stream)

        for k, stream in enumerate(self._streams):
            with torch.cuda.stream(stream):
                noises = self._noise_buffers[k]
                for n in noises:
                    n.normal_(mean=0.0, std=eps)

                # θ += δ — race-by-design on shared parameters.
                torch._foreach_add_(optim_params, noises)
                out = model(*fwd_arg)
                loss_plus = criterion(out, *extra_args, **kwargs)

                # θ -= 2δ
                torch._foreach_add_(optim_params, noises, alpha=-2.0)
                out = model(*fwd_arg)
                loss_minus = criterion(out, *extra_args, **kwargs)

                # θ += δ (nominal restore — other streams may have mutated θ
                # arbitrarily in between; this is the point of the dynamic).
                torch._foreach_add_(optim_params, noises)

                coef = (loss_plus - loss_minus) * inv_2eps_sq

                # Racy addcmul into the shared gradient accumulator.
                for g, n in zip(self._grad_acc, noises):
                    g.addcmul_(n, coef.expand_as(n))

        for s in self._streams:
            current_stream.wait_stream(s)

        torch._foreach_div_(self._grad_acc, float(self.num_perturbations))
        self._apply_update(optim_params, self._grad_acc, group_slices)

    # ------------------------------------------------------------ shared update

    def _apply_update(
        self,
        optim_params: list[Tensor],
        grad_acc: list[Tensor],
        group_slices: list[tuple[int, int]],
    ) -> None:
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
            # Decoupled weight decay (Loshchilov & Hutter 2019): shrink θ
            # before the momentum update so the update itself is not scaled
            # by the decay factor.
            wd = group["weight_decay"]
            if wd > 0.0:
                torch._foreach_mul_(group_params, 1.0 - group["lr"] * wd)
            scale = (-group["lr"] * ratio).to(group_mems[0].dtype)
            scaled = torch._foreach_mul(group_mems, scale)
            torch._foreach_add_(group_params, scaled)
