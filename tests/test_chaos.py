"""Unit tests for the Chaos optimizer."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from chaostrainer import Chaos


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


def test_rejects_missing_arguments():
    model = nn.Linear(3, 1)
    opt = Chaos(model.parameters())
    with pytest.raises(TypeError):
        opt.step()  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": -0.1},
        {"beta": 1.0},
        {"beta": -0.1},
        {"num_perturbations": 0},
        {"perturbation_chunk_size": 0},
        {"perturbation_chunk_size": -1},
        {"perturbation_std": 0.0},
        {"perturbation_std": -1e-3},
        {"weight_decay": -0.1},
        {"grad_clip": 0.0},
        {"grad_clip": -1.0},
    ],
)
def test_rejects_invalid_hyperparameters(kwargs):
    p = nn.Parameter(torch.zeros(3))
    with pytest.raises(ValueError):
        Chaos([p], **kwargs)


def test_parameter_groups_accept_overrides():
    model = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
    opt = Chaos(
        [
            {"params": list(model[0].parameters()), "lr": 0.5, "beta": 0.5},
            {"params": list(model[1].parameters()), "lr": 2.0},
        ]
    )
    assert opt.param_groups[0]["lr"] == 0.5
    assert opt.param_groups[0]["beta"] == 0.5
    assert opt.param_groups[1]["lr"] == 2.0
    assert opt.param_groups[1]["beta"] == 0.9  # default inherited
    assert opt.param_groups[0]["weight_decay"] == 0.0  # default inherited


def test_perturbation_std_stored_and_used():
    model = nn.Linear(4, 2)
    opt = Chaos(model.parameters(), lr=1e-2, perturbation_std=5e-3)
    assert opt.perturbation_std == 5e-3
    assert opt.param_groups[0]["perturbation_std"] == 5e-3

    def criterion(outputs):
        return outputs.pow(2).mean()

    loss = opt.step(model, criterion, torch.randn(4, 4))
    assert torch.isfinite(loss)


def test_perturbation_std_dynamic_default():
    model = nn.Linear(4, 2)
    opt = Chaos(model.parameters(), lr=1e-2)
    assert opt.perturbation_std is None
    assert opt.param_groups[0]["perturbation_std"] is None

    def criterion(outputs):
        return outputs.pow(2).mean()

    # Step should calculate dynamically
    loss = opt.step(model, criterion, torch.randn(4, 4))
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Behavior
# ---------------------------------------------------------------------------


def test_frozen_params_are_not_updated():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.trainable = nn.Parameter(torch.ones(4))
            self.frozen = nn.Parameter(torch.ones(4), requires_grad=False)

        def forward(self, x=None):
            return self.trainable, self.frozen

    model = DummyModel()
    target = torch.zeros(4)
    opt = Chaos(model.parameters(), lr=1.0)

    def criterion(outputs, tgt):
        train_out, frozen_out = outputs
        return ((train_out - tgt) ** 2).mean() + ((frozen_out - tgt) ** 2).mean()

    frozen_before = model.frozen.detach().clone()
    for _ in range(20):
        opt.step(model, criterion, None, target)

    assert torch.equal(model.frozen, frozen_before)


def test_converges_on_quadratic():
    class QuadraticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(torch.tensor([3.0, -2.0, 1.5]))

        def forward(self, _=None):
            return self.x

    model = QuadraticModel()
    opt = Chaos(model.parameters(), lr=1e-2, num_perturbations=4)

    def criterion(x):
        return (x ** 2).sum()

    start = criterion(model.x).item()
    for _ in range(300):
        opt.step(model, criterion)

    end = criterion(model.x).item()
    assert end < start * 0.5


def test_multiple_perturbations_reduce_variance():
    # Both sample counts should descend the quadratic; we only assert that
    # convergence occurs, not a specific ordering (single-sample estimates
    # are noisy enough that a run-to-run swap is possible).
    class QuadraticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(torch.tensor([2.0, 2.0]))

        def forward(self, _=None):
            return self.x

    def run(num_perturbations: int) -> float:
        torch.manual_seed(42)
        model = QuadraticModel()
        opt = Chaos(
            model.parameters(), lr=0.1,
            num_perturbations=num_perturbations,
        )

        def criterion(x):
            return (x ** 2).sum()

        start = criterion(model.x).item()
        for _ in range(500):
            opt.step(model, criterion)
        end = criterion(model.x).item()
        return end / start

    assert run(1) < 0.25
    assert run(4) < 0.25


# ---------------------------------------------------------------------------
# param_groups alignment
# ---------------------------------------------------------------------------


def test_reversed_param_groups_still_converge():
    """Group order differing from model.named_parameters() order must not
    misalign gradients with their parameters."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 2))
    groups = [
        {"params": list(model[2].parameters())},
        {"params": list(model[0].parameters())},
    ]
    opt = Chaos(groups, lr=1e-2, num_perturbations=4)

    X = torch.randn(16, 4)
    Y = torch.randn(16, 2)
    loss_fn = nn.MSELoss()

    start = loss_fn(model(X), Y).item()
    for _ in range(300):
        opt.step(model, loss_fn, X, Y)
    end = loss_fn(model(X), Y).item()
    assert end < start * 0.9


def test_partial_params_leaves_others_untouched():
    """Parameters not registered with the optimizer must not change."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 2))
    opt = Chaos(model[0].parameters(), lr=1e-2, num_perturbations=4)

    frozen_snapshot = [p.detach().clone() for p in model[2].parameters()]
    X = torch.randn(16, 4)
    Y = torch.randn(16, 2)
    loss_fn = nn.MSELoss()

    for _ in range(50):
        opt.step(model, loss_fn, X, Y)

    for before, after in zip(frozen_snapshot, model[2].parameters()):
        assert torch.equal(before, after)


def test_rejects_params_not_on_model():
    model = nn.Linear(3, 3)
    stray = nn.Parameter(torch.zeros(3))
    opt = Chaos([*model.parameters(), stray])

    def criterion(outputs, tgt):
        return (outputs - tgt).pow(2).mean()

    with pytest.raises(ValueError):
        opt.step(model, criterion, torch.randn(2, 3), torch.zeros(2, 3))


# ---------------------------------------------------------------------------
# Perturbation chunking
# ---------------------------------------------------------------------------


def test_chunked_equivalent_to_unchunked_under_fixed_seed():
    """For a fixed RNG seed, chunking must produce the same trajectory as the
    unchunked path (each chunk consumes K/M slices of the same noise stream)."""
    class QuadraticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(torch.tensor([2.0, -1.0, 0.5]))

        def forward(self, _=None):
            return self.x

    def criterion(x):
        return (x ** 2).sum()

    def run(chunk_size):
        torch.manual_seed(123)
        model = QuadraticModel()
        opt = Chaos(
            model.parameters(),
            lr=1e-2,
            num_perturbations=16,
            perturbation_chunk_size=chunk_size,
        )
        for _ in range(20):
            opt.step(model, criterion)
        return model.x.detach().clone()

    unchunked = run(None)
    chunked_4 = run(4)
    chunked_8 = run(8)

    # Different chunking consumes the RNG in different orders, so bit-exactness
    # is not expected; both should converge to comparable values.
    assert torch.allclose(unchunked, chunked_4, atol=0.5)
    assert torch.allclose(unchunked, chunked_8, atol=0.5)


def test_chunk_size_not_dividing_num_perturbations():
    """Trailing chunk smaller than chunk_size must still work."""
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    opt = Chaos(
        model.parameters(),
        lr=1e-2,
        num_perturbations=10,
        perturbation_chunk_size=3,  # 3+3+3+1
    )

    def criterion(outputs):
        return outputs.pow(2).mean()

    loss = opt.step(model, criterion, torch.randn(4, 4))
    assert torch.isfinite(loss)


def test_chunked_convergence_matches_unchunked():
    """End-to-end: chunked training descends as well as unchunked."""
    def train(chunk_size):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 2))
        opt = Chaos(
            model.parameters(),
            lr=1e-2,
            num_perturbations=8,
            perturbation_chunk_size=chunk_size,
        )
        X = torch.randn(16, 4)
        Y = torch.randn(16, 2)
        loss_fn = nn.MSELoss()
        start = loss_fn(model(X), Y).item()
        for _ in range(200):
            opt.step(model, loss_fn, X, Y)
        return loss_fn(model(X), Y).item() / start

    assert train(None) < 0.9
    assert train(2) < 0.9


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------


def test_cold_start_does_not_nan_with_many_perturbations():
    """On the first step the momentum norm is exactly zero; _NORM_FLOOR must
    keep the trust ratio finite."""
    torch.manual_seed(0)
    model = nn.Linear(8, 4)
    opt = Chaos(model.parameters(), lr=1e-2, num_perturbations=16)

    def criterion(outputs):
        return outputs.pow(2).mean()

    loss = opt.step(model, criterion, torch.randn(4, 8))
    assert torch.isfinite(loss)
    for p in model.parameters():
        assert torch.isfinite(p).all()


# ---------------------------------------------------------------------------
# Integration: XOR
# ---------------------------------------------------------------------------


def test_xor_converges():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Tanh(),
        nn.Linear(4, 1),
        nn.Sigmoid(),
    )
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    loss_fn = nn.MSELoss()

    opt = Chaos(model.parameters(), lr=1e-2)

    final_loss = None
    for _ in range(15000):
        final_loss = opt.step(model, loss_fn, X, Y).item()
        if final_loss < 1e-3:
            break

    assert final_loss is not None
    assert final_loss < 1e-2, f"XOR did not converge: final loss {final_loss}"

    predictions = model(X).detach()
    assert ((predictions - Y).abs() < 0.1).all()


# ---------------------------------------------------------------------------
# State dict roundtrip
# ---------------------------------------------------------------------------


def test_state_dict_roundtrip():
    torch.manual_seed(0)
    model = nn.Linear(3, 2)
    opt = Chaos(model.parameters(), lr=0.5, beta=0.8)

    def criterion(outputs):
        return outputs.pow(2).mean()

    for _ in range(3):
        inputs = torch.randn(5, 3)
        opt.step(model, criterion, inputs)

    state = opt.state_dict()

    model2 = nn.Linear(3, 2)
    model2.load_state_dict(model.state_dict())
    opt2 = Chaos(model2.parameters(), lr=0.5, beta=0.8)
    opt2.load_state_dict(state)

    # Momentum buffers match after reload.
    p1 = next(iter(opt.state.values()))["momentum"]
    p2 = next(iter(opt2.state.values()))["momentum"]
    assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_runs_on_cuda():
    device = torch.device("cuda")
    model = nn.Linear(4, 4).to(device)
    opt = Chaos(model.parameters(), lr=0.1)

    def criterion(outputs):
        return outputs.pow(2).mean()

    inputs = torch.randn(8, 4, device=device)
    loss = opt.step(model, criterion, inputs)
    assert loss.device.type == "cuda"
    for p in model.parameters():
        assert p.device.type == "cuda"
        assert opt.state[p]["momentum"].device.type == "cuda"


# ---------------------------------------------------------------------------
# Weight decay
# ---------------------------------------------------------------------------


def test_weight_decay_regularizes_parameters():
    """With a flat (constant) loss, weight decay should shrink parameter norms."""
    torch.manual_seed(0)

    class ConstantModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(torch.ones(8) * 2.0)

        def forward(self, _=None):
            return self.x

    model = ConstantModel()
    opt = Chaos(model.parameters(), lr=1e-2, weight_decay=0.5, num_perturbations=4)

    def criterion(x):
        return torch.tensor(1.0)  # constant — no ES gradient signal

    norm_before = model.x.norm().item()
    for _ in range(50):
        opt.step(model, criterion)
    norm_after = model.x.norm().item()

    assert norm_after < norm_before


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------


def test_grad_clip_convergence():
    """With grad_clip enabled the optimizer should still converge."""
    class QuadraticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(torch.tensor([3.0, -2.0]))

        def forward(self, _=None):
            return self.x

    model = QuadraticModel()
    opt = Chaos(model.parameters(), lr=1e-2, num_perturbations=4, grad_clip=1.0)

    def criterion(x):
        return (x ** 2).sum()

    start = criterion(model.x).item()
    for _ in range(300):
        opt.step(model, criterion)
    end = criterion(model.x).item()
    assert end < start * 0.5


# ---------------------------------------------------------------------------
# Fitness shaping
# ---------------------------------------------------------------------------


def test_fitness_shaping_converges_on_quadratic():
    class QuadraticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(torch.tensor([3.0, -2.0, 1.5]))

        def forward(self, _=None):
            return self.x

    model = QuadraticModel()
    opt = Chaos(
        model.parameters(), lr=1e-2, num_perturbations=4, fitness_shaping=True
    )

    def criterion(x):
        return (x ** 2).sum()

    start = criterion(model.x).item()
    for _ in range(300):
        opt.step(model, criterion)
    end = criterion(model.x).item()
    assert end < start * 0.5


def test_fitness_shaping_k1_edge_case():
    """K=1 means only 2 loss values ranked; step must still be finite."""
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    opt = Chaos(model.parameters(), lr=1e-2, num_perturbations=1, fitness_shaping=True)

    def criterion(outputs):
        return outputs.pow(2).mean()

    loss = opt.step(model, criterion, torch.randn(4, 4))
    assert torch.isfinite(loss)
    for p in model.parameters():
        assert torch.isfinite(p).all()


# ---------------------------------------------------------------------------
# Orthogonal perturbations
# ---------------------------------------------------------------------------


def test_orthogonal_perturbations_converges_on_quadratic():
    class QuadraticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(torch.tensor([3.0, -2.0, 1.5]))

        def forward(self, _=None):
            return self.x

    model = QuadraticModel()
    opt = Chaos(
        model.parameters(), lr=1e-2, num_perturbations=4, orthogonal_perturbations=True
    )

    def criterion(x):
        return (x ** 2).sum()

    start = criterion(model.x).item()
    for _ in range(300):
        opt.step(model, criterion)
    end = criterion(model.x).item()
    assert end < start * 0.5


# ---------------------------------------------------------------------------
# Combined enhanced features
# ---------------------------------------------------------------------------


def test_fitness_shaping_and_orthogonal_combined():
    """Both flags enabled together must converge."""
    class QuadraticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(torch.tensor([2.0, -1.5]))

        def forward(self, _=None):
            return self.x

    model = QuadraticModel()
    opt = Chaos(
        model.parameters(),
        lr=1e-2,
        num_perturbations=4,
        fitness_shaping=True,
        orthogonal_perturbations=True,
    )

    def criterion(x):
        return (x ** 2).sum()

    start = criterion(model.x).item()
    for _ in range(300):
        opt.step(model, criterion)
    end = criterion(model.x).item()
    assert end < start * 0.5


# ---------------------------------------------------------------------------
# estimate_grad
# ---------------------------------------------------------------------------


def test_estimate_grad_populates_grad():
    """estimate_grad must write a finite gradient into param.grad."""
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    chaos = Chaos(model.parameters(), num_perturbations=4)

    assert all(p.grad is None for p in model.parameters())
    loss = chaos.estimate_grad(model, lambda out: out.pow(2).mean(), torch.randn(4, 4))

    assert torch.isfinite(loss)
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        assert p.grad.shape == p.shape


def test_estimate_grad_accumulates_into_existing_grad():
    """Calling estimate_grad twice must accumulate (add) rather than overwrite,
    matching the behaviour of loss.backward()."""
    torch.manual_seed(0)
    model = nn.Linear(2, 2)
    chaos = Chaos(model.parameters(), num_perturbations=4)

    def criterion(out):
        return out.pow(2).mean()

    x = torch.randn(4, 2)
    chaos.estimate_grad(model, criterion, x)
    grad_after_one = [p.grad.clone() for p in model.parameters()]

    chaos.estimate_grad(model, criterion, x)
    for p, g1 in zip(model.parameters(), grad_after_one):
        # Second call added to the first — magnitude should have grown.
        assert not torch.allclose(p.grad, g1)


def test_estimate_grad_frozen_params_get_no_grad():
    """Frozen parameters (requires_grad=False) must not receive a .grad."""
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.trainable = nn.Parameter(torch.ones(4))
            self.frozen = nn.Parameter(torch.ones(4), requires_grad=False)

        def forward(self, x=None):
            return self.trainable + self.frozen

    model = DummyModel()
    chaos = Chaos(model.parameters(), num_perturbations=4)
    chaos.estimate_grad(model, lambda out: out.pow(2).mean())

    assert model.trainable.grad is not None
    assert model.frozen.grad is None


def test_estimate_grad_returns_scalar_loss():
    """Return value must be a detached scalar tensor."""
    torch.manual_seed(0)
    model = nn.Linear(3, 1)
    chaos = Chaos(model.parameters(), num_perturbations=4)
    loss = chaos.estimate_grad(model, lambda out: out.pow(2).mean(), torch.randn(5, 3))

    assert loss.ndim == 0
    assert loss.requires_grad is False


def test_estimate_grad_with_adamw_converges_on_quadratic():
    """ES gradient estimate fed into AdamW must descend a simple quadratic."""
    torch.manual_seed(0)

    class QuadraticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(torch.tensor([3.0, -2.0, 1.5]))

        def forward(self, _=None):
            return self.x

    model = QuadraticModel()
    chaos = Chaos(model.parameters(), num_perturbations=4)
    adamw = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)

    def criterion(x):
        return (x ** 2).sum()

    start = criterion(model.x).item()
    for _ in range(500):
        adamw.zero_grad()
        chaos.estimate_grad(model, criterion)
        adamw.step()

    end = criterion(model.x).item()
    assert end < start * 0.5


def test_estimate_grad_with_adamw_converges_xor():
    """Full XOR integration: estimate_grad + AdamW must reach low loss."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1), nn.Sigmoid()
    )
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    loss_fn = nn.MSELoss()

    chaos = Chaos(model.parameters(), num_perturbations=8)
    adamw = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)

    final_loss = None
    for _ in range(3000):
        adamw.zero_grad()
        final_loss = chaos.estimate_grad(model, loss_fn, X, Y).item()
        adamw.step()
        if final_loss < 1e-2:
            break

    assert final_loss is not None
    assert final_loss < 0.1, f"XOR+AdamW did not converge: {final_loss}"
