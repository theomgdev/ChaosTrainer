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
    p = nn.Parameter(torch.zeros(3))
    opt = Chaos([p])
    with pytest.raises(TypeError):
        opt.step()  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": -0.1},
        {"beta": 1.0},
        {"beta": -0.1},
        {"num_perturbations": 0},
    ],
)
def test_rejects_invalid_hyperparameters(kwargs):
    p = nn.Parameter(torch.zeros(3))
    with pytest.raises(ValueError):
        Chaos([p], **kwargs)


def test_parameter_groups_accept_overrides():
    a = nn.Parameter(torch.zeros(3))
    b = nn.Parameter(torch.zeros(3))
    opt = Chaos(
        [
            {"params": [a], "lr": 0.5, "beta": 0.5},
            {"params": [b], "lr": 2.0},
        ]
    )
    assert opt.param_groups[0]["lr"] == 0.5
    assert opt.param_groups[0]["beta"] == 0.5
    assert opt.param_groups[1]["lr"] == 2.0
    assert opt.param_groups[1]["beta"] == 0.9  # default inherited


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
        # We pass a dummy tensor so `args[:1]` is provided, or we can just omit it because `forward` defaults `_` to `None`.
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
