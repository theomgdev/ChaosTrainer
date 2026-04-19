"""Unit tests for the Chaos optimizer.

Chaos dispatches on parameter device: CUDA parameters go through the
stream-parallel + CUDA-graph path (racy, non-deterministic); other devices
go through the deterministic vmap path. Tests that assert numeric
invariants run on CPU; CUDA tests only check that the racy path runs and
descends.
"""

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
        {"weight_decay": -0.1},
        {"num_perturbations": 0},
        {"perturbation_chunk_size": 0},
        {"perturbation_chunk_size": -1},
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
            {"params": list(model[0].parameters()), "lr": 0.5, "beta": 0.5, "weight_decay": 0.0},
            {"params": list(model[1].parameters()), "lr": 2.0},
        ]
    )
    assert opt.param_groups[0]["lr"] == 0.5
    assert opt.param_groups[0]["beta"] == 0.5
    assert opt.param_groups[0]["weight_decay"] == 0.0
    assert opt.param_groups[1]["lr"] == 2.0
    assert opt.param_groups[1]["beta"] == 0.9
    assert opt.param_groups[1]["weight_decay"] == 0.01


# ---------------------------------------------------------------------------
# CPU path behavior
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
    opt = Chaos(model.parameters(), lr=1.0, weight_decay=0.0)

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
    opt = Chaos(model.parameters(), lr=1e-2, num_perturbations=4, weight_decay=0.0)

    def criterion(x):
        return (x ** 2).sum()

    start = criterion(model.x).item()
    for _ in range(300):
        opt.step(model, criterion)

    end = criterion(model.x).item()
    assert end < start * 0.5


def test_weight_decay_shrinks_unoptimized_params():
    """With no gradient signal, wd > 0 should monotonically shrink ‖θ‖."""
    class PassthroughModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(torch.ones(16) * 5.0)

        def forward(self, _=None):
            return self.x

    model = PassthroughModel()
    opt = Chaos(model.parameters(), lr=1e-1, weight_decay=0.1, num_perturbations=4)

    def criterion(x):
        # Gradient-free constant: ES estimator averages to zero in expectation.
        return x.sum() * 0.0 + torch.tensor(1.0)

    start_norm = model.x.detach().norm().item()
    for _ in range(50):
        opt.step(model, criterion)
    end_norm = model.x.detach().norm().item()

    assert end_norm < start_norm


# ---------------------------------------------------------------------------
# param_groups alignment
# ---------------------------------------------------------------------------


def test_reversed_param_groups_still_converge():
    """Group order differing from model.named_parameters() order must not
    misalign gradients with their parameters."""
    model = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 2))
    groups = [
        {"params": list(model[2].parameters())},
        {"params": list(model[0].parameters())},
    ]
    opt = Chaos(groups, lr=1e-2, num_perturbations=4, weight_decay=0.0)

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
# Perturbation chunking (CPU vmap path)
# ---------------------------------------------------------------------------


def test_chunk_size_not_dividing_num_perturbations():
    """Trailing chunk smaller than chunk_size must still work."""
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
            weight_decay=0.0,
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
    """First step momentum norm is zero; _NORM_FLOOR must keep ratio finite."""
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
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Tanh(),
        nn.Linear(4, 1),
        nn.Sigmoid(),
    )
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    loss_fn = nn.MSELoss()

    # XOR requires non-zero weights; disable weight decay for this test.
    opt = Chaos(model.parameters(), lr=1e-2, num_perturbations=4, weight_decay=0.0)

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
    model = nn.Linear(3, 2)
    opt = Chaos(model.parameters(), lr=0.5, beta=0.8)

    def criterion(outputs):
        return outputs.pow(2).mean()

    for _ in range(3):
        opt.step(model, criterion, torch.randn(5, 3))

    state = opt.state_dict()

    model2 = nn.Linear(3, 2)
    model2.load_state_dict(model.state_dict())
    opt2 = Chaos(model2.parameters(), lr=0.5, beta=0.8)
    opt2.load_state_dict(state)

    p1 = next(iter(opt.state.values()))["momentum"]
    p2 = next(iter(opt2.state.values()))["momentum"]
    assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# CUDA path (stream-parallel + CUDA graph)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("use_cuda_graph", [True, False])
def test_cuda_path_descends(use_cuda_graph):
    """The racy CUDA path must still descend on a convex quadratic — the
    threshold is loose since race dynamics make convergence non-monotonic."""
    device = torch.device("cuda")

    class QuadraticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(torch.tensor([2.0, -1.5, 1.0], device=device))

        def forward(self, _=None):
            return self.x

    model = QuadraticModel()
    opt = Chaos(
        model.parameters(),
        lr=1e-2,
        num_perturbations=8,
        weight_decay=0.0,
        use_cuda_graph=use_cuda_graph,
    )

    def criterion(x):
        return (x ** 2).sum()

    start = criterion(model.x).item()
    for _ in range(300):
        opt.step(model, criterion)
    end = criterion(model.x).item()

    assert end < start * 0.7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_graph_rebuilds_on_input_shape_change():
    """Changing the input shape between calls must trigger transparent
    re-capture."""
    device = torch.device("cuda")
    model = nn.Linear(3, 3).to(device)
    opt = Chaos(model.parameters(), lr=1e-2, num_perturbations=4)

    def criterion(out, tgt):
        return (out - tgt).pow(2).mean()

    x1 = torch.randn(2, 3, device=device)
    y1 = torch.zeros(2, 3, device=device)
    for _ in range(3):
        opt.step(model, criterion, x1, y1)

    x2 = torch.randn(5, 3, device=device)
    y2 = torch.zeros(5, 3, device=device)
    loss = opt.step(model, criterion, x2, y2).item()
    assert torch.isfinite(torch.tensor(loss))
