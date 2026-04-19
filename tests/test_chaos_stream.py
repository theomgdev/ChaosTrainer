"""Tests for the experimental ChaosStream optimizer.

Most assertions are non-deterministic by design (the optimizer is racy), so
we only test invariants that must hold regardless of interleaving: error
surfaces, device constraints, and the existence of any optimization signal.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from chaostrainer.optim import ChaosStream


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
        ChaosStream([p], **kwargs)


def test_raises_on_cpu():
    model = nn.Linear(3, 3)
    opt = ChaosStream(model.parameters(), lr=1e-2, num_perturbations=2)

    def criterion(outputs):
        return outputs.pow(2).mean()

    with pytest.raises(RuntimeError):
        opt.step(model, criterion, torch.randn(2, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_runs_and_reduces_loss_on_cuda():
    """End-to-end smoke test: a well-conditioned quadratic must descend even
    under racy stream updates, though convergence is non-monotonic."""
    torch.manual_seed(0)
    device = torch.device("cuda")

    class QuadraticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Parameter(torch.tensor([2.0, -1.5, 1.0], device=device))

        def forward(self, _=None):
            return self.x

    model = QuadraticModel()
    opt = ChaosStream(model.parameters(), lr=1e-2, num_perturbations=8)

    def criterion(x):
        return (x ** 2).sum()

    start = criterion(model.x).item()
    for _ in range(300):
        opt.step(model, criterion)
    end = criterion(model.x).item()

    # Racy updates should still descend on a convex quadratic; loosen the
    # threshold compared to the deterministic Chaos test.
    assert end < start * 0.7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_rejects_params_not_on_model():
    device = torch.device("cuda")
    model = nn.Linear(3, 3).to(device)
    stray = nn.Parameter(torch.zeros(3, device=device))
    opt = ChaosStream([*model.parameters(), stray], num_perturbations=2)

    def criterion(outputs, tgt):
        return (outputs - tgt).pow(2).mean()

    with pytest.raises(ValueError):
        opt.step(
            model,
            criterion,
            torch.randn(2, 3, device=device),
            torch.zeros(2, 3, device=device),
        )
