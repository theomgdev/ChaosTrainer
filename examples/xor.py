"""Train a tiny MLP on XOR with the Chaos optimizer.

Usage:
    python examples/xor.py
    python examples/xor.py --device cuda
"""

from __future__ import annotations

import argparse

import torch
from torch import nn

from chaostrainer import Chaos


def main() -> None:
    parser = argparse.ArgumentParser(description="Chaos optimizer XOR demo.")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu / cuda / mps")
    parser.add_argument("--epochs", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Tanh(),
        nn.Linear(4, 1),
        nn.Sigmoid(),
    ).to(device)

    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=device)
    Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], device=device)

    loss_fn = nn.MSELoss()
    optimizer = Chaos(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = optimizer.step(model, loss_fn, X, Y).item()

        if loss < 1e-4:
            print(f"[{epoch:>5}] converged, loss={loss:.6f}")
            break
        if epoch % 1500 == 0:
            print(f"[{epoch:>5}] loss={loss:.6f}")

    predictions = model(X).detach().cpu().flatten().tolist()
    targets = Y.detach().cpu().flatten().tolist()
    print("\nFinal predictions:")
    for x, y, yhat in zip(X.cpu().tolist(), targets, predictions):
        status = "ok" if abs(y - yhat) < 0.1 else "off"
        print(f"  {x} -> target={y:.0f}  predicted={yhat:.4f}  [{status}]")


if __name__ == "__main__":
    main()
