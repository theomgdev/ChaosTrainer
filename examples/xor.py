"""Train a tiny MLP on XOR with the Chaos optimizer.

Usage:
    python examples/xor.py
    python examples/xor.py --device cuda
    python examples/xor.py --device cuda --compile
    python examples/xor.py --compile --num-perturbations 16 --perturbation-chunk-size 4
"""

from __future__ import annotations

import argparse
import time

import torch
from torch import nn

from chaostrainer import Chaos


def main() -> None:
    parser = argparse.ArgumentParser(description="Chaos optimizer XOR demo.")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu / cuda / mps")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num-perturbations", type=int, default=8)
    parser.add_argument(
        "--perturbation-chunk-size",
        type=int,
        default=None,
        help="Cap peak VRAM by evaluating perturbations in chunks of this size.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the model with torch.compile to fuse the vmapped forward pass.",
    )
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

    if args.compile:
        model = torch.compile(model)

    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=device)
    Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], device=device)

    loss_fn = nn.MSELoss()
    optimizer = Chaos(
        model.parameters(),
        lr=args.lr,
        num_perturbations=args.num_perturbations,
        perturbation_chunk_size=args.perturbation_chunk_size,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Device: {device} | Params: {total_params} | "
        f"num_perturbations={args.num_perturbations} | "
        f"chunk_size={args.perturbation_chunk_size} | "
        f"compile={args.compile}"
    )

    converged_epoch = None
    for epoch in range(1, args.epochs + 1):
        # Synchronize so Time/Step reflects actual GPU compute, not just
        # dispatch latency. First compiled step includes graph capture.
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_t = time.time()
        loss = optimizer.step(model, loss_fn, X, Y).item()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_t = time.time() - start_t

        if loss < 1e-4:
            print(f"[{epoch:>5}] converged, loss={loss:.6f}, Time/Step: {step_t:.4f}s")
            converged_epoch = epoch
            break
        if epoch % 500 == 0:
            print(f"[{epoch:>5}] loss={loss:.6f}, Time/Step: {step_t:.4f}s")

    if converged_epoch is None:
        print(f"[{args.epochs:>5}] did not converge, final loss={loss:.6f}")

    predictions = model(X).detach().cpu().flatten().tolist()
    targets = Y.detach().cpu().flatten().tolist()
    print("\nFinal predictions:")
    for x, y, yhat in zip(X.cpu().tolist(), targets, predictions):
        status = "ok" if abs(y - yhat) < 0.1 else "off"
        print(f"  {x} -> target={y:.0f}  predicted={yhat:.4f}  [{status}]")


if __name__ == "__main__":
    main()
