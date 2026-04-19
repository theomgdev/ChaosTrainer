"""MNIST demo for the experimental stream-parallel ChaosStream optimizer.

ChaosStream dispatches each perturbation on its own CUDA stream and mutates
the shared parameters in place — updates are racy by design. Trajectories
are not reproducible across runs, and the estimator is not a clean ES
gradient; see chaostrainer.optim.chaos_stream for the full caveat.

CUDA only.

Usage:
    python examples/mnist_stream.py --num-perturbations 16
    python examples/mnist_stream.py --compile --num-perturbations 32
"""

from __future__ import annotations

import argparse
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from torchvision import datasets, transforms
except ImportError:
    raise ImportError(
        "torchvision is required to run the MNIST example. "
        "Please install it using: pip install torchvision"
    )

from chaostrainer.optim import ChaosStream


def main() -> None:
    parser = argparse.ArgumentParser(description="ChaosStream MNIST demo (CUDA only).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num-perturbations", type=int, default=16)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the model with torch.compile.",
    )
    parser.add_argument("--log-every", type=int, default=20, help="Log every N batches.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("ChaosStream requires CUDA; pass --device cuda.")

    # Linear classifier: 7850 params keeps the racy variant's memory modest.
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 10)
    ).to(device)

    if args.compile:
        model = torch.compile(model)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = ChaosStream(
        model.parameters(),
        lr=args.lr,
        num_perturbations=args.num_perturbations,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Device: {device} | Params: {total_params} | "
        f"num_perturbations={args.num_perturbations} | compile={args.compile}"
    )
    print("NOTE: ChaosStream is racy; trajectories vary run-to-run.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            torch.cuda.synchronize()
            start_t = time.time()
            loss = optimizer.step(model, loss_fn, data, target).item()
            torch.cuda.synchronize()
            step_t = time.time() - start_t

            if batch_idx % args.log_every == 0:
                print(f"Train Epoch: {epoch} "
                      f"[{batch_idx * len(data):>5}/{len(train_loader.dataset)} "
                      f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                      f"Loss: {loss:.6f}\tTime/Step: {step_t:.4f}s")

        model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f"\n--- Test set results [Epoch {epoch}]: "
              f"Average loss: {test_loss:.4f}, "
              f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")


if __name__ == "__main__":
    main()
