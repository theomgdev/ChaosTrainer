"""Train a linear classifier on MNIST with Chaos.

Usage:
    python examples/mnist.py --device cuda
    python examples/mnist.py --device cuda --compile --num-perturbations 10000
"""

from __future__ import annotations

import argparse
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from torchvision import datasets, transforms
except ImportError as e:
    raise ImportError(
        "torchvision is required to run the MNIST example. "
        "Please install it using: pip install torchvision"
    ) from e

from chaostrainer import Chaos


def main() -> None:
    parser = argparse.ArgumentParser(description="Chaos MNIST demo.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num-perturbations", type=int, default=1000)
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

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 10),
    ).to(device)

    if args.compile:
        model = torch.compile(model)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        # Constant batch shape lets the CUDA path reuse its captured graph.
        drop_last=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Chaos(
        model.parameters(),
        lr=args.lr,
        num_perturbations=args.num_perturbations,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Device: {device} | Params: {total_params} | "
        f"num_perturbations={args.num_perturbations} | compile={args.compile}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start_t = time.time()
            loss = optimizer.step(model, loss_fn, data, target).item()
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_t = time.time() - start_t

            if batch_idx % args.log_every == 0:
                print(
                    f"Train Epoch: {epoch} "
                    f"[{batch_idx * len(data):>5}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                    f"Loss: {loss:.6f}\tTime/Step: {step_t:.4f}s"
                )

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
        print(
            f"\n--- Test [Epoch {epoch}]: "
            f"loss={test_loss:.4f}  "
            f"accuracy={correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
        )


if __name__ == "__main__":
    main()
