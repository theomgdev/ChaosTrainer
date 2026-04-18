"""Train a tiny model on MNIST using the gradient-free Chaos optimizer.

Note: Chaos is an Evolution-Strategy optimizer designed for non-differentiable
or black-box problems. Using it on a standard differentiable problem like
MNIST with thousands of parameters is relatively inefficient compared to
Adam/SGD, and is provided here purely to demonstrate the API on a standard 
dataset.

Usage:
    python examples/mnist.py
    python examples/mnist.py --device cuda
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

from chaostrainer import Chaos


def main() -> None:
    parser = argparse.ArgumentParser(description="Chaos optimizer MNIST demo.")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu / cuda / mps")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # We use a simple linear model to keep the parameter count (7850)
    # small enough for zeroth-order approximation to remain viable.
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 10)
    ).to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and load the data
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Chaos(model.parameters(), lr=args.lr, num_perturbations=1000)

    print(f"Training on device: {device} | Total parameters: {sum(p.numel() for p in model.parameters())}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            def closure() -> torch.Tensor:
                return loss_fn(model(data), target)

            start_t = time.time()
            loss = optimizer.step(closure).item()
            step_t = time.time() - start_t

            if batch_idx % 20 == 0:
                print(f"Train Epoch: {epoch} "
                      f"[{batch_idx * len(data):>5}/{len(train_loader.dataset)} "
                      f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss:.6f}\tTime/Step: {step_t:.4f}s")

        # Evaluation phase
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