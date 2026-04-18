"""ChaosTrainer: gradient-free PyTorch optimizers based on Evolution Strategies."""

from chaostrainer._version import __version__
from chaostrainer.optim import Chaos

__all__ = ["Chaos", "__version__"]
