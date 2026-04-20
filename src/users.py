from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


@dataclass
class User:
    """Keeps track of all the information for each user"""

    id: int
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler = None
    model_name: str


@dataclass
class IndependentUser:
    """Completely independent user."""

    def __init__(
        self,
        id: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dl: DataLoader,
        val_dl: DataLoader,
        test_dl: DataLoader,
        model_name: str,
        neighbors: list[int],
    ) -> None:
        self.id = id
        self.model = model
        self.optimizer = optimizer
        self.train_dl = train_dl
        self.train_iter = iter(train_dl)
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.model_name = model_name
        self.neighbors = neighbors

    def reset_train_iter(self):
        """When manually iterating over the train_iter, reset once we
        exhaust all the user's data."""
        self.train_iter = iter(self.train_dl)
