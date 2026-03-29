import torch
import torch.nn as nn
from torch import Tensor
from abc import ABC, abstractmethod


class ObservationFn(nn.Module, ABC):
    """
    Abstract base class for observation functions h : R^d → R^m.

    Maps a batch of state vectors (N, d) to a batch of observations (N, m).
    Inheriting from nn.Module allows learnable observation operators to be
    composed with the rest of the model for end-to-end training.
    """

    @abstractmethod
    def forward(self, z: Tensor) -> Tensor:
        """
        Apply the observation operator to a batch of state vectors.

        Args:
            z: state ensemble  (N, d)

        Returns:
            y: observed ensemble  (N, m)
        """


class FullObservation(ObservationFn):
    """
    Identity observation: all state variables are observed.

    h(z) = z,  so m = d.
    """

    def forward(self, z: Tensor) -> Tensor:
        return z


class MaskedObservation(ObservationFn):
    """
    Observes only the variables selected by a boolean mask.

    h(z) = z[:, mask],  so m = mask.sum().

    The mask is registered as a buffer so it moves with the module when
    .to(device) or .to(dtype) is called.

    Args:
        mask: 1-D boolean tensor of length d; True entries are observed.
    """

    def __init__(self, mask: list[bool]) -> None:
        super().__init__()
        self.mask = torch.tensor(mask, dtype=torch.bool, requires_grad=False)

    def forward(self, z: Tensor) -> Tensor:
        return z[:, self.mask]
