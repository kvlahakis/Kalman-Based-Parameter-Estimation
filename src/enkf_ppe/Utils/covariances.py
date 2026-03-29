import torch
import torch.nn as nn
from torch import Tensor
from abc import ABC, abstractmethod


class Covariance(nn.Module, ABC):
    """
    Abstract base class for learnable covariance matrices.

    The matrix dimension is fixed at construction time via `dim`.
    Subclasses must implement forward() to return a symmetric positive
    definite (dim, dim) matrix.  Inheriting from nn.Module means any
    trainable parameters are automatically discovered by optimisers.

    Args:
        dim: matrix size
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    @abstractmethod
    def forward(self) -> Tensor:
        """Return a covariance matrix of shape (dim, dim)."""


class ScaledIdentity(Covariance):
    """
    Learnable scaled-identity covariance: σ² · I.

    Parameterised via log_scale = log(σ) so σ = exp(log_scale) > 0 for all
    real values of log_scale, guaranteeing the matrix is symmetric positive
    definite regardless of the parameter value.

    Args:
        dim:        matrix size
        std:        initial standard deviation σ (must be positive)
        track_grads: whether to track gradients (eg for training)
    """

    def __init__(self, dim: int, std: float = 1.0, track_grads: bool = False) -> None:
        super().__init__(dim)
        self.log_scale = nn.Parameter(torch.tensor(std).log(), requires_grad=track_grads)

    def forward(self) -> Tensor:
        """Return σ² · I of shape (dim, dim)."""
        var = self.log_scale.mul(2).exp()
        return var * torch.eye(self.dim, dtype=self.log_scale.dtype, device=self.log_scale.device)
