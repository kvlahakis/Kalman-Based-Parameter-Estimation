import torch
import torch.nn as nn
from torch import Tensor
from abc import ABC, abstractmethod

from .covariances import Covariance


class Initialisation(nn.Module, ABC):
    """
    Abstract base class for ensemble initialisation strategies.

    Generates N ensemble members spread around a reference centre point.
    Inheriting from nn.Module means any learnable parameters are automatically
    discovered by optimisers for end-to-end training.
    """

    @abstractmethod
    def forward(self, center: Tensor, N: int) -> Tensor:
        """
        Generate an initial ensemble.

        Args:
            center: reference point  (dim,) or (1, dim)
            N:      ensemble size

        Returns:
            ensemble: (N, dim)
        """


class GaussianInit(Initialisation):
    """
    Isotropic Gaussian initialisation:  center + σ · ε,   ε ~ N(0, I).

    Parameterised via log_scale = log(σ) so σ = exp(log_scale) > 0 for all
    real values of log_scale, matching the convention in ScaledIdentity.

    Args:
        std:         initial standard deviation σ (must be positive)
        track_grads: whether to track gradients through σ (for learning)
    """

    def __init__(self, std: float = 1.0, track_grads: bool = False) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(std).log(), requires_grad=track_grads)

    def forward(self, center: Tensor, N: int) -> Tensor:
        c   = center.reshape(1, -1)                                    # (1, dim)
        std = self.log_scale.exp()
        eps = torch.randn(N, c.shape[-1], dtype=c.dtype, device=c.device)
        return c.expand(N, -1) + std * eps                             # (N, dim)


class GaussianInitWithOffset(Initialisation):
    """
    Isotropic Gaussian initialisation with gaussianoffset:  center + ε + o,   ε ~ N(0, I), o ~ N(0, σI).
    """
    def __init__(self, std: float = 1.0, track_grads: bool = False, offset_std: float = 1.0) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(std).log(), requires_grad=track_grads)
        self.offset_std = offset_std
    def forward(self, center: Tensor, N: int) -> Tensor:
        c   = center.reshape(1, -1)                                    # (1, dim)
        std = self.log_scale.exp()
        eps = torch.randn(N, c.shape[-1], dtype=c.dtype, device=c.device)
        offset = torch.randn(1, c.shape[-1], dtype=c.dtype, device=c.device) * self.offset_std
        return (c + offset).expand(N, -1) + std * eps               # (N, dim)


class CovarianceInit(Initialisation):
    """
    Full-covariance Gaussian initialisation:  center + L ε,   ε ~ N(0, I),
    where cov() = L Lᵀ is any Covariance module.

    Useful when the initial spread is correlated across dimensions, or when
    the same Covariance object should be shared with the filter's noise terms.

    Args:
        cov: Covariance module defining the spread  (dim × dim)
    """

    def __init__(self, cov: Covariance) -> None:
        super().__init__()
        self.cov = cov

    def forward(self, center: Tensor, N: int) -> Tensor:
        c   = center.reshape(1, -1)                                    # (1, dim)
        mat = self.cov()                                               # (dim, dim)
        L   = torch.linalg.cholesky(mat)                               # (dim, dim)
        eps = torch.randn(N, mat.shape[0], dtype=mat.dtype, device=mat.device)
        return c.expand(N, -1) + eps @ L.T                             # (N, dim)


class DeterministicInit(Initialisation):
    """
    Deterministic initialisation: all N members equal to center (zero spread).

    Useful as a baseline or when the filter is expected to develop ensemble
    spread purely from its process-noise model.
    """

    def forward(self, center: Tensor, N: int) -> Tensor:
        return center.reshape(1, -1).expand(N, -1).clone()             # (N, dim)
