import torch
from torch import Tensor
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Base class for all timeseriese forecasting models.
    """
    @abstractmethod
    def run(self, X0: Tensor, theta0: Tensor, observations: Tensor, dt: float) -> [Tensor, Tensor]:
        """
        Run the model over a full observation sequence.

        Args:
            X0:          initial state      (N, n)
            theta0:      initial parameters (N, p)
            observations: observation time series  (T, m)
            dt:          time step between observations
        
        Returns:
            X_hist: state history  (T, N, n)
            theta_hist: parameter history  (T, N, p)
        """
        pass