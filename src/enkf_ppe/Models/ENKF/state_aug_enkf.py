"""State-Augmented Ensemble Kalman Filter (EnKF)

Physical parameters θ are appended to the state vector to form the augmented
state z_k = [x_k; θ_k] ∈ R^(n+p). The filter jointly estimates system state
and parameters by exploiting cross-correlations in the forecast covariance P_k^f.

Notation follows the root README:
  N  – ensemble size
  n  – state dimension
  p  – parameter dimension
  m  – observation dimension
  Σ  – process noise covariance       (n × n)
  Z  – artificial param noise cov     (p × p)
  Γ  – observation noise covariance   (m × m)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable

from ..base_model import BaseModel
from ...Utils.covariances import Covariance


class StateAugEnKF(nn.Module, BaseModel):
    """
    State-Augmented Ensemble Kalman Filter.

    The augmented state z_k = [x_k; θ_k] ∈ R^(n+p) is propagated and
    updated jointly at each assimilation cycle.

    Each noise covariance is a Covariance module, so its parameters can be
    trained end-to-end with a standard optimiser.

    Args:
        transition_fn:    Ψ – batched map (N, n), (N, p) → (N, n)
        obs_fn:           h – batched map (N, n+p) → (N, m)
        process_noise_cov:  Σ ∈ R^(n × n)  — n inferred from its .dim
        param_noise_cov:    Z ∈ R^(p × p)  — p inferred from its .dim
                            (artificial noise to prevent ensemble collapse of θ)
        obs_noise_cov:      Γ ∈ R^(m × m)
        time_step:          dt – time step of transition function
    """

    def __init__(
        self,
        transition_fn:     Callable[[Tensor, Tensor], Tensor],
        obs_fn:            Callable[[Tensor], Tensor],
        process_noise_cov: Covariance,
        param_noise_cov:   Covariance,
        obs_noise_cov:     Covariance,
        time_step:         float,
    ) -> None:
        super().__init__()
        self.psi   = transition_fn
        self.h     = obs_fn
        self.Sigma = process_noise_cov   # Σ  (n, n)
        self.Omega = param_noise_cov     # Ω  (p, p)
        self.Gamma = obs_noise_cov       # Γ  (m, m)
        self.n     = process_noise_cov.dim
        self.p     = param_noise_cov.dim
        self.dt    = time_step

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forecast(self, Z_ens: Tensor) -> Tensor:
        """
        Forecast (predict) step.  Propagates each ensemble member forward:

          ẑ_k^(i) = [Ψ(x_{k-1}^(i), θ_{k-1}^(i)) + η_k^(i) ;
                     θ_{k-1}^(i)                   + ζ_k^(i) ]

        where  η^(i) ~ N(0, Σ)  and  ζ^(i) ~ N(0, Ω).

        Args:
            Z_ens: current analysis ensemble  (N, n+p)

        Returns:
            Z_hat: forecast ensemble           (N, n+p)
        """
        N = Z_ens.shape[0]
        X, Theta = Z_ens[:, :self.n], Z_ens[:, self.n:]  # (N, n), (N, p)

        X_hat = self.psi(X, Theta)                        # (N, n)

        eta  = self._sample(self.Sigma, N)                 # (N, n)  η ~ N(0, Σ)
        zeta = self._sample(self.Omega, N)                 # (N, p)  ζ ~ N(0, Ω)

        Z_hat = torch.cat([X_hat + eta, Theta + zeta], dim=-1)  # (N, n+p)
        return Z_hat

    def analysis(self, Z_hat: Tensor, y: Tensor) -> Tensor:
        """
        Analysis (update) step.  Assimilates observation y_k:

          z_k^(i) = ẑ_k^(i) + K_k (y_k + ξ_k^(i) - h(ẑ_k^(i)))

        The Kalman gain avoids forming P_k^f explicitly.  With

          A = [ẑ_k^(1) - z̄_k, …, ẑ_k^(N) - z̄_k]      ∈ R^(n+p × N)
          B = [h(ẑ_k^(1)) - ȳ_k, …, h(ẑ_k^(N)) - ȳ_k]  ∈ R^(m × N)

        the gain is:

          K_k = 1/(N-1) · A B^T · (1/(N-1) · B B^T + Γ)^{-1}

        which is the ensemble approximation of the root README's expression:

          K_k = 1/(N-1) · A A^T H^T · (1/(N-1) · H A A^T H^T + Γ)^{-1}

        where B = H A holds exactly when h is linear.

        Args:
            Z_hat: forecast ensemble  (N, n+p)
            y:     observation        (m,)

        Returns:
            Z_new: analysis ensemble  (N, n+p)
        """
        N    = Z_hat.shape[0]
        coef = 1.0 / (N - 1)

        # Ensemble mean of forecast state
        z_bar = Z_hat.mean(dim=0)          # (n+p,)

        # Perturbation matrix  A  (n+p, N)
        A = (Z_hat - z_bar).T

        # Map each forecast member through h

        Y_hat = self.h(Z_hat[:, :self.n])              # (N, m)
        y_bar = Y_hat.mean(dim=0)          # (m,)

        # Observation-space perturbation matrix  B  (m, N)
        B = (Y_hat - y_bar).T

        # Materialise Γ for this step
        Gamma_mat = self.Gamma()           # (m, m)

        # Innovation covariance  S = 1/(N-1) · B B^T + Γ  (m, m)
        S = coef * (B @ B.T) + Gamma_mat  # (m, m)

        # Kalman gain  K = 1/(N-1) · A B^T · S^{-1}  (n+p, m)
        # Solved as  K = solve(S^T, 1/(N-1) · B A^T)^T  (S is symmetric)
        K = torch.linalg.solve(S, coef * (B @ A.T)).T  # (n+p, m)

        # Synthetic observation perturbations  ξ^(i) ~ N(0, Γ)
        xi = self._sample(self.Gamma, N)      # (N, m)

        # Per-member innovation  d^(i) = y_k + ξ^(i) - h(ẑ_k^(i))
        d = y.unsqueeze(0) + xi - Y_hat    # (N, m)

        # Analysis update
        Z_new = Z_hat + (K @ d.T).T        # (N, n+p)
        return Z_new

    def step(self, Z_ens: Tensor, y: Tensor, n_forecasts: int = 1, x_hist: list[Tensor] = None, theta_hist: list[Tensor] = None) -> Tensor:
        """
        One full EnKF cycle: n_forecasts × forecast → analysis.

        n_forecasts > 1 is useful when observations arrive less frequently
        than the model time step, so the ensemble is propagated forward
        n_forecasts steps before being updated against y.

        Args:
            Z_ens:       current analysis ensemble  (N, n+p)
            y:           observation at time k      (m,)
            n_forecasts: number of forecast steps to take before analysis

        Returns:
            Z_new: updated analysis ensemble  (N, n+p)
        """
        Z_hat = Z_ens
        for i in range(n_forecasts):
            Z_hat = self.forecast(Z_hat)
            if i != n_forecasts - 1:
                x_hist.append(Z_hat[:, :self.n])
                theta_hist.append(Z_hat[:, self.n:])
        output = self.analysis(Z_hat, y)
        x_hist.append(output[:, :self.n])
        theta_hist.append(output[:, self.n:])
        return output

    def run(self, X0: Tensor, theta0: Tensor, observations: Tensor, dt: float) -> [Tensor, Tensor]:
        """
        Run the filter over a full observation sequence.

        Args:
            X0:           initial state ensemble      (N, n)
            theta0:       initial parameter ensemble  (N, p)
            observations: observation time series     (T, m)
            dt:           time step between observations

        Returns:
            X_hist:     state history      (T, N, n)
            theta_hist: parameter history  (T, N, p)
        """

        n_forecasts = int(round(dt / self.dt))
        assert abs(n_forecasts * self.dt - dt) < 1e-9 * dt, (
            f"model time step ({self.dt}) must divide the observation time step ({dt})"
        )
        X_hist = []
        theta_hist = []
        Z = torch.cat([X0, theta0], dim=-1)  # (N, n+p)
        for y in observations:
            Z = self.step(Z, y, n_forecasts=n_forecasts, x_hist=X_hist, theta_hist=theta_hist)
        return torch.stack(X_hist), torch.stack(theta_hist)  # (T, N, n), (T, N, p)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample(cov: Covariance, N: int) -> Tensor:
        """Draw N samples from N(0, cov()) via Cholesky decomposition."""
        mat = cov()                                            # (dim, dim)
        L   = torch.linalg.cholesky(mat)                      # (dim, dim)
        z   = torch.randn(N, mat.shape[0], dtype=mat.dtype, device=mat.device)
        return z @ L.T                                         # (N, dim)
