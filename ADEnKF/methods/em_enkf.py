"""
EM-style EnKF (Term A only) implemented as a standalone method.

This mirrors the structure of torchEnKF.da_methods.EnKF but
detaches the ensemble after each analysis step so that gradients
propagate only through the *current* forecast/analysis step
with respect to parameters (Expectation–Maximization style),
not through the full particle history (no Term B).

Intended usage:
    - L63 and gLV parameter estimation experiments
    - Gradient decomposition experiments
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torchdiffeq import odeint

from torchEnKF.da_methods import inv_logdet


def EnKF_EM(
    ode_func: torch.nn.Module,
    obs_func: torch.nn.Module,
    t_obs: torch.Tensor,
    y_obs: torch.Tensor,
    N_ensem: int,
    init_m: torch.Tensor,
    init_C_param,
    model_Q_param,
    noise_R_param,
    device: torch.device,
    *,
    init_X: Optional[torch.Tensor] = None,
    ode_method: str = "rk4",
    ode_options: Optional[Dict] = None,
    t0: float = 0.0,
    compute_likelihood: bool = True,
    tqdm=None,
) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
    """
    EM-style EnKF with stochastic perturbation and particle detachment.

    Differences vs torchEnKF.da_methods.EnKF:
        - Uses direct odeint (no adjoint).
        - After each analysis update, the ensemble X is detached
          from the computation graph (X = X.detach()).

    This keeps gradients corresponding to "Term A" (direct dependence
    of the likelihood on parameters through the current step) while
    ignoring "Term B" (history dependence of particles on parameters).
    """
    x_dim = init_m.shape[0]
    y_dim = y_obs.shape[-1]
    n_obs = y_obs.shape[0]
    bs = y_obs.shape[1:-1]  # batch shape (possibly empty)

    if ode_options is None:
        if n_obs > 0:
            step_size = (t_obs[1:] - t_obs[:-1]).min()
            ode_options = dict(step_size=float(step_size))

    step_size = ode_options["step_size"]

    if init_X is not None:
        X = init_X.detach()
    else:
        X = init_C_param(init_m.expand(*bs, N_ensem, x_dim))

    log_likelihood = torch.zeros(bs, device=device) if compute_likelihood else None
    res: Dict = {}

    noise_R_mat = noise_R_param.full()
    noise_R_inv = noise_R_param.inv()
    logdet_noise_R = noise_R_param.logdet()

    t_cur = float(t0)
    pbar = tqdm(range(n_obs), desc="Running EM-EnKF", leave=False) if tqdm is not None else range(n_obs)

    for j in pbar:
        # ----- Forecast step -----
        n_steps = round(((t_obs[j] - t_cur) / step_size).item())
        t_span = torch.linspace(t_cur, float(t_obs[j].item()), n_steps + 1, device=device)
        X = odeint(ode_func, X, t_span, method=ode_method, options=dict(step_size=step_size))[-1]
        t_cur = float(t_obs[j].item())

        if model_Q_param is not None:
            X = model_Q_param(X)

        X_m = X.mean(dim=-2).unsqueeze(-2)  # (*bs, 1, x_dim)
        X_ct = X - X_m                      # (*bs, N_ensem, x_dim)

        # ----- Analysis step (linear obs only, as in current experiments) -----
        H = obs_func.H  # (y_dim, x_dim)
        HX = X @ H.T
        HX_m = X_m @ H.T
        HX_ct = HX - HX_m

        y_obs_j = y_obs[j].unsqueeze(-2)  # (*bs, 1, y_dim)
        obs_perturb = noise_R_param(y_obs_j.expand(*bs, N_ensem, y_dim))

        C_ww_sq = 1.0 / math.sqrt(N_ensem - 1) * HX_ct  # (*bs, N_ensem, y_dim)
        v1 = obs_perturb - HX
        v2 = y_obs_j - HX_m
        v = torch.cat((v1, v2), dim=-2)  # (*bs, N_ensem + 1, y_dim)

        C_ww_R_invv, C_ww_R_logdet = inv_logdet(
            v, C_ww_sq, noise_R_mat, noise_R_inv, logdet_noise_R
        )  # (*bs, N_ensem+1, y_dim), (*bs)

        pre = C_ww_R_invv[..., :N_ensem, :]  # (*bs, N_ensem, y_dim)

        if compute_likelihood:
            part1 = -0.5 * (y_dim * math.log(2.0 * math.pi) + C_ww_R_logdet)
            part2 = -0.5 * (
                C_ww_R_invv[..., N_ensem:, :]
                @ (y_obs_j - HX_m).transpose(-1, -2)
            )
            log_likelihood += (part1 + part2.squeeze(-1).squeeze(-1))

        X = X + 1.0 / math.sqrt(N_ensem - 1) * (
            pre @ C_ww_sq.transpose(-1, -2)
        ) @ X_ct

        # Critical EM step: treat updated ensemble as *data* for the next step.
        X = X.detach()

    return X, res, log_likelihood


__all__ = ["EnKF_EM"]

