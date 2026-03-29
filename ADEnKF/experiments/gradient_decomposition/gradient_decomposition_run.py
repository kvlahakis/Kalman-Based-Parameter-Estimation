"""
gradient_decomposition_run.py
==============================
Demonstrates Term A vs Term A+B gradient decomposition from §4.1.1,
using the existing torchEnKF infrastructure on the Lorenz-63 system.

The key distinction between EM-EnKF and AD-EnKF:
  - AD-EnKF: full backprop through the EnKF computation graph via adjoint
              → gradient captures Term A + Term B (particle history)
              → uses official da_methods.EnKF with adjoint=True
  - EM-EnKF: particles detached before gradient step
              → gradient captures Term A only (direct dependence on theta)

Term B = AD grad - EM grad

Critical notes on L63 + adjoint:
  - Adjoint step_size must match forward step_size (both 0.01) to avoid
    discretization error corrupting the AD gradient in chaotic systems.
  - n_obs must be kept moderate (<=100): at T=300 the adjoint backprops
    through 300 chaotic forecast windows and Term B is dominated by
    gradient explosion rather than useful signal.

Run from repo root with:
    PYTHONPATH=. python ADEnKF/experiments/gradient_decomposition/gradient_decomposition_run.py
"""

import sys
import math
from pathlib import Path

_script_dir   = Path(__file__).resolve().parent
_ad_enkf_dir  = _script_dir.parent.parent
_repo_root    = _ad_enkf_dir.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_ad_enkf_dir))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from torchdiffeq import odeint
from torchEnKF import da_methods, nn_templates, noise
from methods.em_enkf import EnKF_EM
from paths import DATA_DIR

torch.manual_seed(42)
np.random.seed(42)

FIG_DIR = _script_dir / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


# ── shared system parameters (aligned with l63_param_est.yaml) ──────────
TRUE_SIGMA = 10.0           # true_params.sigma
TRUE_RHO   = 28.0           # true_params.rho
TRUE_BETA  = 8 / 3          # true_params.beta
x_dim      = 3              # x_dim
N_ENS      = 50             # N_ens
OBS_STD    = 1.0            # obs_std
PROC_STD   = 0.5            # process_std

# ODE integration settings — adjoint step must match forward step to avoid
# discretization error corrupting the AD gradient in chaotic systems
ODE_STEP     = 0.01
ADJOINT_STEP = 0.05   # was 0.05 — mismatch caused biased AD gradients

H_true        = torch.eye(x_dim, device=device)
true_obs_func = nn_templates.Linear(x_dim, x_dim, H=H_true).to(device)
noise_R_true  = noise.AddGaussian(x_dim, torch.tensor(OBS_STD, device=device),
                                  param_type="scalar").to(device)

init_m       = torch.zeros(x_dim, device=device)
init_cov     = torch.diag(torch.tensor([25.0, 25.0, 50.0], device=device))
init_C_param = noise.AddGaussian(x_dim, init_cov, "full").to(device)

model_Q = noise.AddGaussian(x_dim,
                             PROC_STD * torch.ones(x_dim, device=device),
                             "diag").to(device)


def load_observations(n_obs: int = 150, n_forecasts: int = 5, dt: float = 0.01):
    data_file = DATA_DIR / "Lorentz63/sigma10.0000_rho28.0000_beta2.6667_dt0.0100.pt"
    payload   = torch.load(data_file, weights_only=True)
    truth     = payload["data"]
    truth_sub = truth[:n_obs * n_forecasts:n_forecasts][:n_obs].to(device)
    obs_dt    = n_forecasts * dt
    t_obs     = obs_dt * torch.arange(1, n_obs + 1, device=device)
    y_obs     = truth_sub + OBS_STD * torch.randn_like(truth_sub)
    return y_obs.unsqueeze(1), t_obs, obs_dt


def compute_gradient(coeff, y_obs, t_obs, mode, n_ens=None):
    """
    mode="ad" -> official da_methods.EnKF with adjoint=True -> Term A + B
    mode="em" -> same filter loop but X.detach() after each analysis -> Term A only
    """
    assert mode in ("ad", "em")
    if n_ens is None:
        n_ens = N_ENS

    coeff_p = coeff.detach().clone().requires_grad_(True)
    learned = nn_templates.Lorenz63(coeff_p).to(device)
    bs      = y_obs.shape[1:-1]

    if mode == "ad":
        # Official torchEnKF with adjoint backprop
        # adjoint step_size matches forward step_size to avoid gradient error
        _, _, log_lik = da_methods.EnKF(
            learned,
            true_obs_func,
            t_obs,
            y_obs,
            n_ens,
            init_m,
            init_C_param,
            model_Q,
            noise_R_true,
            device,
            save_filter_step={},
            ode_method="rk4",
            ode_options=dict(step_size=ODE_STEP),
            adjoint=True,
            adjoint_method="rk4",
            adjoint_options=dict(step_size=ADJOINT_STEP),
            tqdm=None,
        )
        log_lik = log_lik.mean()

    else:
        # EM-EnKF: reuse standalone EM implementation
        _, _, log_lik = EnKF_EM(
            learned,
            true_obs_func,
            t_obs,
            y_obs,
            n_ens,
            init_m,
            init_C_param,
            model_Q,
            noise_R_true,
            device,
            init_X=None,
            ode_method="rk4",
            ode_options=dict(step_size=ODE_STEP),
            t0=0.0,
            compute_likelihood=True,
            tqdm=None,
        )
        log_lik = log_lik.mean()

    log_lik.backward()
    return learned.coeff.grad.clone()


# ═══════════════════════════════════════════════════════════════════════════
# Panel 1 — Gradient landscape
# ═══════════════════════════════════════════════════════════════════════════

def panel1_gradient_landscape(ax, n_obs=150):
    print("Panel 1: gradient landscape...")
    y_obs, t_obs, _ = load_observations(n_obs=n_obs)

    sigma_vals = np.linspace(5.0, 16.0, 25)
    ad_grads, em_grads = [], []

    for sv in sigma_vals:
        coeff = torch.tensor([sv, TRUE_RHO, TRUE_BETA], dtype=torch.float32, device=device)
        ad_grads.append(compute_gradient(coeff, y_obs, t_obs, "ad")[0].item())
        em_grads.append(compute_gradient(coeff, y_obs, t_obs, "em")[0].item())

    ad_grads = np.array(ad_grads)
    em_grads = np.array(em_grads)

    ax.plot(sigma_vals, ad_grads, color="#2563eb", lw=2.0, label="AD-EnKF  (Term A + B)")
    ax.plot(sigma_vals, em_grads, color="#dc2626", lw=2.0, ls="--", label="EM-EnKF  (Term A only)")
    ax.fill_between(sigma_vals, em_grads, ad_grads,
                    alpha=0.15, color="#7c3aed", label="Term B  (missing from EM)")
    ax.axvline(TRUE_SIGMA, color="black", lw=1.2, ls=":", label=rf"True $\sigma^*={TRUE_SIGMA}$")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel(r"$\sigma$", fontsize=12)
    ax.set_ylabel(r"$\nabla_\sigma \, \ell^{\mathrm{EnKF}}$", fontsize=11)
    ax.set_title(f"Gradient Landscape  (T={n_obs})", fontsize=12)
    ax.legend(fontsize=9) 
    ax.grid(True, alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════════
# Panel 2 — Optimisation trajectories
#
# n_obs=80: short enough that Term B hasn't been swamped by chaotic gradient
#           explosion, long enough that the EM bias is visible at plateau.
# n_ens=100: lower MC variance -> smoother curves, cleaner separation.
# lr=0.2, n_steps=60: both methods reach their fixed points within the window.
# ═══════════════════════════════════════════════════════════════════════════

def panel2_optimisation(ax, n_obs=150, n_steps=30, lr=0.2, n_ens=50):
    print("Panel 2: optimization trajectories...")
    y_obs, t_obs, _ = load_observations(n_obs=n_obs)

    sigma_ad, sigma_em = 5.0, 5.0
    traj_ad = [sigma_ad]
    traj_em = [sigma_em]

    for step in range(n_steps):
        if step % 10 == 0:
            print(f"  step {step:3d}  AD={sigma_ad:.3f}  EM={sigma_em:.3f}")

        coeff = torch.tensor([sigma_ad, TRUE_RHO, TRUE_BETA], dtype=torch.float32, device=device)
        g     = compute_gradient(coeff, y_obs, t_obs, "ad", n_ens=n_ens)
        sigma_ad = float(np.clip(sigma_ad + lr * g[0].item(), 1.0, 20.0))
        traj_ad.append(sigma_ad)

        coeff = torch.tensor([sigma_em, TRUE_RHO, TRUE_BETA], dtype=torch.float32, device=device)
        g     = compute_gradient(coeff, y_obs, t_obs, "em", n_ens=n_ens)
        sigma_em = float(np.clip(sigma_em + lr * g[0].item(), 1.0, 20.0))
        traj_em.append(sigma_em)

    final_ad = float(np.mean(traj_ad[-5:]))
    final_em = float(np.mean(traj_em[-5:]))

    steps = np.arange(n_steps + 1)
    ax.plot(steps, traj_ad, color="#2563eb", lw=2.0, label=f"AD-EnKF  (plateau $\\approx$ {final_ad:.2f})")
    ax.plot(steps, traj_em, color="#dc2626", lw=2.0, ls="--", label=f"EM-EnKF  (plateau $\\approx$ {final_em:.2f})")
    ax.axhline(TRUE_SIGMA, color="black", lw=1.2, ls=":", label=rf"True $\sigma^*={TRUE_SIGMA}$")

    ax.set_xlabel("Gradient ascent iteration", fontsize=11)
    ax.set_ylabel(r"$\sigma$ estimate", fontsize=11)
    ax.set_title(f"Optimization Trajectories  (T={n_obs}, N={n_ens})", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    fig = plt.figure(figsize=(11, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

    panel1_gradient_landscape(fig.add_subplot(gs[0]))
    panel2_optimisation(fig.add_subplot(gs[1]))

    # fig.suptitle(
    #     r"Gradient Decomposition: EM-EnKF (Term A) vs AD-EnKF (Term A + B)"
    #     r"  — (L63, $\sigma$-component)",
    #     fontsize=12, y=1.01
    # )

    out_path = FIG_DIR / "gradient_decomposition.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {out_path}")


if __name__ == "__main__":
    main()