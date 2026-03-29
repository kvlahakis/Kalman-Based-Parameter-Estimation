"""
gradient_decomposition_lg_run.py
=================================
Demonstrates Term A vs Term A+B gradient decomposition on a 2D Linear Gaussian
model, comparing two gradient estimators:

  AD-EnKF  — uses the official torchEnKF implementation (da_methods.EnKF, adjoint=False).
             Full backprop through the entire filter graph → Term A + Term B.

  EM-EnKF  — mirrors da_methods.EnKF's internal loop verbatim, but inserts
             X = X.detach() after each Kalman analysis step.
             Severs the particle-history gradient path → Term A only.

NOTE on what this experiment shows:
  In a Linear Gaussian system, Term B vanishes at the likelihood maximum,
  so EM-EnKF and AD-EnKF converge to the SAME fixed point.  The experiment
  therefore demonstrates convergence SPEED, not estimation bias.  AD-EnKF
  converges faster because Term B provides additional gradient signal during
  the ascent, even though it disappears at the optimum.

  To demonstrate estimation bias (EM converging to the wrong value), a
  nonlinear system such as Lorenz-63 is required.

System:
    x_{t+1} = A(θ) x_t + ε_t,   ε_t ~ N(0, q²I)
    y_t      = H x_t + η_t,       η_t ~ N(0, r²I)
    A(θ) = decay × R(freq),   H = [1, 0]  (partial obs)

Run from repo root:
    PYTHONPATH=. python ADEnKF/experiments/gradient_decomposition/gradient_decomposition_lg_run.py
"""

import sys
import math
from pathlib import Path

_script_dir  = Path(__file__).resolve().parent
_ad_enkf_dir = _script_dir.parent.parent
_repo_root   = _ad_enkf_dir.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_ad_enkf_dir))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from torchdiffeq import odeint
from torchEnKF import da_methods, nn_templates, noise
from torchEnKF.da_methods import inv_logdet

torch.manual_seed(42)
np.random.seed(42)

FIG_DIR = _script_dir / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


# ── true system parameters ────────────────────────────────────────────────
TRUE_DECAY = 0.95
TRUE_FREQ  = 0.3
x_dim_lg   = 2
y_dim_lg   = 1      # observe first coordinate only → amplifies Term B
N_ENS      = 200
Q_STD      = 0.3
R_STD      = 1.0

# ── shared torchEnKF modules ─────────────────────────────────────────────
H_lg        = torch.tensor([[1.0, 0.0]], device=device)
obs_func_lg = nn_templates.Linear(x_dim_lg, y_dim_lg, H=H_lg).to(device)

noise_Q_lg  = noise.AddGaussian(
    x_dim_lg, Q_STD * torch.ones(x_dim_lg, device=device), "diag"
).to(device)
noise_R_lg  = noise.AddGaussian(
    y_dim_lg, torch.tensor(R_STD, device=device), "scalar"
).to(device)
init_C_lg   = noise.AddGaussian(
    x_dim_lg, torch.eye(x_dim_lg, device=device), "full"
).to(device)
init_m_lg   = torch.zeros(x_dim_lg, device=device)


class LinearGaussian2D_ODE(nn.Module):
    """
    ODE form of A(θ) = decay × R(freq).
    dx/dt = (A(θ) − I) x  →  one Euler/RK4 step of Δt=1 gives x_{t+1} = A(θ)x_t
    """
    def __init__(self, decay: float, freq: float):
        super().__init__()
        self.decay = nn.Parameter(torch.tensor(float(decay)))
        self.freq  = nn.Parameter(torch.tensor(float(freq)))

    def transition_matrix(self) -> torch.Tensor:
        c = torch.cos(self.freq)
        s = torch.sin(self.freq)
        return self.decay * torch.stack([
            torch.stack([c, -s]),
            torch.stack([s,  c]),
        ])

    def forward(self, t, X: torch.Tensor) -> torch.Tensor:
        A_minus_I = self.transition_matrix() - torch.eye(x_dim_lg, device=X.device)
        return X @ A_minus_I.T


def generate_lg_data(n_obs: int, seed: int = 42):
    torch.manual_seed(seed)
    A = TRUE_DECAY * torch.tensor([
        [math.cos(TRUE_FREQ), -math.sin(TRUE_FREQ)],
        [math.sin(TRUE_FREQ),  math.cos(TRUE_FREQ)],
    ], dtype=torch.float32, device=device)

    x, ys = torch.zeros(x_dim_lg, device=device), []
    for _ in range(n_obs):
        x = A @ x + Q_STD * torch.randn(x_dim_lg, device=device)
        ys.append(H_lg @ x + R_STD * torch.randn(y_dim_lg, device=device))

    y_obs = torch.stack(ys).unsqueeze(1)
    t_obs = torch.arange(1, n_obs + 1, dtype=torch.float32, device=device)
    return y_obs, t_obs


def compute_gradient_lg(decay_val, freq_val, y_obs, t_obs, mode):
    assert mode in ("ad", "em")
    model     = LinearGaussian2D_ODE(decay_val, freq_val).to(device)
    n_obs_val = y_obs.shape[0]

    if mode == "ad":
        _, _, log_lik = da_methods.EnKF(
            model, obs_func_lg, t_obs, y_obs, N_ENS,
            init_m_lg, init_C_lg, noise_Q_lg, noise_R_lg, device,
            save_filter_step={},
            ode_method="rk4", ode_options=dict(step_size=1),
            adjoint=False, linear_obs=False, tqdm=None,
        )
        log_lik = log_lik.mean() / n_obs_val

    else:
        y_dim, bs = y_obs.shape[-1], y_obs.shape[1:-1]
        noise_R_mat    = noise_R_lg.full()
        noise_R_inv    = noise_R_lg.inv()
        logdet_noise_R = noise_R_lg.logdet()

        X       = init_C_lg(init_m_lg.expand(*bs, N_ENS, x_dim_lg))
        log_lik = torch.zeros(bs, device=device)
        t_cur   = 0.0

        for j in range(n_obs_val):
            t_span = torch.tensor([t_cur, t_obs[j].item()], device=device)
            X      = odeint(model, X, t_span, method="rk4", options=dict(step_size=1))[-1]
            t_cur  = t_obs[j].item()
            X      = noise_Q_lg(X)

            X_m, X_ct   = X.mean(dim=-2).unsqueeze(-2), X - X.mean(dim=-2).unsqueeze(-2)
            y_obs_j     = y_obs[j].unsqueeze(-2)
            obs_perturb = noise_R_lg(y_obs_j.expand(*bs, N_ENS, y_dim))

            HX        = obs_func_lg(X)
            HX_m      = HX.mean(dim=-2).unsqueeze(-2)
            C_ww_sqrt = 1.0 / math.sqrt(N_ENS - 1) * (HX - HX_m)

            v = torch.cat([obs_perturb - HX, y_obs_j - HX_m], dim=-2)
            C_ww_R_invv, C_ww_R_logdet = inv_logdet(
                v, C_ww_sqrt, noise_R_mat, noise_R_inv, logdet_noise_R)

            log_lik += (
                -0.5 * (y_dim * math.log(2.0 * math.pi) + C_ww_R_logdet)
                + (-0.5 * (C_ww_R_invv[..., N_ENS:, :]
                           @ (y_obs_j - HX_m).transpose(-1, -2))).squeeze(-1).squeeze(-1)
            )

            pre = C_ww_R_invv[..., :N_ENS, :]
            X   = X + (1.0 / math.sqrt(N_ENS - 1)
                       * (pre @ C_ww_sqrt.transpose(-1, -2)) @ X_ct)
            X   = X.detach()   # ← severs Term B

        log_lik = log_lik.mean() / n_obs_val

    log_lik.backward()
    return torch.stack([model.decay.grad.clone(), model.freq.grad.clone()])


# ═══════════════════════════════════════════════════════════════════════════
# Panel 1 — Gradient landscape
# ═══════════════════════════════════════════════════════════════════════════

def panel1_gradient_landscape(ax, n_obs=100):
    print("Panel 1: gradient landscape...")
    y_obs, t_obs = generate_lg_data(n_obs)

    decay_vals = np.linspace(0.70, 1.05, 30)
    ad_grads, em_grads = [], []
    for dv in decay_vals:
        ad_grads.append(compute_gradient_lg(dv, TRUE_FREQ, y_obs, t_obs, "ad")[0].item())
        em_grads.append(compute_gradient_lg(dv, TRUE_FREQ, y_obs, t_obs, "em")[0].item())

    ad_grads, em_grads = np.array(ad_grads), np.array(em_grads)

    ax.plot(decay_vals, ad_grads, color="#2563eb", lw=2.0, label="AD-EnKF  (Term A+B)")
    ax.plot(decay_vals, em_grads, color="#dc2626", lw=2.0, ls="--", label="EM-EnKF  (Term A only)")
    ax.fill_between(decay_vals, em_grads, ad_grads,
                    alpha=0.15, color="#7c3aed", label="Term B  (missing from EM)")
    ax.axvline(TRUE_DECAY, color="black", lw=1.2, ls=":",
               label=rf"True decay$^* = {TRUE_DECAY}$")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel(r"decay  $(\theta)$", fontsize=11)
    ax.set_ylabel(r"$\nabla_{\theta}\,\ell^{\mathrm{EnKF}}$", fontsize=11)
    ax.set_title(f"Panel 1: Gradient Landscape  (T={n_obs})", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════════
# Panel 2 — Convergence speed
# ═══════════════════════════════════════════════════════════════════════════

def panel2_optimisation(ax, n_obs=100, n_steps=150, lr=0.01):
    print("Optimization trajectories...")
    y_obs, t_obs = generate_lg_data(n_obs)

    decay_ad, decay_em = 0.70, 0.70
    traj_ad, traj_em   = [decay_ad], [decay_em]

    for step in range(n_steps):
        if step % 30 == 0:
            print(f"  step {step:3d}  AD={decay_ad:.3f}  EM={decay_em:.3f}")

        g_ad = compute_gradient_lg(decay_ad, TRUE_FREQ, y_obs, t_obs, "ad")
        decay_ad = float(np.clip(decay_ad + lr * g_ad[0].item(), 0.01, 1.30))
        traj_ad.append(decay_ad)

        g_em = compute_gradient_lg(decay_em, TRUE_FREQ, y_obs, t_obs, "em")
        decay_em = float(np.clip(decay_em + lr * g_em[0].item(), 0.01, 1.30))
        traj_em.append(decay_em)

    plateau    = float(np.mean(traj_ad[-10:]))   # both converge here
    thresh     = 0.005
    conv_ad    = next((i for i, v in enumerate(traj_ad) if abs(v - plateau) < thresh), n_steps)
    conv_em    = next((i for i, v in enumerate(traj_em) if abs(v - plateau) < thresh), n_steps)

    steps = np.arange(n_steps + 1)
    ax.plot(steps, traj_ad, color="#2563eb", lw=2.0, label="AD-EnKF")
    ax.plot(steps, traj_em, color="#dc2626", lw=2.0, ls="--", label="EM-EnKF")
    ax.axhline(TRUE_DECAY, color="black", lw=1.2, ls=":",
               label=rf"True decay$^* = {TRUE_DECAY}$")

    for conv, color, label in [(conv_ad, "#2563eb", "AD"), (conv_em, "#dc2626", "EM")]:
        if conv < n_steps:
            ax.axvline(conv, color=color, lw=1.0, ls="--", alpha=0.5)
            ax.text(conv + 1, 0.715 if label == "AD" else 0.73,
                    f"{label}: step {conv}", color=color, fontsize=8)

    ax.set_xlabel("Gradient ascent iteration", fontsize=11)
    ax.set_ylabel(r"decay estimate  $(\hat\theta)$", fontsize=11)
    ax.set_title(
        f"Panel 2: Convergence Speed  (T={n_obs})\n"
        "Both converge to same value — AD-EnKF arrives faster",
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    fig = plt.figure(figsize=(11, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.42)

    panel1_gradient_landscape(fig.add_subplot(gs[0]))
    panel2_optimisation(fig.add_subplot(gs[1]))

    fig.suptitle(
        r"Gradient Decomposition: EM-EnKF (Term A) vs AD-EnKF (Term A+B)"
        r"  —  2D Linear Gaussian  (convergence speed, not bias)",
        fontsize=11, y=1.02,
    )

    out_path = FIG_DIR / "gradient_decomposition_lg.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {out_path}")
    return out_path


if __name__ == "__main__":
    main()