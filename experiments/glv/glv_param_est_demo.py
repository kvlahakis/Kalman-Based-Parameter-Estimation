"""
AD-EnKF Parameter Estimation for gLV
=====================================
Uses the auto-differentiable EnKF (torchEnKF) to learn the parameters of the
5-species generalized Lotka-Volterra system from noisy observations.

Produces:
    glv_estimated_params.py   — R_EST, A_EST in glv_data_generator.py format
    mse_history_full_obs.npy  — MSE convergence (or mse_history_part_obs.npy)

Figures (saved to --out):
    adenkf_apex_traj_*.png/pdf         — apex predator trajectory (Fig 6/7 report)
    adenkf_param_errors_*.png/pdf      — absolute parameter errors  (Fig 8)
    adenkf_mse_history_*.png/pdf       — MSE convergence            (Fig 8b)

Usage
-----
    # From repo root:
    python experiments/glv/glv_param_est_demo.py
    python experiments/glv/glv_param_est_demo.py --a_hidden 0.0 --observe_all
    python experiments/glv/glv_param_est_demo.py --partial_obs
    python experiments/glv/glv_param_est_demo.py --out experiments/glv/figures/adenkf
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure repo root is importable regardless of invocation location.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from torchEnKF import da_methods, nn_templates, noise
from paths import GLV_DATA_DIR


# ---------------------------------------------------------------------------
# Differentiable gLV ODE module
# ---------------------------------------------------------------------------

class gLV_Net(nn.Module):
    """
    PyTorch auto-differentiable Generalized Lotka-Volterra ODE.

        dx_i/dt = x_i * (r_i + sum_j A_ij * x_j)

    Only the non-zero entries of A (as defined by the filter's sparsity mask)
    are learnable parameters; the sparsity pattern is fixed throughout training.
    """

    def __init__(self, mask_A: np.ndarray, x_dim: int = 5):
        super().__init__()
        self.x_dim    = x_dim
        self.mask_A   = torch.tensor(mask_A, dtype=torch.bool)
        self.r        = nn.Parameter(torch.zeros(x_dim))
        self.A_nonzero = nn.Parameter(torch.zeros(int(mask_A.sum())))

    def forward(self, t, x):
        A = torch.zeros((self.x_dim, self.x_dim),
                        device=x.device, dtype=x.dtype)
        A[self.mask_A.to(x.device)] = self.A_nonzero

        # ReLU enforces non-negative populations for the EnKF Gaussian approx.
        x_safe    = torch.relu(x)
        interaction = x_safe @ A.t()
        return x_safe * (self.r + interaction)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_adenkf(
    a_hidden: float = 0.0,
    observe_all: bool = True,
    data_dir: str | None = None,
    out_dir: str = "figures/adenkf",
    n_epochs: int = 60,
    N_ensem: int = 50,
    chunk_len: int = 10,
    lr: float = 2e-2,
    verbose: bool = True,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────
    if data_dir is None:
        data_dir = str(GLV_DATA_DIR)

    label = f"ahidden{a_hidden:.2f}".replace(".", "p")
    truth = np.load(os.path.join(data_dir, f"glv_{label}_truth.npz"))
    obs   = np.load(os.path.join(data_dir, f"glv_{label}_obs.npz"))

    A_base    = truth["A_base"]
    mask_A    = (A_base != 0.0)
    x_dim     = 5
    obs_species = list(obs["observed_species"])

    if observe_all:
        # Use all 5 species for full-obs run
        H_np    = np.eye(x_dim, dtype=np.float32)
        Y_np    = truth["X"].astype(np.float32)  # (5, T) noiseless truth
        # Add obs noise matching generator default
        rng     = np.random.default_rng(42)
        obs_noise_std = float(obs["obs_noise_std"])
        Y_np    = np.maximum(Y_np + rng.normal(0, obs_noise_std, Y_np.shape), 0.0)
    else:
        H_np           = obs["H"].astype(np.float32)
        Y_np           = obs["Y"].astype(np.float32)
        obs_noise_std  = float(obs["obs_noise_std"])

    y_dim = H_np.shape[0]

    # Tensors — AD-EnKF expects (n_timepoints, batch_dim=1, y_dim)
    t_grid       = torch.tensor(truth["t"], dtype=torch.float32, device=device)
    Y_tensor     = torch.tensor(Y_np, dtype=torch.float32, device=device).T.unsqueeze(1)
    t0           = t_grid[0].item()
    t_obs        = t_grid[1:]
    y_obs_fit    = Y_tensor[1:]
    H_true       = torch.tensor(H_np, dtype=torch.float32, device=device)
    theta_true   = torch.tensor(truth["theta"], dtype=torch.float32, device=device)

    # ── Model and filter setup ─────────────────────────────────────────────
    learned_ode_func = gLV_Net(mask_A, x_dim=x_dim).to(device)
    true_obs_func    = nn_templates.Linear(x_dim, y_dim, H=H_true).to(device)
    noise_R_true     = noise.AddGaussian(
        y_dim, torch.tensor(obs_noise_std ** 2), param_type="scalar").to(device)

    init_m       = torch.tensor(truth["x0"], dtype=torch.float32, device=device)
    init_C_param = noise.AddGaussian(
        x_dim, 0.5 * torch.eye(x_dim), "full").to(device)
    init_Q       = 0.05 * torch.ones(x_dim)
    learned_Q    = noise.AddGaussian(x_dim, init_Q, "diag").to(device)

    optimizer = torch.optim.Adam([
        {"params": learned_ode_func.parameters(), "lr": lr},
        {"params": learned_Q.parameters(),        "lr": lr},
    ])

    # ── Training loop ─────────────────────────────────────────────────────
    n_steps     = len(t_obs)
    mse_history = []

    if verbose:
        print("Starting AD-EnKF Parameter Estimation for gLV...")

    for epoch in range(n_epochs):
        train_log_likelihood = torch.tensor(0., device=device)
        t_start = t0
        X = init_C_param(init_m.expand(1, N_ensem, x_dim))

        for start in range(0, n_steps, chunk_len):
            optimizer.zero_grad()
            end = min(start + chunk_len, n_steps)

            X, _, log_likelihood = da_methods.EnKF(
                ode_func       = learned_ode_func,
                obs_func       = true_obs_func,
                t_obs          = t_obs[start:end],
                y_obs          = y_obs_fit[start:end],
                N_ensem        = N_ensem,
                init_m         = init_m,
                init_C_param   = init_C_param,
                model_Q_param  = learned_Q,
                noise_R_param  = noise_R_true,
                device         = device,
                save_filter_step = {},
                t0             = t_start,
                init_X         = X,
                ode_options    = dict(step_size=0.1),
                adjoint_options= dict(step_size=0.1),
                linear_obs     = True,
            )
            t_start = t_obs[end - 1].item()
            (-log_likelihood).mean().backward()
            train_log_likelihood += log_likelihood.detach().mean()
            optimizer.step()

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            theta_hat = torch.cat(
                [learned_ode_func.r, learned_ode_func.A_nonzero])
            param_mse = torch.nn.functional.mse_loss(theta_hat, theta_true)
            mse_history.append(param_mse.item())
            if verbose:
                print(
                    f"Epoch {epoch:03d} | LL: {train_log_likelihood.item():.2f} "
                    f"| Param MSE: {param_mse.item():.4f}"
                )

    # ── Export estimated parameters ────────────────────────────────────────
    r_est_np = learned_ode_func.r.detach().cpu().numpy()
    A_est_full = torch.zeros((x_dim, x_dim))
    A_est_full[mask_A] = learned_ode_func.A_nonzero.detach().cpu()
    A_est_np = A_est_full.numpy()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    est_file = out_path / "glv_estimated_params.py"
    with open(est_file, "w") as f:
        f.write('"""Auto-generated parameter estimates from AD-EnKF"""\n')
        f.write("import numpy as np\n\n")
        f.write("# Intrinsic growth rates\n")
        f.write(f"R_EST = np.array({repr(r_est_np.tolist())})\n\n")
        f.write("# Interaction matrix\n")
        f.write(f"A_EST = np.array({repr(A_est_np.tolist())})\n")
    if verbose:
        print(f"Exported estimated parameters to {est_file}")

    # ── Save MSE history ───────────────────────────────────────────────────
    mse_array = np.array(mse_history)
    obs_tag   = "full_obs" if observe_all else "part_obs"
    npy_path  = out_path / f"mse_history_{obs_tag}.npy"
    np.save(npy_path, mse_array)
    if verbose:
        print(f"Saved MSE history to {npy_path}")

    # ── Figures ────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    # Helper: integrate noiseless ODE with estimated params
    def _integrate(r, A, x0, t_start, t_end, dt=0.5):
        t_eval = np.arange(t_start, t_end + 0.5 * dt, dt)
        sol = solve_ivp(
            lambda t, x: np.maximum(x, 0.0) * (r + A @ np.maximum(x, 0.0)),
            [t_start, t_end], x0, method="RK45",
            rtol=1e-8, atol=1e-10, t_eval=t_eval,
        )
        return sol.t, np.maximum(sol.y, 0.0)

    t_np   = truth["t"]
    X_true = truth["X"]
    x0_np  = truth["x0"]
    r_true = truth["r"]
    A_true = truth["A"]

    t_hat, X_hat = _integrate(r_est_np, A_est_np, x0_np, t_np[0], t_np[-1])

    SPECIES_NAMES  = ["Producer 1", "Producer 2", "Herbivore 1",
                      "Herbivore 2", "Apex Predator"]
    SPECIES_COLORS = ["#1a7a1a", "#7fd67f", "#1a3a8f", "#63b3ed", "#e74c3c"]

    # -- Apex predator trajectory (report Fig 6) --
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_np,   X_true[4], color=SPECIES_COLORS[4], lw=1.8, label="True")
    # interpolate X_hat to t_np for fair comparison
    from numpy import interp
    ax.plot(t_np, interp(t_np, t_hat, X_hat[4]),
            color=SPECIES_COLORS[4], lw=1.8, ls="--", label="AD-EnKF estimate")
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Population", fontsize=10)
    ax.set_title(f"Apex Predator Trajectory  AD-EnKF  [{obs_tag}]", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_path / f"adenkf_apex_traj_{obs_tag}.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close(fig)

    # -- Absolute parameter errors (report Fig 8) --
    # Load true theta from npz (already in truth file)
    theta_true_np = truth["theta"]
    labels_np     = [s.decode() if isinstance(s, bytes) else str(s)
                     for s in truth["theta_labels"]]
    theta_hat_np  = np.concatenate([r_est_np,
                                     A_est_np[mask_A]])
    errors        = np.abs(theta_hat_np - theta_true_np)
    bar_colors    = ["#c0392b" if e > 0.05 else "#7f8c8d" for e in errors]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(np.arange(len(errors)), errors, color=bar_colors, alpha=0.85)
    ax.axhline(0.05, color="#c0392b", lw=1.2, ls="--", label="0.05 tolerance")
    ax.set_xticks(np.arange(len(labels_np)))
    ax.set_xticklabels(labels_np, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(r"$|\hat{\theta} - \theta_{\mathrm{true}}|$", fontsize=10)
    ax.set_title(f"AD-EnKF absolute parameter error  [{obs_tag}]", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_path / f"adenkf_param_errors_{obs_tag}.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close(fig)

    # -- MSE convergence (report Fig 8b) --
    epochs_logged = list(range(0, n_epochs, 5)) + ([n_epochs - 1]
                                                    if n_epochs - 1 not in
                                                    range(0, n_epochs, 5) else [])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(epochs_logged[:len(mse_array)], mse_array,
                lw=2, marker="o", ms=4, markevery=1, color="#2563eb")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Parameter MSE (log scale)", fontsize=11)
    ax.set_title(f"AD-EnKF parameter convergence  [{obs_tag}]", fontsize=12)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_path / f"adenkf_mse_history_{obs_tag}.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close(fig)

    if verbose:
        print(f"Figures saved to {out_path}")
        print("Done.")

    return dict(
        r_est=r_est_np, A_est=A_est_np,
        mse_history=mse_array,
        theta_hat=theta_hat_np, theta_true=theta_true_np,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AD-EnKF parameter estimation for the gLV system."
    )
    parser.add_argument("--a_hidden",    type=float, default=0.0,
                        help="Hidden interaction strength (0.0 = well-specified).")
    parser.add_argument("--observe_all", action="store_true",
                        help="Use all 5 species as observations (full-obs run).")
    parser.add_argument("--partial_obs", action="store_true",
                        help="Use partial observations from the data file "
                             "(default: P1, P2, H3 only).")
    parser.add_argument("--data_dir",    type=str, default=None,
                        help="Directory containing glv_ahidden*.npz files. "
                             "Defaults to Data/gLV/data/ relative to repo root.")
    parser.add_argument("--out",         type=str,
                        default=str(Path(__file__).resolve().parent / "figures" / "adenkf"),
                        help="Output directory for figures and estimates.")
    parser.add_argument("--epochs",      type=int, default=60)
    parser.add_argument("--n_ensemble",  type=int, default=50)
    parser.add_argument("--chunk_len",   type=int, default=10)
    parser.add_argument("--lr",          type=float, default=2e-2)
    parser.add_argument("--quiet",       action="store_true")
    args = parser.parse_args()

    # --observe_all takes priority; --partial_obs uses file's obs spec
    observe_all = args.observe_all or not args.partial_obs

    run_adenkf(
        a_hidden   = args.a_hidden,
        observe_all= observe_all,
        data_dir   = args.data_dir,
        out_dir    = args.out,
        n_epochs   = args.epochs,
        N_ensem    = args.n_ensemble,
        chunk_len  = args.chunk_len,
        lr         = args.lr,
        verbose    = not args.quiet,
    )


if __name__ == "__main__":
    main()
