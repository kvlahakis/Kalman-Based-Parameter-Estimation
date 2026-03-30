"""
ES-MDA for gLV Parameter Estimation and State Reconstruction
=============================================================
Ensemble Smoother with Multiple Data Assimilation (ES-MDA) applied to the
gLV system from glv_data_generator.py.

ES-MDA (Emerick & Reynolds, 2013) is a batch smoother: it updates the FULL
trajectory (all time steps at once) rather than filtering sequentially.

Usage
-----
    # From repo root:
    python experiments/glv/glv_esmda.py
    python experiments/glv/glv_esmda.py --partial_obs
    python experiments/glv/glv_esmda.py --a_hidden 0.2
    python experiments/glv/glv_esmda.py --compare
"""

import argparse
import os
import sys
from pathlib import Path

# Repo root on path so `paths` and `Data/gLV` are importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "Data" / "gLV"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

from glv_data_generator import (
    R_TRUE, A_TRUE, EPS,
    get_true_A, get_theta, glv_rhs,
    generate_experiment, OBSERVED_SPECIES, N_SPECIES,
)
from paths import GLV_DATA_DIR

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

SPECIES_NAMES  = ["Producer 1", "Producer 2", "Herbivore 1",
                  "Herbivore 2", "Apex Predator"]
SPECIES_COLORS = ["#1a7a1a", "#7fd67f", "#1a3a8f", "#63b3ed", "#e74c3c"]
COLOR_TRUE = "#2c3e50"
COLOR_EST  = "#e67e22"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "figure.dpi": 150,
})


# ---------------------------------------------------------------------------
# Parameter <-> matrix helpers
# ---------------------------------------------------------------------------

def theta_to_rA(theta):
    """Split theta (15,) into r (5,) and A (5,5) using A_TRUE's sparsity."""
    r    = theta[:5].copy()
    A    = np.zeros((5, 5))
    mask = (A_TRUE != 0.0)
    nz   = [(i, j) for i in range(5) for j in range(5) if mask[i, j]]
    for k, (i, j) in enumerate(nz):
        A[i, j] = theta[5 + k]
    return r, A


def integrate_member(theta, x0, t_grid):
    """Integrate gLV ODE for one ensemble member over t_grid. Returns X (5,T)."""
    r, A = theta_to_rA(theta)
    T    = len(t_grid)
    X    = np.zeros((5, T))
    X[:, 0] = x0
    x_cur = x0.copy()
    for k in range(1, T):
        sol = solve_ivp(
            glv_rhs, [t_grid[k - 1], t_grid[k]], x_cur,
            args=(r, A), method="RK45", rtol=1e-6, atol=1e-8,
        )
        if not sol.success or np.any(sol.y[:, -1] > 1e6) or np.any(
                np.isnan(sol.y[:, -1])):
            X[:, k:] = np.nan
            return X
        x_cur    = np.maximum(sol.y[:, -1], 1e-10)
        X[:, k]  = x_cur
    return X


# ---------------------------------------------------------------------------
# ES-MDA core
# ---------------------------------------------------------------------------

def run_esmda(t_grid, Y_obs, H, x0, a_hidden=0.0,
              n_ensemble=150, n_iterations=8,
              obs_noise_std=0.05, verbose=True):
    """
    ES-MDA for joint parameter + state estimation.

    Returns
    -------
    theta_mean   : (15,)      posterior mean parameters
    theta_ens    : (Ne, 15)   posterior ensemble
    X_mean       : (5, T)     posterior mean trajectory
    X_ens        : (Ne, 5, T) posterior ensemble trajectories
    rmse_history : (N_a,)     parameter RMSE per iteration
    """
    T  = len(t_grid)
    Ne = n_ensemble
    Na = n_iterations
    M  = H.shape[0]

    alphas = [float(Na)] * Na

    A_true_full        = get_true_A(a_hidden)
    theta_true, _      = get_theta(A_true_full)
    theta_dim          = len(theta_true)

    rng = np.random.default_rng(0)
    theta_prior, _ = get_theta(A_TRUE)
    theta_ens = np.zeros((Ne, theta_dim))
    for k in range(theta_dim):
        std = max(abs(theta_prior[k]) * 0.30, 0.05)
        theta_ens[:, k] = theta_prior[k] + rng.normal(0, std, size=Ne)

    rmse_history = []

    for iteration in range(Na):
        alpha = alphas[iteration]
        C_D   = (alpha * obs_noise_std ** 2) * np.eye(M)

        if verbose:
            rmse = np.sqrt(np.mean((theta_ens.mean(axis=0) - theta_true) ** 2))
            rmse_history.append(rmse)
            print(f"  Iteration {iteration+1}/{Na}  prior RMSE = {rmse:.4f}")

        G_ens     = np.zeros((Ne, M * T))
        X_ens_cur = np.zeros((Ne, 5, T))
        n_failed  = 0

        for j in range(Ne):
            X_j = integrate_member(theta_ens[j], x0, t_grid)
            X_ens_cur[j] = X_j
            pred_obs     = H @ X_j
            G_ens[j]     = pred_obs.ravel()
            if np.any(np.isnan(X_j)):
                n_failed += 1

        if n_failed > Ne // 2:
            print(f"  WARNING: {n_failed}/{Ne} ensemble members blew up. "
                  "Consider reducing prior spread.")

        nan_mask = np.isnan(G_ens)
        if nan_mask.any():
            col_means = np.nanmean(G_ens, axis=0)
            G_ens     = np.where(nan_mask, col_means, G_ens)

        # ── Woodbury ensemble-space update ─────────────────────────────────
        d_obs_flat = Y_obs.ravel()
        D_ens = (d_obs_flat[None, :]
                 + np.sqrt(alpha) * obs_noise_std
                 * rng.standard_normal((Ne, M * T)))

        theta_bar = theta_ens.mean(axis=0)
        G_bar     = G_ens.mean(axis=0)
        dTheta    = theta_ens - theta_bar
        dG        = G_ens     - G_bar

        reg = (Ne - 1) * alpha * obs_noise_std ** 2
        S   = dG @ dG.T + reg * np.eye(Ne)
        W   = np.linalg.solve(S, dG)
        innov = D_ens - G_ens
        theta_ens += (innov @ W.T) @ dTheta / (Ne - 1)

        if verbose:
            rmse_post = np.sqrt(
                np.mean((theta_ens.mean(axis=0) - theta_true) ** 2))
            print(f"             post RMSE = {rmse_post:.4f}")

    rmse_final = np.sqrt(np.mean((theta_ens.mean(axis=0) - theta_true) ** 2))
    rmse_history.append(rmse_final)

    if verbose:
        print("  Final forward run...")
    X_ens_final = np.zeros((Ne, 5, T))
    for j in range(Ne):
        X_ens_final[j] = integrate_member(theta_ens[j], x0, t_grid)

    theta_mean = theta_ens.mean(axis=0)
    X_mean     = np.nanmean(X_ens_final, axis=0)

    return theta_mean, theta_ens, X_mean, X_ens_final, np.array(rmse_history)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save(fig, name, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ["png", "pdf"]:
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_parameter_errors(theta_mean, a_hidden, full_obs, out_dir="."):
    A_true_full        = get_true_A(a_hidden)
    theta_true, labels = get_theta(A_true_full)
    errors   = np.abs(theta_mean - theta_true)
    obs_tag  = "full_obs" if full_obs else "part_obs"
    obs_lbl  = "[full obs]" if full_obs else "[partial obs]"
    c_est    = COLOR_EST if full_obs else "#17a589"
    bar_colors = [c_est if e > 0.05 else "#95a5a6" for e in errors]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(np.arange(len(errors)), errors, color=bar_colors, alpha=0.85)
    ax.axhline(0.05, color="#c0392b", lw=1.2, ls="--", label="0.05 tolerance")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(r"$|\hat{\theta} - \theta_{\mathrm{true}}|$", fontsize=10)
    ax.set_title(f"ES-MDA parameter error  {obs_lbl}", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, f"esmda_param_errors_{obs_tag}", out_dir)


def plot_trajectories(t, X_true, X_ens, Y_obs, H, full_obs,
                      out_dir=".", show_obs=False):
    obs_idx   = np.where(H.sum(axis=0) > 0)[0]
    unobs_idx = np.array([i for i in range(5) if i not in set(obs_idx.tolist())])
    obs_tag   = "full_obs" if full_obs else "part_obs"
    obs_lbl   = "[full obs]" if full_obs else "[partial obs]"

    mse_per   = np.nanmean((X_ens - X_true[None, :, :]) ** 2, axis=(1, 2))
    best_j    = int(np.argmin(mse_per))
    X_best    = X_ens[best_j]

    if full_obs or len(unobs_idx) == 0:
        groups = [([0, 1], "producers"), ([2, 3], "herbivores"), ([4], "apex")]
    else:
        groups = [(list(obs_idx), "observed"), (list(unobs_idx), "unobserved")]

    for species_group, group_name in groups:
        fig, ax = plt.subplots(figsize=(9, 4))
        for i in species_group:
            c = SPECIES_COLORS[i]
            ax.plot(t, X_true[i], color=c, lw=1.8, alpha=0.9)
            ax.plot(t, X_best[i], color=c, lw=1.8, ls="--", alpha=0.9)
            if show_obs and i in obs_idx:
                k = np.where(obs_idx == i)[0][0]
                ax.scatter(t[::2], Y_obs[k, ::2], color=c, s=8, alpha=0.55, zorder=3)
        handles = [
            Line2D([0], [0], color=SPECIES_COLORS[i], lw=2,
                   label=SPECIES_NAMES[i]) for i in species_group
        ] + [
            Line2D([0], [0], color="gray", lw=2, label="True trajectory"),
            Line2D([0], [0], color="gray", lw=2, ls="--",
                   label="ES-MDA best member"),
        ]
        ax.legend(handles=handles, fontsize=8)
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel("Population", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.set_title(f"ES-MDA trajectory reconstruction  {obs_lbl}", fontsize=11)
        plt.tight_layout()
        _save(fig, f"esmda_traj_{group_name}_{obs_tag}", out_dir)


def plot_rmse(rmse_history, full_obs, out_dir="."):
    obs_tag = "full_obs" if full_obs else "part_obs"
    obs_lbl = "[full obs]" if full_obs else "[partial obs]"
    color   = "#1a7a1a" if full_obs else "#1a3a8f"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.arange(len(rmse_history)), rmse_history,
                lw=2, color=color, marker="o", ms=5)
    ax.set_xlabel("ES-MDA iteration", fontsize=11)
    ax.set_ylabel("Parameter RMSE (log scale)", fontsize=11)
    ax.set_title(f"ES-MDA convergence  {obs_lbl}", fontsize=12)
    plt.tight_layout()
    _save(fig, f"esmda_rmse_{obs_tag}", out_dir)


def plot_comparison(results_full, results_part, out_dir="."):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(results_full["rmse"], lw=2, color="#1a7a1a",
                marker="o", ms=5, label="Full observation")
    ax.semilogy(results_part["rmse"], lw=2, color="#1a3a8f",
                marker="o", ms=5, label="Partial observation (P1, P2, H1)")
    ax.set_xlabel("ES-MDA iteration", fontsize=11)
    ax.set_ylabel("Parameter RMSE (log scale)", fontsize=11)
    ax.set_title("ES-MDA: full vs partial observability", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(fig, "esmda_comparison_rmse", out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one(a_hidden, observed_species, n_ensemble, n_iterations,
            data_dir, out_dir, verbose):
    full_obs = (len(observed_species) == 5)
    obs_tag  = "full_obs" if full_obs else "part_obs"
    print(f"\n{'='*60}")
    print(f"ES-MDA  |  a_hidden={a_hidden:.2f}  |  {obs_tag}")
    print(f"  ensemble={n_ensemble}  iterations={n_iterations}")
    print(f"{'='*60}")

    generate_experiment(a_hidden, seed=42, dt=0.5, t_end=100.0,
                        save_dir=data_dir, observed_species=observed_species)

    label = f"ahidden{a_hidden:.2f}".replace(".", "p")
    truth = np.load(os.path.join(data_dir, f"glv_{label}_truth.npz"))
    obs   = np.load(os.path.join(data_dir, f"glv_{label}_obs.npz"))

    t_grid = truth["t"]
    X_true = truth["X"]
    x0     = truth["x0"]
    Y_obs  = obs["Y"]
    H      = obs["H"]

    print(f"\n  Running ES-MDA...")
    theta_mean, theta_ens, X_mean, X_ens, rmse_history = run_esmda(
        t_grid, Y_obs, H, x0,
        a_hidden=a_hidden, n_ensemble=n_ensemble,
        n_iterations=n_iterations, obs_noise_std=0.05, verbose=verbose,
    )

    A_true_full        = get_true_A(a_hidden)
    theta_true, labels = get_theta(A_true_full)
    print(f"\n  Final parameter RMSE: {rmse_history[-1]:.4f}")
    print(f"\n  Generating plots -> {out_dir}")
    plot_parameter_errors(theta_mean, a_hidden, full_obs, out_dir)
    plot_trajectories(t_grid, X_true, X_ens, Y_obs, H, full_obs, out_dir)
    plot_rmse(rmse_history, full_obs, out_dir)

    return dict(theta_mean=theta_mean, theta_ens=theta_ens,
                X_mean=X_mean, X_ens=X_ens,
                rmse=rmse_history, t=t_grid, X_true=X_true)


def main():
    parser = argparse.ArgumentParser(
        description="ES-MDA for gLV parameter estimation and state reconstruction."
    )
    parser.add_argument("--a_hidden",     type=float, default=0.0)
    parser.add_argument("--partial_obs",  action="store_true")
    parser.add_argument("--compare",      action="store_true",
                        help="Run both full and partial obs and compare.")
    parser.add_argument("--n_ensemble",   type=int,   default=150)
    parser.add_argument("--n_iterations", type=int,   default=8)
    parser.add_argument("--data_dir",     type=str,   default=None,
                        help="Data directory (default: Data/gLV/data/).")
    parser.add_argument("--out",          type=str,   default=None,
                        help="Output directory for figures.")
    parser.add_argument("--quiet",        action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir or str(GLV_DATA_DIR)
    out_dir  = args.out or str(
        Path(__file__).resolve().parent / "figures" / "esmda")

    obs_full = list(range(N_SPECIES))
    obs_part = OBSERVED_SPECIES

    if args.compare:
        res_full = run_one(args.a_hidden, obs_full,
                           args.n_ensemble, args.n_iterations,
                           data_dir, out_dir, not args.quiet)
        res_part = run_one(args.a_hidden, obs_part,
                           args.n_ensemble, args.n_iterations,
                           data_dir, out_dir, not args.quiet)
        plot_comparison(res_full, res_part, out_dir=out_dir)
    else:
        observed = obs_part if args.partial_obs else obs_full
        run_one(args.a_hidden, observed,
                args.n_ensemble, args.n_iterations,
                data_dir, out_dir, not args.quiet)


if __name__ == "__main__":
    main()
