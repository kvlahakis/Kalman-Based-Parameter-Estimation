"""
Iterative Ensemble Kalman Smoother (IEnKS) for gLV Parameter Estimation
========================================================================

Algorithm: Bocquet & Sakov (2014) "An iterative ensemble Kalman smoother"
           Q. J. R. Meteorol. Soc. 140, 1521-1535.

Key idea
--------
The standard EnKF augments the state with parameters and runs a single
forward pass.  This works when dynamics are near-linear over the assimilation
window, but fails for sensitive oscillatory systems.

IEnKS fixes this by iterating:
  1. Run the ensemble forward through a window of length L.
  2. Compute an ensemble-based Gauss-Newton step in the space of initial
     conditions (and parameters).
  3. Re-centre the ensemble around the updated mean and repeat.

Implementation: bundle variant (Bocquet & Sakov §3).  No adjoint required.

Usage
-----
    # From repo root:
    python experiments/glv/glv_ienks.py
    python experiments/glv/glv_ienks.py --a_hidden 0.2
    python experiments/glv/glv_ienks.py --partial_obs
    python experiments/glv/glv_ienks.py --window 10 --n_ensemble 60 --n_iter 6
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Repo root on path so `paths` is importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

from paths import GLV_DATA_DIR

# ── gLV parameters (mirrors glv_data_generator.py) ──────────────────────────
EPS    = 0.5
R_TRUE = np.array([1.3, 1.1, -0.05, -0.3, -0.2])
A_TRUE = np.array([
    [-0.01,  0.0,   -0.80,  0.0,   0.0 ],
    [ 0.0,   0.0,    0.0,  -0.70,  0.0 ],
    [ 0.60,  0.0,    0.0,   0.0,  -0.25],
    [ 0.0,   0.45,   0.0,   0.0,  -0.2 ],
    [ 0.0,   0.0,    0.15,  0.1,  -0.1 ],
])
N             = 5
SPARSITY_MASK = (A_TRUE != 0.0)
A_ROW, A_COL  = np.where(SPARSITY_MASK)
N_A           = int(SPARSITY_MASK.sum())   # 10 free A entries
N_THETA       = N + N_A                    # 15
N_AUG         = N + N_THETA                # 20

OBSERVED_SPECIES = [0, 1, 2]
SPECIES_COLORS   = ["#2ecc71", "#27ae60", "#3498db", "#2980b9", "#e74c3c"]
SPECIES_NAMES    = ["Producer 1", "Producer 2", "Herbivore 1",
                    "Herbivore 2", "Apex"]


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def get_true_A(a_hidden):
    A = A_TRUE.copy()
    if a_hidden > 0.0:
        A[3, 0] =  a_hidden * EPS
        A[0, 3] = -a_hidden
    return A


def get_theta_true(a_hidden):
    A = get_true_A(a_hidden)
    th = np.concatenate([R_TRUE, A[A_ROW, A_COL]])
    labels = ([f"r_{i+1}" for i in range(N)] +
              [f"a_{r+1}{c+1}" for r, c in zip(A_ROW, A_COL)])
    return th, labels


def unpack_theta(theta):
    """theta (N_THETA,) → r (N,), A (N,N)."""
    r = theta[:N].copy()
    A = np.zeros((N, N))
    A[A_ROW, A_COL] = theta[N:]
    return r, A


# ---------------------------------------------------------------------------
# gLV forward model
# ---------------------------------------------------------------------------

def glv_rhs(x, r, A):
    x = np.maximum(x, 0.0)
    return x * (r + A @ x)


def integrate_glv(x0, t0, t1, r, A):
    sol = solve_ivp(
        lambda t, x: glv_rhs(x, r, A),
        [t0, t1], x0,
        method="RK45", rtol=1e-7, atol=1e-9,
    )
    return np.maximum(sol.y[:, -1], 1e-10)


def propagate_augmented(u, t0, t1):
    """Propagate augmented state u = [x (N,), theta (N_THETA,)]."""
    x     = u[:N]
    theta = u[N:]
    r, A  = unpack_theta(theta)
    x_new = integrate_glv(x, t0, t1, r, A)
    return np.concatenate([x_new, theta])


def propagate_ensemble(E, t0, t1):
    """E : (N_AUG, Ne) → propagated ensemble (N_AUG, Ne)."""
    Ne    = E.shape[1]
    E_new = np.empty_like(E)
    for j in range(Ne):
        try:
            E_new[:, j] = propagate_augmented(E[:, j], t0, t1)
        except Exception:
            E_new[:, j] = E[:, j]
    return E_new


# ---------------------------------------------------------------------------
# IEnKS bundle variant (Bocquet & Sakov 2014, Algorithm 1 + §4.3)
# ---------------------------------------------------------------------------

def ienks_window(E_prior, t_grid_window, Y_window, H, R_obs,
                 n_iter=4, bundle_eps=1e-4):
    """
    One IEnKS assimilation cycle over a window of observations.

    Parameters
    ----------
    E_prior      : (N_AUG, Ne)  prior ensemble at window start
    t_grid_window: (L+1,)       time grid including window start
    Y_window     : (M, L)       observations at t_grid_window[1:]
    H            : (M, N_AUG)   observation operator
    R_obs        : (M, M)       observation error covariance
    n_iter       : int          Gauss-Newton iterations
    bundle_eps   : float        bundle perturbation size

    Returns
    -------
    E_post   : (N_AUG, Ne)  posterior ensemble at window end
    u0_post  : (N_AUG,)     posterior mean at window start
    """
    N_aug, Ne = E_prior.shape
    L         = len(t_grid_window) - 1
    M         = Y_window.shape[0]
    R_inv     = np.linalg.inv(R_obs)

    u_b  = E_prior.mean(axis=1)
    A_b  = (E_prior - u_b[:, None]) / np.sqrt(Ne - 1)

    w = np.zeros(Ne)

    for it in range(n_iter):
        u0 = u_b + A_b @ w

        U_mean = np.empty((N_aug, L + 1))
        U_mean[:, 0] = u0
        for l in range(L):
            U_mean[:, l + 1] = propagate_augmented(
                U_mean[:, l], t_grid_window[l], t_grid_window[l + 1])

        G_list   = []
        inn_list = []

        for l in range(L):
            y_l  = Y_window[:, l]
            Hu_l = H @ U_mean[:, l + 1]
            inn_list.append(y_l - Hu_l)

            G_l = np.empty((M, Ne))
            for j in range(Ne):
                u_pert = u0 + bundle_eps * A_b[:, j]
                u_pert_f = u_pert.copy()
                for ll in range(l + 1):
                    u_pert_f = propagate_augmented(
                        u_pert_f, t_grid_window[ll], t_grid_window[ll + 1])
                G_l[:, j] = (H @ u_pert_f - Hu_l) / bundle_eps
            G_list.append(G_l)

        grad = w.copy()
        HJ   = np.eye(Ne)
        for l in range(L):
            RiG   = R_inv @ G_list[l]
            grad -= G_list[l].T @ (R_inv @ inn_list[l])
            HJ   += G_list[l].T @ RiG

        dw = np.linalg.solve(HJ, -grad)
        w  = w + dw

        if np.linalg.norm(dw) < 1e-6 * (1 + np.linalg.norm(w)):
            break

    u0_post  = u_b + A_b @ w

    HJ_inv   = np.linalg.inv(HJ)
    eigvals, eigvecs = np.linalg.eigh(HJ_inv)
    eigvals  = np.maximum(eigvals, 0.0)
    sqrt_HJi = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    A_post0  = A_b @ sqrt_HJi * np.sqrt(Ne - 1)
    E_post0  = u0_post[:, None] + A_post0

    E_post_end = propagate_ensemble(E_post0,
                                    t_grid_window[0],
                                    t_grid_window[-1])
    return E_post_end, u0_post


# ---------------------------------------------------------------------------
# Full IEnKS run
# ---------------------------------------------------------------------------

def run_ienks(a_hidden=0.0, full_obs=True, window=10, n_ensemble=60,
              n_iter=4, obs_noise_std=0.05, bundle_eps=1e-4,
              data_dir=None, out_dir=None, verbose=True):

    if data_dir is None:
        data_dir = str(GLV_DATA_DIR)
    if out_dir is None:
        out_dir = str(Path(__file__).resolve().parent / "figures" / "ienks")

    label = f"ahidden{a_hidden:.2f}".replace(".", "p")
    truth = np.load(os.path.join(data_dir, f"glv_{label}_truth.npz"))
    obs   = np.load(os.path.join(data_dir, f"glv_{label}_obs.npz"))

    t_grid   = truth["t"]
    X_true   = truth["X"]
    x0       = truth["x0"]
    Y_obs    = obs["Y"]
    H_state  = obs["H"]
    T        = len(t_grid)

    if full_obs:
        H_state = np.eye(N)
        Y_obs   = np.maximum(
            X_true + np.random.default_rng(42).normal(
                0, obs_noise_std, X_true.shape),
            0.0,
        )

    M = H_state.shape[0]
    H_aug = np.zeros((M, N_AUG))
    H_aug[:, :N] = H_state
    R_obs = obs_noise_std ** 2 * np.eye(M)

    th_true, labels = get_theta_true(a_hidden)

    if verbose:
        tag = "full obs" if full_obs else "partial obs"
        print(f"\n── IEnKS  [{tag}  a_hidden={a_hidden}] ──")
        print(f"   T={T}  window={window}  Ne={n_ensemble}  "
              f"n_iter={n_iter}  bundle_eps={bundle_eps:.0e}")

    # ── Prior ensemble ─────────────────────────────────────────────────────
    rng          = np.random.default_rng(1)
    theta_prior  = get_theta_true(0.0)[0]
    E0           = np.empty((N_AUG, n_ensemble))
    for j in range(n_ensemble):
        x_j     = np.maximum(x0 * (1 + 0.10 * rng.standard_normal(N)), 1e-3)
        theta_j = theta_prior * (1 + 0.20 * rng.standard_normal(N_THETA))
        E0[:, j] = np.concatenate([x_j, theta_j])

    # ── Cycle through windows ──────────────────────────────────────────────
    n_cycles         = (T - 1) // window
    E                = E0.copy()
    theta_history    = np.empty((n_cycles, N_THETA))
    X_mean_hist      = np.empty((N, T))
    X_mean_hist[:, 0] = E[:N, :].mean(axis=1)

    t_start_run = time.time()

    for cyc in range(n_cycles):
        i0    = cyc * window
        i1    = min(i0 + window, T - 1)
        t_win = t_grid[i0 : i1 + 1]
        Y_win = Y_obs[:, i0 + 1 : i1 + 1]

        if Y_win.shape[1] == 0:
            break

        E, u0_post = ienks_window(
            E, t_win, Y_win, H_aug, R_obs,
            n_iter=n_iter, bundle_eps=bundle_eps)

        theta_history[cyc] = u0_post[N:]

        for l in range(min(window, T - 1 - i0)):
            X_mean_hist[:, i0 + l + 1] = np.maximum(
                E[:N, :].mean(axis=1), 0.0)

        if verbose and (cyc + 1) % max(1, n_cycles // 10) == 0:
            th_est = theta_history[cyc]
            rmse   = np.sqrt(np.mean((th_est - th_true) ** 2))
            print(f"   cycle {cyc+1:3d}/{n_cycles}  param_RMSE={rmse:.4f}")

    elapsed    = time.time() - t_start_run
    theta_hat  = theta_history[-1]
    param_rmse = float(np.sqrt(np.mean((theta_hat - th_true) ** 2)))

    if verbose:
        print(f"\n   Wall time: {elapsed:.1f}s")
        print(f"   Final param RMSE: {param_rmse:.4f}")
        print(f"\n   {'Label':<8}  {'True':>8}  {'Estimate':>8}  {'Error':>8}")
        for lbl, tv, ev in zip(labels, th_true, theta_hat):
            print(f"   {lbl:<8}  {tv:>8.4f}  {ev:>8.4f}  {abs(ev-tv):>8.4f}")

    # ── Forward trajectory with final estimated parameters ─────────────────
    r_hat, A_hat = unpack_theta(theta_hat)
    sol = solve_ivp(
        lambda t, x: glv_rhs(x, r_hat, A_hat),
        [t_grid[0], t_grid[-1]], x0,
        method="RK45", t_eval=t_grid, rtol=1e-8, atol=1e-10,
    )
    X_hat = np.maximum(sol.y, 0.0)

    rmse_history = np.array([
        np.sqrt(np.mean((theta_history[c] - th_true) ** 2))
        for c in range(n_cycles)
    ])

    # ── Plots ──────────────────────────────────────────────────────────────
    obs_tag = "full_obs" if full_obs else "part_obs"
    obs_lbl = "(full obs)" if full_obs else "(partial obs)"
    _plot_param_errors(theta_hat, th_true, labels, obs_tag, obs_lbl, out_dir)
    _plot_trajectories(t_grid, X_true, X_hat, obs_tag, obs_lbl, out_dir)
    _plot_rmse(rmse_history, obs_tag, obs_lbl, out_dir)

    return dict(
        theta_hat=theta_hat, theta_true=th_true, labels=labels,
        theta_history=theta_history, rmse_history=rmse_history,
        X_hat=X_hat, X_true=X_true, t=t_grid,
        param_rmse=param_rmse, elapsed=elapsed,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save(fig, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ["png", "pdf"]:
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {path}")
    plt.close(fig)


def _plot_param_errors(theta_hat, th_true, labels, obs_tag, obs_lbl, out_dir):
    errors     = np.abs(theta_hat - th_true)
    bar_colors = ["#c0392b" if e > 0.05 else "#7f8c8d" for e in errors]
    fig, ax    = plt.subplots(figsize=(14, 4))
    ax.bar(np.arange(len(errors)), errors, color=bar_colors, alpha=0.85)
    ax.axhline(0.05, color="#c0392b", lw=1.2, ls="--", label="0.05 tolerance")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(r"$|\hat{\theta} - \theta_{\mathrm{true}}|$", fontsize=10)
    ax.set_title(f"IEnKS absolute error per parameter  {obs_lbl}", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, f"ienks_param_errors_{obs_tag}", out_dir)


def _plot_trajectories(t, X_true, X_hat, obs_tag, obs_lbl, out_dir,
                       show_obs=False, Y_obs=None, H_mat=None):
    groups  = [([0, 1], "producers"), ([2, 3], "herbivores"), ([4], "apex")]
    obs_idx = (np.where(H_mat.sum(axis=0) > 0)[0]
               if H_mat is not None else np.array([]))
    for species_group, group_name in groups:
        fig, ax = plt.subplots(figsize=(9, 4))
        for i in species_group:
            c = SPECIES_COLORS[i]
            ax.plot(t, X_true[i], color=c, lw=1.8, alpha=0.9)
            ax.plot(t, X_hat[i],  color="k", lw=1.8, ls="--", alpha=0.9)
            if show_obs and Y_obs is not None and i in obs_idx:
                k = int(np.where(obs_idx == i)[0][0])
                ax.scatter(t[::2], Y_obs[k, ::2], color=c, s=8, alpha=0.5, zorder=3)
        handles = [
            Line2D([0], [0], color="gray", lw=2,
                   label="True trajectory"),
            Line2D([0], [0], color="k", lw=2, ls="--",
                   label="IEnKS estimate"),
        ]
        ax.legend(handles=handles, fontsize=8)
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel("Population", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.set_title(f"IEnKS Trajectory  {obs_lbl}", fontsize=11)
        plt.tight_layout()
        _save(fig, f"ienks_traj_{group_name}_{obs_tag}", out_dir)


def _plot_rmse(rmse_history, obs_tag, obs_lbl, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.arange(1, len(rmse_history) + 1), rmse_history,
                lw=2, color="#e67e22", marker="o", ms=4, markevery=2)
    ax.set_xlabel("Assimilation cycle", fontsize=11)
    ax.set_ylabel("Parameter RMSE (log scale)", fontsize=11)
    ax.set_title(f"IEnKS parameter convergence  {obs_lbl}", fontsize=12)
    plt.tight_layout()
    _save(fig, f"ienks_rmse_{obs_tag}", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="IEnKS for gLV parameter estimation."
    )
    parser.add_argument("--a_hidden",    type=float, default=0.0)
    parser.add_argument("--partial_obs", action="store_true")
    parser.add_argument("--window",      type=int,   default=10,
                        help="Assimilation window length in time steps.")
    parser.add_argument("--n_ensemble",  type=int,   default=60)
    parser.add_argument("--n_iter",      type=int,   default=4,
                        help="Gauss-Newton iterations per cycle.")
    parser.add_argument("--bundle_eps",  type=float, default=1e-4)
    parser.add_argument("--data_dir",    type=str,   default=None,
                        help="Data directory (default: Data/gLV/data/).")
    parser.add_argument("--out",         type=str,   default=None,
                        help="Output directory for figures.")
    parser.add_argument("--quiet",       action="store_true")
    args = parser.parse_args()

    run_ienks(
        a_hidden   = args.a_hidden,
        full_obs   = not args.partial_obs,
        window     = args.window,
        n_ensemble = args.n_ensemble,
        n_iter     = args.n_iter,
        bundle_eps = args.bundle_eps,
        data_dir   = args.data_dir,
        out_dir    = args.out,
        verbose    = not args.quiet,
    )


if __name__ == "__main__":
    main()
