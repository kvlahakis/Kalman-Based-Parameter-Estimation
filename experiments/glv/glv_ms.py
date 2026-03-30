"""
Multiple Shooting + Sensitivity-Equation Gradients for gLV
===========================================================

Breaks the trajectory into K short segments with free initial states,
enforcing continuity as a soft penalty.  Exact first-order gradients via
forward sensitivity equations — no ensemble, no adjoint.

Usage
-----
    # From repo root:
    python experiments/glv/glv_ms.py
    python experiments/glv/glv_ms.py --a_hidden 0.2
    python experiments/glv/glv_ms.py --partial_obs
    python experiments/glv/glv_ms.py --seg_len 10
"""

import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

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
N_A           = int(SPARSITY_MASK.sum())   # 10
N_THETA       = N + N_A                    # 15

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
    r = theta[:N].copy()
    A = np.zeros((N, N))
    A[A_ROW, A_COL] = theta[N:]
    return r, A


# ---------------------------------------------------------------------------
# gLV RHS and Jacobians
# ---------------------------------------------------------------------------

def glv_rhs(x, r, A):
    x = np.maximum(x, 0.0)
    return x * (r + A @ x)


def jac_x(x, r, A):
    """df/dx  (N, N)"""
    x = np.maximum(x, 0.0)
    return np.diag(r + A @ x) + np.diag(x) @ A


def jac_theta(x):
    """df/d_theta  (N, N_THETA)"""
    x = np.maximum(x, 0.0)
    J = np.zeros((N, N_THETA))
    J[:, :N] = np.diag(x)
    for k, (row, col) in enumerate(zip(A_ROW, A_COL)):
        J[row, N + k] = x[row] * x[col]
    return J


# ---------------------------------------------------------------------------
# Augmented ODE for sensitivity equations
# ---------------------------------------------------------------------------

def augmented_rhs(t, z, r, A):
    x      = z[:N]
    phi_th = z[N : N + N * N_THETA].reshape(N, N_THETA)
    phi_s  = z[N + N * N_THETA :].reshape(N, N)
    Jx     = jac_x(x, r, A)
    Jth    = jac_theta(x)
    return np.concatenate([
        glv_rhs(x, r, A),
        (Jx @ phi_th + Jth).ravel(),
        (Jx @ phi_s).ravel(),
    ])


def integrate_segment(s_k, t_seg, r, A):
    """Returns X (len,N), Phi_th (len,N,N_THETA), Phi_s (len,N,N)."""
    z0  = np.concatenate([s_k, np.zeros(N * N_THETA), np.eye(N).ravel()])
    sol = solve_ivp(
        augmented_rhs, [t_seg[0], t_seg[-1]], z0,
        args=(r, A), method="RK45", t_eval=t_seg,
        rtol=1e-7, atol=1e-9, max_step=1.0,
    )
    if not sol.success or sol.y.ndim < 2:
        L = len(t_seg)
        return (np.zeros((L, N)),
                np.zeros((L, N, N_THETA)),
                np.tile(np.eye(N), (L, 1, 1)))
    Z      = sol.y.T
    X      = Z[:, :N]
    Phi_th = Z[:, N : N + N * N_THETA].reshape(-1, N, N_THETA)
    Phi_s  = Z[:, N + N * N_THETA :].reshape(-1, N, N)
    return X, Phi_th, Phi_s


# ---------------------------------------------------------------------------
# Objective + exact gradient
# ---------------------------------------------------------------------------

def objective_and_grad(packed, t_grid, Y_obs, H_mat, seg_starts, gamma, sigma):
    K     = len(seg_starts)
    T     = len(t_grid)
    theta = packed[:N_THETA]
    S     = packed[N_THETA:].reshape(K, N)

    r, A   = unpack_theta(theta)
    inv_s2 = 1.0 / sigma ** 2
    loss   = 0.0
    g_th   = np.zeros(N_THETA)
    g_s    = np.zeros((K, N))

    for k in range(K):
        i0    = seg_starts[k]
        i1    = seg_starts[k + 1] if k + 1 < K else T
        t_seg = t_grid[i0:i1]

        X, Phi_th, Phi_s = integrate_segment(S[k], t_seg, r, A)

        resid  = H_mat @ X.T - Y_obs[:, i0:i1]
        loss  += inv_s2 * np.sum(resid ** 2)

        for ti in range(len(t_seg)):
            HtR    = H_mat.T @ resid[:, ti]
            g_th  += 2 * inv_s2 * Phi_th[ti].T @ HtR
            g_s[k] += 2 * inv_s2 * Phi_s[ti].T @ HtR

        if k + 1 < K:
            diff        = X[-1] - S[k + 1]
            loss       += gamma * inv_s2 * np.dot(diff, diff)
            g_th       += 2 * gamma * inv_s2 * Phi_th[-1].T @ diff
            g_s[k]     += 2 * gamma * inv_s2 * Phi_s[-1].T @ diff
            g_s[k + 1] -= 2 * gamma * inv_s2 * diff

    return loss, np.concatenate([g_th, g_s.ravel()])


# ---------------------------------------------------------------------------
# Main fitting routine
# ---------------------------------------------------------------------------

def run_multishoot(a_hidden=0.0, full_obs=True, seg_len=20, gamma=1e3,
                   sigma=0.05, data_dir=None, out_dir=None,
                   maxiter=800, verbose=True):

    if data_dir is None:
        data_dir = str(GLV_DATA_DIR)
    if out_dir is None:
        out_dir = str(Path(__file__).resolve().parent / "figures" / "ms")

    label  = f"ahidden{a_hidden:.2f}".replace(".", "p")
    truth  = np.load(os.path.join(data_dir, f"glv_{label}_truth.npz"))
    obs    = np.load(os.path.join(data_dir, f"glv_{label}_obs.npz"))

    t_grid = truth["t"]
    X_true = truth["X"]
    x0     = truth["x0"]
    Y_obs  = obs["Y"]
    H_mat  = obs["H"]
    T      = len(t_grid)

    if full_obs:
        H_mat = np.eye(N)
        Y_obs = np.maximum(
            X_true + np.random.default_rng(42).normal(0, sigma, X_true.shape), 0.0)

    th_true, labels = get_theta_true(a_hidden)
    seg_starts = [i for i in range(0, T, seg_len) if i < T - 1]
    K          = len(seg_starts)

    if verbose:
        tag = "full obs" if full_obs else "partial obs"
        print(f"\n── Multiple Shooting  [{tag}  a_hidden={a_hidden}] ──")
        print(f"   T={T}  seg_len={seg_len}  K={K}  sigma={sigma}")
        print(f"   Free vars: {N_THETA} (theta) + {K*N} (seg states) = {N_THETA+K*N}")

    rng       = np.random.default_rng(0)
    theta0    = th_true * (1.0 + 0.30 * rng.standard_normal(N_THETA))
    s0_guess  = np.array([X_true[:, i] for i in seg_starts]).flatten()
    s0_guess *= (1.0 + 0.05 * rng.standard_normal(s0_guess.shape))
    packed0   = np.concatenate([theta0, s0_guess])

    bounds = [(None, None)] * N_THETA
    if full_obs:
        for val in Y_obs[:, 0]:
            bounds.append((max(1e-5, val * 0.90), val * 1.10))
        bounds += [(1e-5, None)] * ((K - 1) * N)
    else:
        bounds += [(1e-5, None)] * (K * N)

    loss_history = []
    iter_count   = [0]

    def make_callback(pass_name):
        def callback(packed):
            iter_count[0] += 1
            if verbose and iter_count[0] % 50 == 0:
                rmse = np.sqrt(np.mean((packed[:N_THETA] - th_true) ** 2))
                print(f"   [{pass_name}] iter {iter_count[0]:4d}  "
                      f"param_RMSE={rmse:.4f}")
        return callback

    t0_run = time.time()

    # Pass 1: warm-up with low continuity penalty
    if verbose:
        print("\n   --- Pass 1: Multiple Shooting Warm-up (gamma=1.0) ---")
    iter_count[0] = 0

    def fg_pass1(packed):
        val, grad = objective_and_grad(
            packed, t_grid, Y_obs, H_mat, seg_starts, 1.0, sigma)
        loss_history.append(float(val))
        return val, grad

    res_pass1 = minimize(
        fg_pass1, packed0, method="L-BFGS-B", jac=True,
        bounds=bounds, callback=make_callback("Pass 1"),
        options=dict(maxiter=200, ftol=1e-11, gtol=1e-6))

    # Pass 2: single-shooting polish from warm-up estimate
    if verbose:
        print("\n   --- Pass 2: Single Shooting Polish ---")
    iter_count[0] = 0

    theta_pass1   = res_pass1.x[:N_THETA]
    s0_pass1      = res_pass1.x[N_THETA : N_THETA + N]
    packed_polish = np.concatenate([theta_pass1, s0_pass1])
    seg_starts_ss = [0]
    bounds_ss     = [(None, None)] * N_THETA + [(1e-5, None)] * N

    def fg_pass2(packed):
        val, grad = objective_and_grad(
            packed, t_grid, Y_obs, H_mat, seg_starts_ss, 0.0, sigma)
        loss_history.append(float(val))
        return val, grad

    result = minimize(
        fg_pass2, packed_polish, method="L-BFGS-B", jac=True,
        bounds=bounds_ss, callback=make_callback("Pass 2"),
        options=dict(maxiter=1000, ftol=1e-14, gtol=1e-10))

    elapsed    = time.time() - t0_run
    theta_hat  = result.x[:N_THETA]
    param_rmse = float(np.sqrt(np.mean((theta_hat - th_true) ** 2)))

    if verbose:
        print(f"\n   Converged: {result.success}  ({result.message})")
        print(f"   Wall time: {elapsed:.1f}s")
        print(f"   Final param RMSE: {param_rmse:.4f}")
        print(f"\n   {'Label':<8}  {'True':>8}  {'Estimate':>8}  {'Error':>8}")
        for lbl, tv, ev in zip(labels, th_true, theta_hat):
            print(f"   {lbl:<8}  {tv:>8.4f}  {ev:>8.4f}  {abs(ev-tv):>8.4f}")

    r_hat, A_hat = unpack_theta(theta_hat)
    sol = solve_ivp(
        lambda t, x: glv_rhs(x, r_hat, A_hat),
        [t_grid[0], t_grid[-1]], x0,
        method="RK45", t_eval=t_grid, rtol=1e-8, atol=1e-10,
    )
    X_hat = np.maximum(sol.y, 0.0)

    obs_tag = "full_obs" if full_obs else "part_obs"
    obs_lbl = "[full obs]" if full_obs else "[partial obs]"
    _plot_param_errors(theta_hat, th_true, labels, obs_tag, obs_lbl, out_dir)
    _plot_trajectories(t_grid, X_true, X_hat, obs_tag, obs_lbl, out_dir,
                       show_obs=True, Y_obs=Y_obs, H_mat=H_mat)
    _plot_loss(loss_history, obs_tag, obs_lbl, out_dir)

    return dict(theta_hat=theta_hat, theta_true=th_true, labels=labels,
                X_hat=X_hat, X_true=X_true, t=t_grid,
                param_rmse=param_rmse, loss_history=loss_history,
                elapsed=elapsed)


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
    ax.set_title(f"Multiple shooting parameter error  {obs_lbl}", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, f"ms_param_errors_{obs_tag}", out_dir)


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
            ax.plot(t, X_hat[i],  color=c, lw=1.8, ls="--", alpha=0.9)
            if show_obs and Y_obs is not None and i in obs_idx:
                k = int(np.where(obs_idx == i)[0][0])
                ax.scatter(t[::2], Y_obs[k, ::2], color=c, s=8, alpha=0.5, zorder=3)
        handles = [
            Line2D([0], [0], color=SPECIES_COLORS[i], lw=2,
                   label=SPECIES_NAMES[i]) for i in species_group
        ] + [
            Line2D([0], [0], color="gray", lw=2,          label="True trajectory"),
            Line2D([0], [0], color="gray", lw=2, ls="--", label="MS estimate"),
        ]
        ax.legend(handles=handles, fontsize=8)
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel("Population", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.set_title(f"Multiple shooting trajectory  {obs_lbl}", fontsize=11)
        plt.tight_layout()
        _save(fig, f"ms_traj_{group_name}_{obs_tag}", out_dir)


def _plot_loss(loss_history, obs_tag, obs_lbl, out_dir):
    if not loss_history:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(loss_history, lw=2, color="#8e44ad", marker="o", ms=4,
                markevery=5)
    ax.set_xlabel("L-BFGS-B iteration", fontsize=11)
    ax.set_ylabel("Objective (log scale)", fontsize=11)
    ax.set_title(f"Multiple shooting convergence  {obs_lbl}", fontsize=12)
    plt.tight_layout()
    _save(fig, f"ms_loss_{obs_tag}", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multiple shooting + sensitivity equations for gLV."
    )
    parser.add_argument("--a_hidden",    type=float, default=0.0)
    parser.add_argument("--partial_obs", action="store_true")
    parser.add_argument("--seg_len",     type=int,   default=20,
                        help="Segment length in time steps (default 20).")
    parser.add_argument("--data_dir",    type=str,   default=None,
                        help="Data directory (default: Data/gLV/data/).")
    parser.add_argument("--out",         type=str,   default=None,
                        help="Output directory for figures.")
    parser.add_argument("--quiet",       action="store_true")
    args = parser.parse_args()

    run_multishoot(
        a_hidden = args.a_hidden,
        full_obs = not args.partial_obs,
        seg_len  = args.seg_len,
        data_dir = args.data_dir,
        out_dir  = args.out,
        verbose  = not args.quiet,
    )


if __name__ == "__main__":
    main()
