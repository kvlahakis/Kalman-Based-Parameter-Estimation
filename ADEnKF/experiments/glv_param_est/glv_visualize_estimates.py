"""
gLV Estimate Visualisation
==========================
Compares AD-EnKF parameter estimates against ground truth by overlaying
trajectories, parameter values, and projected phase plots.

Reads:
    - A parameter estimate file of the form glv_estimated_params.py
      (output by the AD-EnKF run), which defines R_EST and A_EST.
    - The corresponding truth/obs .npz files from the data directory.

Figures produced
----------------
Fig A — Parameter comparison bar chart
    Side-by-side bars for each of the 18 theta entries: true vs estimated.
    Highlights which parameters the filter recovered well and which it missed.

Fig B — Trajectory comparison (time series)
    True noisy trajectory (with observations) vs noiseless ODE integrated
    under R_EST / A_EST from the same initial condition x0.
    One panel per species, 2 columns: observed species (left) / unobserved (right).

Fig C — Projected trajectory comparison
    2 rows x 1 column: Producer 1 vs Herb3 (top), Producer 1 vs Herb4 (bottom).
    True noiseless trajectory in one color, estimated in another.
    Shows whether the estimated parameters reproduce the correct attractor.

Usage
-----
    python glv_visualize_estimates.py
    python glv_visualize_estimates.py --est_file runs/glv_param_est_torch_full_obs/glv_estimated_params.py
    python glv_visualize_estimates.py --a_hidden 0.0 --data_dir Data/gLV --out figures/estimates
"""

import argparse
import importlib.util
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Data', 'gLV'))
from glv_data_generator import (
    R_TRUE, A_TRUE, A_HIDDEN_SWEEP,
    get_true_A, get_theta, glv_rhs
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

SPECIES_NAMES  = ['Producer 1', 'Producer 2', 'Herbivore 1',
                  'Herbivore 2', 'Apex Predator']
SPECIES_COLORS = [
    '#1a7a1a',   # Producer 1  — deep forest green
    '#7fd67f',   # Producer 2  — light lime green
    '#1a3a8f',   # Herbivore 1 — deep navy blue
    '#63b3ed',   # Herbivore 2 — light sky blue
    '#e74c3c',   # Apex Predator — red
]

COLOR_TRUE = '#2c3e50'    # dark slate  — true system
COLOR_EST  = '#e67e22'    # orange      — estimated system

plt.rcParams.update({
    'font.family'      : 'sans-serif',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.25,
    'grid.linestyle'   : '--',
    'figure.dpi'       : 150,
})


# ---------------------------------------------------------------------------
# Load estimate file
# ---------------------------------------------------------------------------

def load_estimates(est_file):
    """
    Loads R_EST and A_EST from a glv_estimated_params.py file.
    Returns (R_est, A_est).
    """
    spec   = importlib.util.spec_from_file_location("est", est_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.R_EST, module.A_EST


def load_data(a_hidden, data_dir):
    label = f"ahidden{a_hidden:.2f}".replace('.', 'p')
    truth = np.load(os.path.join(data_dir, f'data/glv_{label}_truth.npz'))
    obs   = np.load(os.path.join(data_dir, f'data/glv_{label}_obs.npz'))
    return truth, obs


# ---------------------------------------------------------------------------
# Integrate noiseless ODE
# ---------------------------------------------------------------------------

def integrate_noiseless(r, A, x0, t_span, dt=0.5):
    t_eval = np.arange(t_span[0], t_span[1] + dt * 0.5, dt)
    sol = solve_ivp(
        glv_rhs, t_span, x0, args=(r, A),
        method='RK45', rtol=1e-9, atol=1e-11,
        t_eval=t_eval, dense_output=False
    )
    return sol.t, sol.y   # (T,), (5, T)


# ---------------------------------------------------------------------------
# Fig A — Parameter comparison
# ---------------------------------------------------------------------------

def fig_parameter_comparison(R_true, A_true, R_est, A_est, a_hidden, out_dir):
    theta_true, labels = get_theta(A_true, r=R_true)
    theta_est,  _      = get_theta(A_est,  r=R_est)

    n   = len(theta_true)
    idx = np.arange(n)
    w   = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={'height_ratios': [2, 1]})

    # top: side-by-side bars
    ax = axes[0]
    ax.bar(idx - w/2, theta_true, w, label='True $\\theta$',
           color=COLOR_TRUE, alpha=0.8)
    ax.bar(idx + w/2, theta_est,  w, label='Estimated $\\hat{\\theta}$',
           color=COLOR_EST,  alpha=0.8)
    ax.axhline(0, color='black', lw=0.8, ls='-')
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Parameter value', fontsize=10)
    ax.set_title(
        f'Parameter recovery  ($a_{{\\mathrm{{hidden}}}}={a_hidden:.2f}$)',
        fontsize=12
    )
    ax.legend(fontsize=10)

    # bottom: absolute error
    ax = axes[1]
    errors = np.abs(theta_est - theta_true)
    bar_colors = [COLOR_EST if e > 0.05 else '#95a5a6' for e in errors]
    ax.bar(idx, errors, color=bar_colors, alpha=0.85)
    ax.axhline(0.05, color='#c0392b', lw=1.2, ls='--',
               label='5% tolerance')
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('$|\\hat{\\theta} - \\theta|$', fontsize=10)
    ax.set_title('Absolute error per parameter', fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, f'figA_param_comparison_ah{a_hidden:.2f}', out_dir)


# ---------------------------------------------------------------------------
# Fig B — Time series comparison
# ---------------------------------------------------------------------------

def fig_trajectory_comparison(truth, obs, R_est, A_est, a_hidden, out_dir):
    t     = truth['t']
    X     = truth['X']      # (5, T) true stochastic trajectory
    Y     = obs['Y']        # (3, T) noisy observations
    H     = obs['H']        # (3, 5) observation operator
    x0    = truth['x0']
    obs_idx = np.where(H.sum(axis=0) > 0)[0]

    # integrate estimated model from same x0
    _, X_est = integrate_noiseless(R_est, A_est, x0, (t[0], t[-1]), dt=float(t[1]-t[0]))
    # also integrate true noiseless for clean comparison
    A_true_full = truth['A']
    R_true_full = truth['r']
    _, X_true_clean = integrate_noiseless(R_true_full, A_true_full, x0,
                                          (t[0], t[-1]), dt=float(t[1]-t[0]))

    obs_set   = set(obs_idx.tolist())
    unobs_idx = [i for i in range(5) if i not in obs_set]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)


    panel_specs = [
        (axes[0, 0], obs_idx,   'Producers and Herbivore 1 — true (noiseless) vs estimated'),
        (axes[0, 1], unobs_idx, 'Herbivore 2 and Apex Predator — true (noiseless) vs estimated'),
    ]

    for ax, idxs, title in panel_specs:
        for i in idxs:
            c = SPECIES_COLORS[i]
            # true noiseless
            ax.plot(t, X_true_clean[i], color=c, lw=1.8, alpha=0.9,
                    label=f'{SPECIES_NAMES[i]} (true)')
            # estimated
            ax.plot(t, X_est[i], color=c, lw=1.8, ls='--', alpha=0.9,
                    label=f'{SPECIES_NAMES[i]} (est.)')
        ax.set_title(title, fontsize=10)
        ax.set_ylim(bottom=0)

    # bottom-left: observations and estimated only (no noiseless true)
    ax = axes[1, 0]
    for k, i in enumerate(obs_idx):
        ax.plot(t, X_est[i], color=SPECIES_COLORS[i], lw=1.8, ls='--',
                label=f'{SPECIES_NAMES[i]} (est.)')
        
        ax.plot(t[::2], Y[k, ::2], color=SPECIES_COLORS[i], label=f'{SPECIES_NAMES[i]} (stochastic)')

    ax.set_title('Producers and Herbivore 1 — noisy observations vs estimated', fontsize=10)
    ax.set_ylim(bottom=0)

    # bottom-right: show true stochastic trajectory for unobserved species
    ax = axes[1, 1]
    ax.cla()
    for i in unobs_idx:
        ax.plot(t, X[i], color=SPECIES_COLORS[i], lw=1.8,
                label=f'{SPECIES_NAMES[i]} (stochastic)')
        ax.plot(t, X_est[i], color=SPECIES_COLORS[i], lw=1.8, ls='--',
                label=f'{SPECIES_NAMES[i]} (est.)')
        
    ax.set_title('Unobserved species — stochastic truth vs estimated', fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25, ls='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for ax in axes[1]:
        ax.set_xlabel('Time', fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel('Population', fontsize=10)

    # legend: solid=true, dashed=estimated
    style_handles = [
        Line2D([0], [0], color='gray', lw=2,       label='True (noiseless ODE)'),
        Line2D([0], [0], color='gray', lw=2, ls='--', label='Estimated parameters'),
        Line2D([0], [0], color='gray', lw=0, marker='o', ms=5, label='Noisy observation $y_k$'),
    ]
    species_handles = [
        Line2D([0], [0], color=SPECIES_COLORS[i], lw=2, label=SPECIES_NAMES[i])
        for i in range(5)
    ]
    fig.legend(handles=species_handles + style_handles,
               loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(
        f'Trajectory comparison: true vs estimated  ($a_{{\\mathrm{{hidden}}}}={a_hidden:.2f}$)',
        fontsize=13
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    _save(fig, f'figB_trajectory_comparison_ah{a_hidden:.2f}', out_dir)


# ---------------------------------------------------------------------------
# Fig C — Projected trajectory comparison
# ---------------------------------------------------------------------------

def fig_projected_trajectories(truth, R_est, A_est, a_hidden, out_dir):
    x0    = truth['x0']
    A_true_full = truth['A']
    R_true_full = truth['r']
    t_end = float(truth['t'][-1])
    dt    = float(truth['t'][1] - truth['t'][0])

    T_total     = 200.0   # longer integration to show full attractor
    T_transient = 80.0
    dt_fine     = 0.02

    t_eval = np.arange(0, T_total, dt_fine)

    sol_true = solve_ivp(glv_rhs, [0, T_total], x0,
                         args=(R_true_full, A_true_full),
                         method='RK45', rtol=1e-9, atol=1e-11,
                         t_eval=t_eval)
    sol_est  = solve_ivp(glv_rhs, [0, T_total], x0,
                         args=(R_est, A_est),
                         method='RK45', rtol=1e-9, atol=1e-11,
                         t_eval=t_eval)

    t    = sol_true.t
    split = np.searchsorted(t, T_transient)

    row_specs = [
        (2, 'Herbivore 1'),
        (3, 'Herbivore 2'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (species_idx, species_name) in zip(axes, row_specs):
        x1_true = sol_true.y[0]
        xi_true = sol_true.y[species_idx]
        x1_est  = sol_est.y[0]
        xi_est  = sol_est.y[species_idx]

        # transient faded
        ax.plot(x1_true[:split], xi_true[:split],
                color=COLOR_TRUE, lw=0.7, alpha=0.2)
        ax.plot(x1_est[:split],  xi_est[:split],
                color=COLOR_EST,  lw=0.7, alpha=0.2)

        # post-transient
        ax.plot(x1_true[split:], xi_true[split:],
                color=COLOR_TRUE, lw=1.4, alpha=0.9, label='True parameters')
        ax.plot(x1_est[split:],  xi_est[split:],
                color=COLOR_EST,  lw=1.4, alpha=0.9, ls='--',
                label='Estimated parameters')

        # fixed points
        ax.scatter(x1_true[-1], xi_true[-1], color=COLOR_TRUE,
                   s=80, marker='*', zorder=5)
        ax.scatter(x1_est[-1],  xi_est[-1],  color=COLOR_EST,
                   s=80, marker='*', zorder=5)

        ax.set_xlabel('Producer 1', fontsize=10)
        ax.set_ylabel(species_name, fontsize=10,
                      color=SPECIES_COLORS[species_idx])
        ax.set_title(f'Producer 1 vs {species_name}', fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle(
        f'Projected trajectories: true vs estimated  ($a_{{\\mathrm{{hidden}}}}={a_hidden:.2f}$)\n'
        'Noiseless ODE, transient faded — stars mark fixed points',
        fontsize=12
    )
    plt.tight_layout()
    _save(fig, f'figC_projected_trajectories_ah{a_hidden:.2f}', out_dir)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ['pdf', 'png']:
        path = os.path.join(out_dir, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Visualise AD-EnKF parameter estimates vs ground truth.'
    )
    parser.add_argument('--est_file', type=str,
                        default='runs/glv_param_est_torch_full_obs/glv_estimated_params.py',
                        help='Path to glv_estimated_params.py output by the filter.')
    parser.add_argument('--a_hidden', type=float, default=0.0,
                        help='Which sweep value the estimates correspond to.')
    parser.add_argument('--data_dir', type=str, default='../../../Data/gLV')
    parser.add_argument('--out',      type=str, default='runs/glv_param_est_torch_full_obs/figures/estimates')
    parser.add_argument('--fig',      type=str, default=None,
                        choices=['A', 'B', 'C'],
                        help='Single figure to produce. Omit for all.')
    args = parser.parse_args()

    print(f"Loading estimates from: {args.est_file}")
    R_est, A_est = load_estimates(args.est_file)

    print(f"Loading data for a_hidden={args.a_hidden:.2f} from: {args.data_dir}")
    truth, obs = load_data(args.a_hidden, args.data_dir)

    A_true = get_true_A(args.a_hidden)

    to_run = ['A', 'B', 'C'] if args.fig is None else [args.fig]

    if 'A' in to_run:
        print("\n--- Fig A: Parameter comparison ---")
        fig_parameter_comparison(R_TRUE, A_true, R_est, A_est,
                                 args.a_hidden, args.out)

    if 'B' in to_run:
        print("\n--- Fig B: Trajectory comparison ---")
        fig_trajectory_comparison(truth, obs, R_est, A_est,
                                  args.a_hidden, args.out)

    if 'C' in to_run:
        print("\n--- Fig C: Projected trajectories ---")
        fig_projected_trajectories(truth, R_est, A_est,
                                   args.a_hidden, args.out)


if __name__ == '__main__':
    main()
