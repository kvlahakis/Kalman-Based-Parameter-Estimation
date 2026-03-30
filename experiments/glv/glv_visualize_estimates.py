"""
gLV Estimate Visualisation -- one panel per image file.

Every panel is saved as its own PNG/PDF so they can be arranged
freely as subplots in presentation slides.

Output files  (s = full_obs or part_obs, optionally _ah0.15 etc.)
------------------------------------------------------------------
figA1_param_values_{s}.png          -- true vs estimated bar chart
figA2_param_errors_{s}.png          -- absolute error per parameter

Full-obs trajectory panels:
  figB1_traj_producers_{s}.png
  figB2_traj_herbivores_{s}.png
  figB3_traj_apex_{s}.png

Partial-obs trajectory panels:
  figB1_traj_observed_noiseless_{s}.png
  figB2_traj_unobserved_noiseless_{s}.png
  figB3_traj_observed_noisy_{s}.png
  figB4_traj_unobserved_stochastic_{s}.png

figC1_projected_herb1_{s}.png
figC2_projected_herb2_{s}.png

figD_mse_history.png

Usage
-----
    python glv_visualize_estimates.py                   # partial obs, well-specified
    python glv_visualize_estimates.py --full_obs        # full obs run
    python glv_visualize_estimates.py --a_hidden 0.15   # show a_hidden in titles
    python glv_visualize_estimates.py --fig A           # only Fig A panels
"""

import argparse
import importlib.util
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Data', 'gLV'))
from glv_data_generator import (
    R_TRUE, A_TRUE, A_HIDDEN_SWEEP,
    get_true_A, get_theta, glv_rhs
)

SPECIES_NAMES  = ['Producer 1', 'Producer 2', 'Herbivore 1', 'Herbivore 2', 'Apex Predator']
SPECIES_COLORS = ['#1a7a1a', '#7fd67f', '#1a3a8f', '#63b3ed', '#e74c3c']
COLOR_TRUE = '#2c3e50'
COLOR_EST  = '#e67e22'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': '--',
    'figure.dpi': 150,
})


def _ahidden_label(a_hidden):
    return '' if a_hidden is None else f'  ($a_{{\\mathrm{{hidden}}}}={a_hidden:.2f}$)'

def _obs_label(full_obs):
    return '  (full obs)' if full_obs else '  (partial obs)'

def _obs_suffix(full_obs):
    return 'full_obs' if full_obs else 'part_obs'

def _suffix(a_hidden, full_obs):
    a = f'_ah{a_hidden:.2f}' if a_hidden is not None else ''
    return f'_{_obs_suffix(full_obs)}{a}'


def load_estimates(est_file):
    spec = importlib.util.spec_from_file_location("est", est_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.R_EST, module.A_EST


def load_data(a_hidden, data_dir):
    a_h   = a_hidden if a_hidden is not None else 0.0
    label = f"ahidden{a_h:.2f}".replace('.', 'p')
    truth = np.load(os.path.join(data_dir, f'data/glv_{label}_truth.npz'))
    obs   = np.load(os.path.join(data_dir, f'data/glv_{label}_obs.npz'))
    return truth, obs


def integrate_noiseless(r, A, x0, t_span, dt=0.5):
    t_eval = np.arange(t_span[0], t_span[1] + dt * 0.5, dt)
    sol = solve_ivp(glv_rhs, t_span, x0, args=(r, A),
                    method='RK45', rtol=1e-9, atol=1e-11,
                    t_eval=t_eval, dense_output=False)
    if sol.y.shape[1] < len(t_eval):
        print(f"  Warning: ODE stopped at t={sol.t[-1]:.1f} -- parameters may be unstable.")
        pad = len(t_eval) - sol.y.shape[1]
        return t_eval, np.concatenate(
            [sol.y, np.full((sol.y.shape[0], pad), np.nan)], axis=1)
    return sol.t, sol.y


def _save(fig, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ['pdf', 'png']:
        path = os.path.join(out_dir, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        print(f"  Saved: {path}")
    plt.close(fig)


def _traj_legend(ax, species_indices, include_obs=False, stochastic=False):
    handles = [
        Line2D([0], [0], color=SPECIES_COLORS[i], lw=2, label=SPECIES_NAMES[i])
        for i in species_indices
    ]
    if stochastic:
        handles += [
            Line2D([0], [0], color='gray', lw=2, alpha=0.5, label='Stochastic truth'),
            Line2D([0], [0], color='gray', lw=2, ls='--',   label='Estimated'),
        ]
    else:
        handles += [
            Line2D([0], [0], color='gray', lw=2,          label='True (noiseless)'),
            Line2D([0], [0], color='gray', lw=2, ls='--', label='Estimated'),
        ]
    if include_obs:
        handles.append(
            Line2D([0], [0], color='gray', lw=0, marker='o', ms=5, label='Noisy obs $y_k$'))
    ax.legend(handles=handles, fontsize=8)


# ---------------------------------------------------------------------------
# Fig A -- two separate images
# ---------------------------------------------------------------------------

def fig_parameter_comparison(R_true, A_true, R_est, A_est, a_hidden, out_dir, full_obs=True):
    theta_true, labels = get_theta(A_true, r=R_true)
    theta_est,  _      = get_theta(A_est,  r=R_est)
    idx, w = np.arange(len(theta_true)), 0.35

    # Full obs: slate/orange. Partial obs: navy/teal -- distinct for side-by-side slides
    c_true = COLOR_TRUE if full_obs else '#1a3a8f'
    c_est  = COLOR_EST  if full_obs else '#17a589'
    s  = _suffix(a_hidden, full_obs)
    tl = _ahidden_label(a_hidden) + _obs_label(full_obs)

    # A1 -- parameter values
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(idx - w/2, theta_true, w, label='True $\\theta$',            color=c_true, alpha=0.8)
    ax.bar(idx + w/2, theta_est,  w, label='Estimated $\\hat{\\theta}$', color=c_est,  alpha=0.8)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Parameter value', fontsize=10)
    ax.set_title('Parameter recovery' + tl, fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(fig, f'figA1_param_values{s}', out_dir)

    # A2 -- absolute errors
    errors     = np.abs(theta_est - theta_true)
    bar_colors = [c_est if e > 0.05 else '#95a5a6' for e in errors]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(idx, errors, color=bar_colors, alpha=0.85)
    ax.axhline(0.05, color='#c0392b', lw=1.2, ls='--', label='0.05 tolerance')
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('$|\\hat{\\theta} - \\theta|$', fontsize=10)
    ax.set_title('Absolute error per parameter' + tl, fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, f'figA2_param_errors{s}', out_dir)


# ---------------------------------------------------------------------------
# Fig B -- one image per panel
# ---------------------------------------------------------------------------

def fig_trajectory_comparison(truth, obs, R_est, A_est, a_hidden, out_dir, full_obs=True):
    t  = truth['t'];  X  = truth['X'];  x0 = truth['x0']
    Y  = obs['Y'];    H  = obs['H']

    obs_idx   = np.where(H.sum(axis=0) > 0)[0]
    unobs_idx = np.array([i for i in range(5) if i not in set(obs_idx.tolist())])
    all_obs   = len(unobs_idx) == 0

    _, X_est        = integrate_noiseless(R_est,      A_est,      x0,
                                          (t[0], t[-1]), dt=float(t[1]-t[0]))
    _, X_true_clean = integrate_noiseless(truth['r'], truth['A'], x0,
                                          (t[0], t[-1]), dt=float(t[1]-t[0]))

    s  = _suffix(a_hidden, full_obs)
    tl = _ahidden_label(a_hidden) + _obs_label(full_obs)

    def _panel(idxs, fname, title, noisy_k=None, stochastic=False):
        fig, ax = plt.subplots(figsize=(8, 4))
        for i in idxs:
            if stochastic:
                ax.plot(t, X[i],     color=SPECIES_COLORS[i], lw=1.8, alpha=0.5)
                ax.plot(t, X_est[i], color='k', lw=1.8, ls='--')
            elif noisy_k is not None:
                ax.plot(t, X_est[i], color='k', lw=1.8, ls='--')
            else:
                ax.plot(t, X_true_clean[i], color=SPECIES_COLORS[i], lw=1.8)
                ax.plot(t, X_est[i],        color='k', lw=1.8, ls='--')
        if noisy_k is not None:
            for k, i in zip(noisy_k, idxs):
                ax.scatter(t[::2], Y[k, ::2], color=SPECIES_COLORS[i],
                           s=8, alpha=0.6, zorder=3)
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Population', fontsize=10)
        ax.set_title(title + tl, fontsize=11)
        ax.set_ylim(bottom=0)
        _traj_legend(ax, idxs,
                     include_obs=(noisy_k is not None),
                     stochastic=stochastic)
        plt.tight_layout()
        _save(fig, f'{fname}{s}', out_dir)

    if all_obs:
        _panel([0], 'figB1_traj_producers',  'Producer 1 -- true vs estimated')
        _panel([2], 'figB2_traj_herbivores',  'Herbivore 1 -- true vs estimated')
        _panel([4],    'figB3_traj_apex',         'Apex Predator -- true vs estimated')
    else:
        _panel(obs_idx,   'figB1_traj_observed_noiseless',
               'Observed -- true (noiseless) vs estimated')
        _panel(unobs_idx, 'figB2_traj_unobserved_noiseless',
               'Unobserved -- true (noiseless) vs estimated')
        _panel(obs_idx,   'figB3_traj_observed_noisy',
               'Observed -- noisy observations vs estimated',
               noisy_k=list(range(len(obs_idx))))
        _panel(unobs_idx, 'figB4_traj_unobserved_stochastic',
               'Unobserved -- stochastic truth vs estimated',
               stochastic=True)


# ---------------------------------------------------------------------------
# Fig C -- one image per herbivore
# ---------------------------------------------------------------------------

def fig_projected_trajectories(truth, R_est, A_est, a_hidden, out_dir, full_obs=True):
    x0 = truth['x0']
    T_total, T_transient, dt_fine = 200.0, 80.0, 0.02
    t_eval = np.arange(0, T_total, dt_fine)

    sol_true = solve_ivp(glv_rhs, [0, T_total], x0,
                         args=(truth['r'], truth['A']),
                         method='RK45', rtol=1e-9, atol=1e-11, t_eval=t_eval)
    sol_est  = solve_ivp(glv_rhs, [0, T_total], x0,
                         args=(R_est, A_est),
                         method='RK45', rtol=1e-9, atol=1e-11, t_eval=t_eval)

    split = np.searchsorted(sol_true.t, T_transient)
    s  = _suffix(a_hidden, full_obs)
    tl = _ahidden_label(a_hidden) + _obs_label(full_obs)

    for fname, sp_idx, sp_name in [
        ('figC1_projected_herb1', 2, 'Herbivore 1'),
        ('figC2_projected_herb2', 3, 'Herbivore 2'),
    ]:
        x1_true = sol_true.y[0];  xi_true = sol_true.y[sp_idx]
        x1_est  = sol_est.y[0];   xi_est  = sol_est.y[sp_idx]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(x1_true[:split], xi_true[:split], color=COLOR_TRUE, lw=0.7, alpha=0.2)
        ax.plot(x1_est[:split],  xi_est[:split],  color=COLOR_EST,  lw=0.7, alpha=0.2)
        ax.plot(x1_true[split:], xi_true[split:], color=COLOR_TRUE, lw=1.4, alpha=0.9,
                label='True parameters')
        ax.plot(x1_est[split:],  xi_est[split:],  color=COLOR_EST,  lw=1.4, alpha=0.9,
                ls='--', label='Estimated parameters')
        ax.scatter(x1_true[-1], xi_true[-1], color=COLOR_TRUE, s=80, marker='*', zorder=5)
        ax.scatter(x1_est[-1],  xi_est[-1],  color=COLOR_EST,  s=80, marker='*', zorder=5)
        ax.set_xlabel('Producer 1', fontsize=10)
        ax.set_ylabel(sp_name, fontsize=10, color=SPECIES_COLORS[sp_idx])
        ax.set_title(
            f'Producer 1 vs {sp_name}' + tl + '\n'
            'Transient faded -- stars mark fixed points',
            fontsize=10
        )
        ax.legend(fontsize=9)
        plt.tight_layout()
        _save(fig, f'{fname}{s}', out_dir)


# ---------------------------------------------------------------------------
# Fig D -- MSE history (single image, both runs)
# ---------------------------------------------------------------------------

def fig_mse_history(full_obs_file, part_obs_file, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for fpath, label, color in [
        (full_obs_file, 'Full observation (all 5 species)', '#1a7a1a'),
        (part_obs_file, 'Partial observation (P1, P2, H1)', '#1a3a8f'),
    ]:
        if not os.path.exists(fpath):
            print(f"  Warning: {fpath} not found, skipping.")
            continue
        mse    = np.load(fpath)
        epochs = np.arange(len(mse)) * 5
        ax.semilogy(epochs, mse, lw=2, color=color, label=label)
        ax.scatter(epochs[-1], mse[-1], color=color, s=60, zorder=5)
        ax.annotate(f'{mse[-1]:.4f}', xy=(epochs[-1], mse[-1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color=color)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Parameter MSE  (log scale)', fontsize=11)
    ax.set_title('AD-EnKF convergence: full vs partial observability', fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(fig, 'figD_mse_history', out_dir)

def fig_rmse_history(full_obs_file, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for fpath, label, color in [
        (full_obs_file, 'Full observation (all 5 species)', '#1a7a1a'),
    ]:
        if not os.path.exists(fpath):
            print(f"  Warning: {fpath} not found, skipping.")
            continue
        mse    = np.load(fpath)
        epochs = np.arange(len(mse)) * 5
        ax.semilogy(epochs, np.sqrt(mse), lw=2, color=color, label=label)
        ax.scatter(epochs[-1], np.sqrt(mse[-1]), color=color, s=60, zorder=5)
        ax.annotate(f'{np.sqrt(mse[-1]):.4f}', xy=(epochs[-1], np.sqrt(mse[-1])),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color=color)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Parameter RMSE  (log scale)', fontsize=11)
    ax.set_title('AD-EnKF convergence', fontsize=12)
    plt.tight_layout()
    _save(fig, 'figE_rmse_history', out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Visualise AD-EnKF parameter estimates vs ground truth.'
    )
    parser.add_argument('--est_file',     type=str,   default='./glv_estimated_params.py')
    parser.add_argument('--a_hidden',     type=float, default=None,
                        help='a_hidden value -- shown in titles if provided.')
    parser.add_argument('--data_dir',     type=str,   default='./')
    parser.add_argument('--out',          type=str,   default='./figures')
    parser.add_argument('--fig',          type=str,   default=None,
                        choices=['A', 'B', 'C', 'D'],
                        help='Single figure group to produce. Omit for all.')
    parser.add_argument('--full_obs',     action='store_true',
                        help='Flag: estimates came from a full-observation run.')
    parser.add_argument('--full_obs_mse', type=str,   default='mse_history_full_obs.npy')
    parser.add_argument('--part_obs_mse', type=str,   default='mse_history_part_obs.npy')
    args = parser.parse_args()

    a_h    = args.a_hidden
    to_run = ['A', 'B', 'C', 'D', 'E'] if args.fig is None else [args.fig]

    if any(f in to_run for f in ['A', 'B', 'C']):
        print(f"Loading estimates from: {args.est_file}")
        R_est, A_est = load_estimates(args.est_file)
        a_h_load = a_h if a_h is not None else 0.0
        print(f"Loading data (a_hidden={a_h_load:.2f}) from: {args.data_dir}")
        truth, obs = load_data(a_h, args.data_dir)
        A_true = get_true_A(a_h_load)

    if 'A' in to_run:
        print("\n--- Fig A: Parameter comparison ---")
        fig_parameter_comparison(R_TRUE, A_true, R_est, A_est, a_h, args.out,
                                 full_obs=args.full_obs)

    if 'B' in to_run:
        print("\n--- Fig B: Trajectory comparison ---")
        fig_trajectory_comparison(truth, obs, R_est, A_est, a_h, args.out,
                                  full_obs=args.full_obs)

    if 'C' in to_run:
        print("\n--- Fig C: Projected trajectories ---")
        fig_projected_trajectories(truth, R_est, A_est, a_h, args.out,
                                   full_obs=args.full_obs)

    if 'D' in to_run:
        print("\n--- Fig D: MSE history ---")
        fig_mse_history(args.full_obs_mse, args.part_obs_mse, args.out)
    if 'E' in to_run:
        print("\n--- Fig E: RMSE history ---")
        fig_rmse_history(args.full_obs_mse, args.out)


if __name__ == '__main__':
    main()