"""
gLV Visualisation
=================
Produces all figures based on the generated gLV dataset.
Reads from pre-generated .npz files in ./data/ — run glv_data_generator.py first.

Figures produced
----------------
Fig 1 — Food web diagram
    Directed graph with edge weights shown as thickness AND numeric labels.
    Left panel: filter model (assumed). Right panel: true system (with hidden edge).

Fig 2 — Time series panel (well-specified vs most misspecified)
    Side-by-side for a_hidden=0.00 and a_hidden=0.20.
    True trajectories (solid) overlaid with noisy observations (scatter).

Fig 3 — Long-run population shift across sweep
    All species (left) and herbivore competition zoom (right).

Fig 4 — Trajectories (noiseless, Producer 1 vs Herbivore 3, Producer 1 vs Herbivore 4)
    Uses clean noiseless integration of model so the fixed-point attractor is visible.

Usage
-----
    python glv_visualize.py                          # all figures
    python glv_visualize.py --fig 1                  # single figure
    python glv_visualize.py --data_dir ./data --out ./figures
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

from glv_data_generator import (
    R_TRUE, A_TRUE, EPS, A_HIDDEN_SWEEP,
    get_true_A, glv_rhs
)

# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

SPECIES_NAMES  = ['Producer 1', 'Producer 2', 'Herbivore 3',
                  'Herbivore 4', 'Apex Predator']
SPECIES_SHORT  = ['P1', 'P2', 'H3', 'H4', 'AP']

# Deliberately distinguishable: dark/light split within each trophic group
SPECIES_COLORS = [
    "#177817",   # Producer 1  — deep forest green
    "#a9c838",   # Producer 2  — light lime green
    '#1a3a8f',   # Herbivore 3 — deep navy blue
    "#6ec1fc",   # Herbivore 4 — light sky blue
    '#e74c3c',   # Apex Predator — red
]

# Colors for the a_hidden sweep panels (sequential, colorblind-friendly)
SWEEP_COLORS = ['#2c3e50', '#2471a3', '#1e8449', "#fe9d01", '#c0392b']

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
# Helper: load datasets
# ---------------------------------------------------------------------------

def load(a_hidden, data_dir='./data'):
    label = f"ahidden{a_hidden:.2f}".replace('.', 'p')
    truth = np.load(os.path.join(data_dir, f'glv_{label}_truth.npz'))
    obs   = np.load(os.path.join(data_dir, f'glv_{label}_obs.npz'))
    return truth, obs


def load_all(data_dir='./data'):
    datasets = []
    for a_h in A_HIDDEN_SWEEP:
        try:
            truth, obs = load(a_h, data_dir)
            datasets.append((a_h, truth, obs))
        except FileNotFoundError:
            print(f"Warning: data for a_hidden={a_h:.2f} not found, skipping.")
    return datasets


def _save(fig, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ['pdf', 'png']:
        path = os.path.join(out_dir, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 1 — Food web diagram with edge weights
# ---------------------------------------------------------------------------

def fig_food_web(a_hidden_shown=0.15, out_dir='.'):
    """
    Two-panel food web diagram.
    Edge thickness proportional to |a_ij|; numeric weight shown as edge label.
    Positive edges (gain) in green, negative (loss/competition) in red.
    Hidden interaction shown dashed.
    """
    try:
        import networkx as nx
    except ImportError:
        print("networkx not installed. Run: pip install networkx")
        return

    # Fixed trophic layout (same in both panels for direct comparison)
    pos = {
        0: (-1.2,  0.0),   # P1
        1: ( 1.2,  0.0),   # P2
        2: (-1.2,  1.8),   # H3
        3: ( 1.2,  1.8),   # H4
        4: ( 0.0,  3.4),   # AP
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    titles = [
        'Filter model (assumed)',
        f'True system  ($a_{{\\mathrm{{hidden}}}}={a_hidden_shown:.2f}$)'
    ]

    for ax, show_hidden, title in zip(axes, [False, True], titles):
        A = get_true_A(a_hidden_shown if show_hidden else 0.0)
        G = nx.DiGraph()
        G.add_nodes_from(range(5))

        solid_pos, solid_neg, dashed_pos, dashed_neg = [], [], [], []
        solid_pos_w, solid_neg_w, dashed_pos_w, dashed_neg_w = [], [], [], []
        edge_label_dict = {}

        for i in range(5):
            for j in range(5):
                val = A[i, j]
                if abs(val) < 1e-9 or i == j:
                    continue
                G.add_edge(j, i)
                is_hidden = show_hidden and (A_TRUE[i, j] == 0.0)
                w = abs(val) * 14   # scale for visibility

                if is_hidden:
                    if val > 0:
                        dashed_pos.append((j, i)); dashed_pos_w.append(w)
                    else:
                        dashed_neg.append((j, i)); dashed_neg_w.append(w)
                else:
                    if val > 0:
                        solid_pos.append((j, i)); solid_pos_w.append(w)
                    else:
                        solid_neg.append((j, i)); solid_neg_w.append(w)

                edge_label_dict[(j, i)] = f'{val:+.3f}'

        rad = 'arc3,rad=0.12'

        # draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_color=SPECIES_COLORS,
                               node_size=1400, alpha=0.92, linewidths=1.5,
                               edgecolors='white')
        nx.draw_networkx_labels(G, pos, ax=ax,
                                labels={i: SPECIES_SHORT[i] for i in range(5)},
                                font_size=12, font_weight='bold',
                                font_color='white')

        kw = dict(ax=ax, arrows=True, arrowsize=16,
                  connectionstyle=rad, min_source_margin=20, min_target_margin=20)

        if solid_pos:
            nx.draw_networkx_edges(G, pos, edgelist=solid_pos,
                                   edge_color='#1e8449', width=solid_pos_w,
                                   style='solid', **kw)
        if solid_neg:
            nx.draw_networkx_edges(G, pos, edgelist=solid_neg,
                                   edge_color='#c0392b', width=solid_neg_w,
                                   style='solid', **kw)
        if dashed_pos:
            nx.draw_networkx_edges(G, pos, edgelist=dashed_pos,
                                   edge_color='#1e8449', width=dashed_pos_w,
                                   style='dashed', **kw)
        if dashed_neg:
            nx.draw_networkx_edges(G, pos, edgelist=dashed_neg,
                                   edge_color='#c0392b', width=dashed_neg_w,
                                   style='dashed', **kw)

        # edge weight labels — offset slightly so they don't sit on the arrow

        for (u, v), label in edge_label_dict.items():
            val = A[v, u]
            color = '#1e8449' if val > 0 else '#c0392b'
            nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): label},
                                        ax=ax, font_size=7.5,
                                        font_color=color,
                                        bbox=dict(boxstyle='round,pad=0.15',
                                                fc='white', alpha=0.85, ec='none'),
                                        label_pos=0.25)
        

        ax.set_title(title, fontsize=12, pad=14)
        ax.axis('off')

        # trophic level annotations
        for level, label_text in [(0.0, 'Producers'),
                                   (1.8, 'Herbivores'),
                                   (3.4, 'Apex predator')]:
            ax.text(-2.2, level, label_text, fontsize=9,
                    color='gray', va='center', style='italic')

    # legend
    legend_elements = [
        Line2D([0], [0], color='#1e8449', lw=2.5, label='Positive effect (gain)'),
        Line2D([0], [0], color='#c0392b', lw=2.5, label='Negative effect (loss / competition)'),
        Line2D([0], [0], color='#888888', lw=2.0, ls='--',
               label='Hidden interaction (unknown to filter)'),
        Line2D([0], [0], color='black',   lw=1.5,
               label='Edge width $\\propto$ interaction strength $|a_{ij}|$'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle('Generalised Lotka-Volterra food web', fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, 'fig1_food_web', out_dir)


# ---------------------------------------------------------------------------
# Fig 2 — Time series
# ---------------------------------------------------------------------------

def fig_time_series(data_dir='./data', out_dir='.'):
    """
    Side-by-side time series for a_hidden=0.00 and a_hidden=0.20.
    Top row: all true trajectories (observed solid, unobserved dashed).
    Bottom row: observed species only, true vs noisy observations.
    """
    cases = [0.00, 0.20]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    for col, a_h in enumerate(cases):
        truth, obs = load(a_h, data_dir)
        t       = truth['t']
        X       = truth['X']
        Y       = obs['Y']
        H       = obs['H']
        obs_idx = np.where(H.sum(axis=0) > 0)[0]

        title = ('Well-specified  ($a_{\\mathrm{hidden}}=0$)'
                 if a_h == 0.0 else
                 f'Misspecified  ($a_{{\\mathrm{{hidden}}}}={a_h:.2f}$)')

        # top: all true trajectories
        ax = axes[0, col]
        for i in range(5):
            ls  = '-'   if i in obs_idx else '--'
            alp = 1.0   if i in obs_idx else 0.55
            ax.plot(t, X[i], color=SPECIES_COLORS[i], lw=1.8,
                    ls=ls, alpha=alp, label=SPECIES_NAMES[i])
        ax.set_title(title, fontsize=11)
        if col == 0:
            ax.set_ylabel('True population', fontsize=10)
        ax.set_ylim(bottom=0)

        # bottom: observations vs truth for observed species only
        ax = axes[1, col]
        for k, i in enumerate(obs_idx):
            ax.plot(t, X[i], color=SPECIES_COLORS[i], lw=1.8, alpha=0.6)
            ax.scatter(t[::2], Y[k, ::2], color=SPECIES_COLORS[i],
                       s=10, alpha=0.85, zorder=3)
        if col == 0:
            ax.set_ylabel('Observed population', fontsize=10)
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylim(bottom=0)

        # annotate herb3 suppression on the misspecified column
        if a_h > 0:
            x3_mean = X[2, int(0.5 * len(t)):].mean()
            ax.annotate(
                'Herb3 suppressed\nby hidden competition',
                xy=(70, x3_mean + 0.02),
                xytext=(55, x3_mean + 0.35),
                fontsize=8.5, color=SPECIES_COLORS[2],
                arrowprops=dict(arrowstyle='->', color=SPECIES_COLORS[2], lw=1.2)
            )

    # single shared legend — species colors + line style meaning
    species_handles = [
        Line2D([0], [0], color=SPECIES_COLORS[i], lw=2.2, label=SPECIES_NAMES[i])
        for i in range(5)
    ]
    style_handles = [
        Line2D([0], [0], color='gray', lw=2,       label='True (observed species)'),
        Line2D([0], [0], color='gray', lw=2, ls='--', label='True (unobserved species)'),
        Line2D([0], [0], color='gray', lw=0, marker='o', ms=5,
               label='Noisy observation $y_k$'),
    ]
    fig.legend(handles=species_handles + style_handles,
               loc='lower center', ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.07))

    fig.suptitle('Population dynamics: true trajectories and noisy observations',
                 fontsize=13)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, 'fig2_time_series', out_dir)


# ---------------------------------------------------------------------------
# Fig 3 — Long-run population vs a_hidden
# ---------------------------------------------------------------------------

def fig_competition_effect(data_dir='./data', out_dir='.'):
    """
    Long-run mean population (last 30% of trajectory) vs a_hidden.
    Left: all species. Right: herbivore competition zoom.
    """
    datasets = load_all(data_dir)
    a_vals   = np.array([d[0] for d in datasets])
    X_means  = np.zeros((len(datasets), 5))
    X_stds   = np.zeros((len(datasets), 5))

    for idx, (_, truth, _) in enumerate(datasets):
        X   = truth['X']
        T   = X.shape[1]
        win = X[:, int(0.7 * T):]
        X_means[idx] = win.mean(axis=1)
        X_stds[idx]  = win.std(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: all species
    ax = axes[0]
    for i in range(5):
        ax.plot(a_vals, X_means[:, i], 'o-',
                color=SPECIES_COLORS[i], lw=2, ms=6, label=SPECIES_NAMES[i])
        ax.fill_between(a_vals,
                        X_means[:, i] - X_stds[:, i],
                        X_means[:, i] + X_stds[:, i],
                        color=SPECIES_COLORS[i], alpha=0.12)
    ax.axvline(0, color='gray', lw=1, ls=':', alpha=0.7)
    ax.set_xlabel('Hidden interaction strength $a_{\\mathrm{hidden}}$', fontsize=11)
    ax.set_ylabel('Long-run mean population', fontsize=11)
    ax.set_title('All species — long-run mean $\\pm$ std', fontsize=11)
    ax.legend(fontsize=9)

    # right: herbivore zoom
    ax = axes[1]
    for i in [2, 3]:
        ax.plot(a_vals, X_means[:, i], 'o-',
                color=SPECIES_COLORS[i], lw=2.5, ms=8,
                label=SPECIES_NAMES[i])
        ax.fill_between(a_vals,
                        X_means[:, i] - X_stds[:, i],
                        X_means[:, i] + X_stds[:, i],
                        color=SPECIES_COLORS[i], alpha=0.18)

    # annotate competitive asymmetry at rightmost sweep point
    y_lo = X_means[-1, 2]
    y_hi = X_means[-1, 3]
    ax.annotate('', xy=(a_vals[-1] + 0.005, y_hi),
                xytext=(a_vals[-1] + 0.005, y_lo),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(a_vals[-1] + 0.012, (y_lo + y_hi) / 2,
            'Competitive\nasymmetry', fontsize=8.5, va='center')

    ax.axvline(0, color='gray', lw=1, ls=':', alpha=0.7,
               label='Well-specified baseline')
    ax.set_xlabel('Hidden interaction strength $a_{\\mathrm{hidden}}$', fontsize=11)
    ax.set_ylabel('Long-run mean population', fontsize=11)
    ax.set_title('Indirect competition between herbivores', fontsize=11)
    ax.legend(fontsize=9)

    fig.suptitle('Effect of hidden interaction on long-run population dynamics',
                 fontsize=13)
    plt.tight_layout()
    _save(fig, 'fig3_competition_effect', out_dir)


# ---------------------------------------------------------------------------
# Fig 4 — Trajectories (noiseless)
# ---------------------------------------------------------------------------

def fig_trajectories(data_dir='./data', out_dir='.'):
    """
    Noiseless projected trajectories: 2 rows x 5 columns.
    Row 0: Producer 1 vs Herbivore 3  (suppressed by hidden competition)
    Row 1: Producer 1 vs Herbivore 4  (benefits from hidden competition)
    Each column is one sweep value of a_hidden.
    Transient faded; post-transient trajectory in full color.
    """
    T_total     = 200.0
    T_transient = 80.0
    dt          = 0.02
    x0          = np.array([1.5, 1.2, 0.8, 0.7, 0.4])

    n   = len(A_HIDDEN_SWEEP)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 8))

    row_specs = [
        (2, 'Herbivore 3'),
        (3, 'Herbivore 4'),
    ]

    for col, (a_h, sweep_color) in enumerate(zip(A_HIDDEN_SWEEP, SWEEP_COLORS)):
        A      = get_true_A(a_h)
        t_eval = np.arange(0, T_total, dt)

        sol = solve_ivp(
            glv_rhs, [0, T_total], x0, args=(R_TRUE, A),
            method='RK45', rtol=1e-9, atol=1e-11,
            t_eval=t_eval, dense_output=False
        )
        x1    = sol.y[0]   # Producer 1
        t     = sol.t
        split = np.searchsorted(t, T_transient)

        for row, (species_idx, species_name) in enumerate(row_specs):
            ax = axes[row, col]
            xi = sol.y[species_idx]

            ax.plot(x1[:split], xi[:split],
                    color=sweep_color, lw=0.7, alpha=0.25)
            ax.plot(x1[split:], xi[split:],
                    color=sweep_color, lw=1.2, alpha=0.9)
            ax.scatter(x1[-1], xi[-1], color=sweep_color,
                       s=80, zorder=5, marker='*')
            ax.scatter(x1[0],  xi[0],  color=sweep_color,
                       s=40, zorder=5, marker='o', alpha=0.5)

            ax.set_xlabel('Producer 1', fontsize=9)
            if col == 0:
                ax.set_ylabel(species_name, fontsize=9)
            if row == 0:
                ax.set_title(f'$a_{{\\mathrm{{hidden}}}}={a_h:.2f}$', fontsize=10)
            ax.tick_params(labelsize=8)

    legend_handles = [
        Line2D([0], [0], color='gray', lw=0.8, alpha=0.3, label='Transient'),
        Line2D([0], [0], color='gray', lw=1.5,             label='Post-transient'),
        Line2D([0], [0], color='gray', lw=0, marker='*', ms=8, label='Fixed point'),
        Line2D([0], [0], color='gray', lw=0, marker='o', ms=5,
               alpha=0.5,                                  label='Initial condition'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        'Projected trajectories (noiseless ODE, transient faded)\n'
        'Top: Producer 1 vs Herbivore 3   |   Bottom: Producer 1 vs Herbivore 4',
        fontsize=12
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, 'fig4_trajectories', out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate all gLV figures for EP6 project.'
    )
    parser.add_argument('--fig',      type=int, default=None,
                        choices=[1, 2, 3, 4],
                        help='Single figure number. Omit for all.')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--out',      type=str, default='./figures')
    args = parser.parse_args()

    figs = {
        1: lambda: fig_food_web(out_dir=args.out),
        2: lambda: fig_time_series(data_dir=args.data_dir, out_dir=args.out),
        3: lambda: fig_competition_effect(data_dir=args.data_dir, out_dir=args.out),
        4: lambda: fig_trajectories(data_dir=args.data_dir, out_dir=args.out),
    }

    to_run = [args.fig] if args.fig is not None else [1, 2, 3, 4]
    for f in to_run:
        print(f"\n--- Figure {f} ---")
        figs[f]()


if __name__ == '__main__':
    main()
