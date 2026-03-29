"""
gLV Diagnostics
===============
Classifies the long-run behaviour of the gLV system for each a_hidden value
in the sweep: stable fixed point, stable limit cycle, or neither.

Methods
-------
1. Fixed-point test:
   Integrate to T=500 and T=1000 from the same initial condition.
   If ||x(500) - x(1000)|| < tol the system has settled to a fixed point.

2. Limit cycle test (if fixed-point test fails):
   Integrate a long transient, then record local maxima of each species over
   the next window. If each species has a consistent, finite oscillation
   amplitude the system is on a limit cycle. Period is estimated from
   successive maxima of the total biomass signal.

3. Fixed-point residual:
   For a candidate fixed point x*, verify r_i + sum_j A_ij x*_j ≈ 0 for all i.
   A small residual confirms x* is a true equilibrium of the ODE, not just a
   coincidental stopping point of the integrator.

Usage
-----
    python glv_diagnostics.py                  # full sweep summary table
    python glv_diagnostics.py --a_hidden 0.10  # single value, verbose output
    python glv_diagnostics.py --save_dir ./data --plot
"""

import argparse
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from glv_data_generator import (
    R_TRUE, A_TRUE, EPS, A_HIDDEN_SWEEP,
    get_true_A, glv_rhs
)

SPECIES_NAMES  = ['Producer 1', 'Producer 2', 'Herbivore 3',
                  'Herbivore 4', 'Apex Predator']
SPECIES_COLORS = [
    "#177817",   # Producer 1  — deep forest green
    "#a9c838",   # Producer 2  — light lime green
    '#1a3a8f',   # Herbivore 3 — deep navy blue
    "#6ec1fc",   # Herbivore 4 — light sky blue
    '#e74c3c',   # Apex Predator — red
]


# ---------------------------------------------------------------------------
# 1.  Long integration helper
# ---------------------------------------------------------------------------

def integrate(r, A, x0, T, max_step=0.05):
    """Integrates gLV from x0 for time T. Returns full Solution object."""
    sol = solve_ivp(
        glv_rhs, [0, T], x0, args=(r, A),
        method='RK45', rtol=1e-9, atol=1e-11,
        max_step=max_step, dense_output=True
    )
    return sol


# ---------------------------------------------------------------------------
# 2.  Fixed-point test
# ---------------------------------------------------------------------------

def fixed_point_test(r, A, x0, T1=500.0, T2=1000.0, tol=1e-4):
    """
    Compares x(T1) and x(T2). A small difference indicates convergence to a
    fixed point; a large difference indicates ongoing oscillation.

    Returns
    -------
    is_fp    : bool — True if likely a fixed point
    x_fp     : (N,) candidate fixed point (x at T2)
    residual : (N,) ODE residual at x_fp  (should be ~0 at true equilibrium)
    delta    : scalar ||x(T1) - x(T2)||
    """
    sol1 = integrate(r, A, x0, T1)
    sol2 = integrate(r, A, x0, T2)
    x1   = sol1.y[:, -1]
    x2   = sol2.y[:, -1]

    delta    = np.linalg.norm(x2 - x1)
    is_fp    = delta < tol
    residual = x2 * (r + A @ x2)   # should be 0 at fixed point

    return is_fp, x2, residual, delta


# ---------------------------------------------------------------------------
# 3.  Limit cycle test
# ---------------------------------------------------------------------------

def limit_cycle_test(r, A, x0, T_transient=500.0, T_window=200.0, dt=0.05):
    """
    After discarding a transient, samples the trajectory at high resolution
    and looks for periodic oscillations using peak detection.

    Returns
    -------
    is_lc      : bool — True if consistent oscillations detected
    period     : estimated period (nan if not periodic)
    amplitude  : (N,) per-species oscillation amplitude (max - min over window)
    t_window   : (T,) time array for the analysis window
    X_window   : (N, T) trajectory over the analysis window
    """
    # discard transient
    sol_t = integrate(r, A, x0, T_transient)
    x_post_transient = sol_t.y[:, -1]

    # fine-grained window
    t_eval  = np.arange(0, T_window, dt)
    sol_w   = solve_ivp(
        glv_rhs, [0, T_window], x_post_transient, args=(r, A),
        method='RK45', rtol=1e-9, atol=1e-11,
        t_eval=t_eval, dense_output=False
    )
    t_w = sol_w.t
    X_w = sol_w.y

    # use total biomass as the signal for period estimation
    biomass = X_w.sum(axis=0)
    peaks, _  = find_peaks(biomass, prominence=1e-3)
    amplitude = X_w.max(axis=1) - X_w.min(axis=1)

    if len(peaks) >= 3:
        periods   = np.diff(t_w[peaks])
        cv        = periods.std() / periods.mean()   # coefficient of variation
        is_lc     = cv < 0.05                        # consistent period
        period    = periods.mean() if is_lc else np.nan
    else:
        is_lc  = False
        period = np.nan

    return is_lc, period, amplitude, t_w, X_w


# ---------------------------------------------------------------------------
# 4.  Full classification
# ---------------------------------------------------------------------------

def classify(a_hidden, x0=None, verbose=False):
    """
    Runs both tests and returns a classification dict for one a_hidden value.
    """
    if x0 is None:
        x0 = np.array([1.5, 1.2, 0.8, 0.7, 0.4])

    r = R_TRUE
    A = get_true_A(a_hidden)

    is_fp, x_fp, residual, delta = fixed_point_test(r, A, x0)

    result = {
        'a_hidden'      : a_hidden,
        'is_fixed_point': is_fp,
        'is_limit_cycle': False,
        'x_fp'          : x_fp,
        'fp_residual'   : residual,
        'fp_delta'      : delta,
        'period'        : np.nan,
        'amplitude'     : np.zeros(5),
        't_window'      : None,
        'X_window'      : None,
    }

    if not is_fp:
        is_lc, period, amplitude, t_w, X_w = limit_cycle_test(r, A, x0)
        result['is_limit_cycle'] = is_lc
        result['period']         = period
        result['amplitude']      = amplitude
        result['t_window']       = t_w
        result['X_window']       = X_w

    if verbose:
        _print_result(result)

    return result


def _print_result(res):
    a  = res['a_hidden']
    print(f"\n{'='*55}")
    print(f"  a_hidden = {a:.2f}")
    print(f"{'='*55}")
    print(f"  Fixed-point test:  ||x(500)-x(1000)|| = {res['fp_delta']:.2e}")

    if res['is_fixed_point']:
        print(f"  Classification:    STABLE FIXED POINT")
        print(f"  Equilibrium x*:    {np.round(res['x_fp'], 4)}")
        print(f"  ODE residual:      {np.round(res['fp_residual'], 2e-6)}")
        max_res = np.abs(res['fp_residual']).max()
        print(f"  Max |residual|:    {max_res:.2e}  "
              f"({'OK' if max_res < 1e-3 else 'LARGE — may not be true eq.'})")
    elif res['is_limit_cycle']:
        print(f"  Classification:    STABLE LIMIT CYCLE")
        print(f"  Estimated period:  {res['period']:.3f} time units")
        print(f"  Oscillation amplitudes:")
        for i, (name, amp) in enumerate(zip(SPECIES_NAMES, res['amplitude'])):
            print(f"    {name:15s}: {amp:.4f}")
    else:
        print(f"  Classification:    INDETERMINATE (neither FP nor clean LC)")
        print(f"  Possible chaos or very long transient — inspect trajectory.")


# ---------------------------------------------------------------------------
# 5.  Summary table
# ---------------------------------------------------------------------------

def summary_table(sweep=None):
    """Prints a compact summary table for all sweep values."""
    if sweep is None:
        sweep = A_HIDDEN_SWEEP

    x0 = np.array([1.5, 1.2, 0.8, 0.7, 0.4])

    print(f"\n{'a_hidden':>10}  {'Classification':>18}  "
          f"{'||delta||':>12}  {'period':>10}  {'x3_eq or x3_amp':>18}")
    print("-" * 78)

    results = []
    for a_h in sweep:
        res = classify(a_h, x0=x0, verbose=False)
        results.append(res)

        if res['is_fixed_point']:
            classif = "Fixed point"
            period  = "—"
            x3_info = f"{res['x_fp'][2]:.4f}"
        elif res['is_limit_cycle']:
            classif = "Limit cycle"
            period  = f"{res['period']:.3f}"
            x3_info = f"±{res['amplitude'][2]/2:.4f}"
        else:
            classif = "Indeterminate"
            period  = "?"
            x3_info = "?"

        print(f"{a_h:>10.2f}  {classif:>18}  "
              f"{res['fp_delta']:>12.2e}  {period:>10}  {x3_info:>18}")

    return results


# ---------------------------------------------------------------------------
# 6.  Diagnostic plots
# ---------------------------------------------------------------------------

def plot_diagnostics(results, save_dir='.'):
    """
    Produces two figures:

    Figure 1 — Long-run trajectories for each a_hidden value.
               Useful for visually confirming fixed point vs limit cycle.

    Figure 2 — Herb3 long-run population vs a_hidden.
               Illustrates the indirect competition effect quantitatively.
               For fixed points: single line (equilibrium value).
               For limit cycles: shaded band (min to max).
    """
    _plot_trajectories(results, save_dir)
    _plot_herb3_vs_ahidden(results, save_dir)


def _plot_trajectories(results, save_dir):
    n     = len(results)
    fig   = plt.figure(figsize=(5 * n, 4))
    gs    = gridspec.GridSpec(1, n, figure=fig, wspace=0.35)

    for col, res in enumerate(results):
        ax = fig.add_subplot(gs[0, col])
        a  = res['a_hidden']

        if res['t_window'] is not None:
            t = res['t_window']
            X = res['X_window']
        else:
            # Re-integrate a short noiseless window around the fixed point
            r_  = R_TRUE
            A_  = get_true_A(a)
            x0_ = res['x_fp']
            sol = solve_ivp(glv_rhs, [0, 50], x0_, args=(r_, A_),
                            method='RK45', rtol=1e-9, atol=1e-11,
                            t_eval=np.linspace(0, 50, 500))
            t = sol.t
            X = sol.y

        for i in range(5):
            ax.plot(t, X[i], color=SPECIES_COLORS[i],
                    label=SPECIES_NAMES[i], lw=1.4)

        ax.set_title(f'$a_{{\\mathrm{{hidden}}}}={a:.2f}$', fontsize=11)
        ax.set_xlabel('Time', fontsize=9)
        if col == 0:
            ax.set_ylabel('Population', fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)

    # shared legend below figure
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle('Long-run trajectories across misspecification sweep',
                 fontsize=13, y=1.02)
    plt.savefig(f'{save_dir}/diag_trajectories.pdf',
                bbox_inches='tight', dpi=150)
    plt.savefig(f'{save_dir}/diag_trajectories.png',
                bbox_inches='tight', dpi=150)
    print(f"Saved: {save_dir}/diag_trajectories.pdf")
    plt.show()


def _plot_herb3_vs_ahidden(results, save_dir):
    fig, ax = plt.subplots(figsize=(7, 4))

    a_vals, x3_lo, x3_hi = [], [], []

    for res in results:
        a = res['a_hidden']
        a_vals.append(a)
        if res['is_fixed_point']:
            x3_lo.append(res['x_fp'][2])
            x3_hi.append(res['x_fp'][2])
        elif res['is_limit_cycle']:
            X3 = res['X_window'][2]
            x3_lo.append(X3.min())
            x3_hi.append(X3.max())
        else:
            x3_lo.append(np.nan)
            x3_hi.append(np.nan)

    a_vals = np.array(a_vals)
    x3_lo  = np.array(x3_lo)
    x3_hi  = np.array(x3_hi)
    x3_mid = (x3_lo + x3_hi) / 2

    ax.fill_between(a_vals, x3_lo, x3_hi,
                    alpha=0.25, color=SPECIES_COLORS[2],
                    label='Population range (min–max)')
    ax.plot(a_vals, x3_mid, 'o-',
            color=SPECIES_COLORS[2], lw=2, ms=6,
            label='Herbivore 3 (mid / equilibrium)')

    # also show herb4 for contrast
    x4_lo, x4_hi = [], []
    for res in results:
        if res['is_fixed_point']:
            x4_lo.append(res['x_fp'][3])
            x4_hi.append(res['x_fp'][3])
        elif res['is_limit_cycle']:
            X4 = res['X_window'][3]
            x4_lo.append(X4.min())
            x4_hi.append(X4.max())
        else:
            x4_lo.append(np.nan)
            x4_hi.append(np.nan)

    x4_lo  = np.array(x4_lo)
    x4_hi  = np.array(x4_hi)
    x4_mid = (x4_lo + x4_hi) / 2

    ax.fill_between(a_vals, x4_lo, x4_hi,
                    alpha=0.15, color=SPECIES_COLORS[3])
    ax.plot(a_vals, x4_mid, 's--',
            color=SPECIES_COLORS[3], lw=1.8, ms=6,
            label='Herbivore 4 (mid / equilibrium)')

    ax.axvline(0, color='gray', lw=1, ls=':', label='Well-specified baseline')
    ax.set_xlabel('Hidden interaction strength $a_{\\mathrm{hidden}}$', fontsize=12)
    ax.set_ylabel('Population', fontsize=12)
    ax.set_title('Indirect competition: Herb3 suppressed as $a_{\\mathrm{hidden}}$ grows',
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/diag_competition.pdf', bbox_inches='tight', dpi=150)
    plt.savefig(f'{save_dir}/diag_competition.png', bbox_inches='tight', dpi=150)
    print(f"Saved: {save_dir}/diag_competition.pdf")
    plt.show()


# ---------------------------------------------------------------------------
# 7.  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Classify gLV dynamics (fixed point vs limit cycle) for each sweep value.'
    )
    parser.add_argument('--a_hidden', type=float, default=None,
                        help='Single a_hidden value for verbose output. '
                             'Omit to run summary table for full sweep.')
    parser.add_argument('--plot',     action='store_true',
                        help='Generate and save diagnostic plots.')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='Directory for saved figures (default: current dir).')
    args = parser.parse_args()

    if args.a_hidden is not None:
        res = classify(args.a_hidden, verbose=True)
        if args.plot:
            plot_diagnostics([res], save_dir=args.save_dir)
    else:
        results = summary_table()
        if args.plot:
            plot_diagnostics(results, save_dir=args.save_dir)


if __name__ == '__main__':
    main()
