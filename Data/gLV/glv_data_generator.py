"""
Generalized Lotka-Volterra (gLV) Data Generator
=================================================
Generates synthetic observations for a 5-species food web governed by:

    dx_i/dt = x_i * (r_i + sum_j a_ij * x_j),   i = 1, ..., 5

Trophic structure:
    Species 1, 2  : Producers
    Species 3     : Herbivore — eats Producer 1 only
    Species 4     : Herbivore — eats Producer 2 (and secretly Producer 1 when a_hidden > 0)
    Species 5     : Apex predator — eats both herbivores

The filter's model always assumes Herb4 eats only Producer 2. The true data-
generating system additionally gives Herb4 access to Producer 1 with strength
a_hidden. This creates emergent indirect competition between Herb3 and Herb4
through their shared resource (Producer 1) that the filter cannot represent.

Experiment design (misspecification sweep):
    a_hidden = 0.0   : well-specified baseline (no hidden interaction)
    a_hidden > 0     : increasing misspecification severity

    For each value of a_hidden, one dataset is generated and saved. The filter
    always uses A_TRUE's sparsity mask, so the hidden entries are never
    estimated — only their effect on the other parameter estimates is observed.

Usage
-----
    python glv_data_generator.py                    # runs full sweep (partial obs: P1, P2, H3)
    python glv_data_generator.py --a_hidden 0.0     # single value
    python glv_data_generator.py --observe_all      # all 5 species observed
    python glv_data_generator.py --a_hidden 0.15 --observe_all --save_dir ./data_full_obs

Output (saved to ./data/ or --save_dir)
---------------------------------------
    glv_ahidden<value>_truth.npz  : true trajectories, parameters, time grid
    glv_ahidden<value>_obs.npz    : noisy observations (Y, H, observed_species)

For visualisation and diagnostics, see:
    glv_diagnostics.py   — equilibrium verification, limit cycle detection
    glv_visualize.py     — all publication-quality figures
"""

import argparse
import os
import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# 1.  Ground-truth base parameters
# ---------------------------------------------------------------------------

# Intrinsic growth rates.
# Producers are self-sustaining (r > 0); consumers starve without prey (r < 0).
# Herb3's r is only slightly negative so it remains viable even when Producer 1
# is partially depleted by the hidden interaction at higher a_hidden values.
R_TRUE = np.array([1.2, 1.0, -0.05, -0.3, -0.1])

# Conversion efficiency: fraction of consumed biomass converted to predator growth.
# Encodes the ecological rule that energy is lost at each trophic transfer.
EPS = 0.75

# Base interaction matrix — the filter ALWAYS assumes this sparsity pattern.
# Convention: A[i, j] = effect of species j on species i's per-capita growth rate.
#   A[i, j] > 0  : species j benefits species i  (prey->predator gain)
#   A[i, j] < 0  : species j harms species i     (predation loss, competition, self-reg)
# Predation pairs satisfy: A[predator, prey] = +alpha*EPS, A[prey, predator] = -alpha
# fmt: off
A_TRUE = np.array([
    [-0.50,    0.0,   -0.25,   0.0,     0.0 ],  # prod1:    self-reg; loss to herb3
    [  0.0,   -0.40,   0.0,   -0.40,    0.0 ],  # prod2:    self-reg; loss to herb4
    [  0.215,  0.0,   -0.10,   0.0,    -0.25 ],  # herb3:    gain from prod1; loss to predator
    [  0.0,    0.2625, 0.0,   -0.15,   -0.25 ],  # herb4:    gain from prod2; loss to predator
    [  0.0,    0.0,    0.215,  0.215,  -0.1 ],  # predator: gain from herb3 and herb4
])
# fmt: on

# Sweep values for the hidden interaction strength.
# a_hidden = 0.0 is the well-specified baseline.
# Upper limit of 0.20 keeps all five species viable across the full sweep.
A_HIDDEN_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20]

# Observation settings
OBS_NOISE_STD    = 0.05
OBSERVED_SPECIES = [0, 1, 2]   # Default: Producer 1, Producer 2, Herbivore 3 (partial obs)
N_SPECIES        = 5


# ---------------------------------------------------------------------------
# 2.  Parameter construction
# ---------------------------------------------------------------------------

def get_true_A(a_hidden):
    """
    Returns the true interaction matrix used for data generation.

    When a_hidden > 0, Herb4 also feeds on Producer 1:
        A[3, 0] = +a_hidden * EPS   (herb4 gains, discounted by conversion efficiency)
        A[0, 3] = -a_hidden         (producer 1 suffers the full biomass loss)

    The filter's forward model always uses A_TRUE's sparsity, so these entries
    are invisible to the estimator regardless of their magnitude.
    """
    A = A_TRUE.copy()
    if a_hidden > 0.0:
        A[3, 0] =  a_hidden * EPS
        A[0, 3] = -a_hidden
    return A


def get_theta(A, r=None):
    """
    Extracts the free parameter vector theta that the filter will estimate,
    using A_TRUE's sparsity pattern as the mask.

    The hidden entries (A[0,3] and A[3,0]) are never included regardless of
    a_hidden — this is the misspecification.

    Returns
    -------
    theta        : (18,) array of free parameters [r_1..r_5, nonzero A entries]
    theta_labels : list of human-readable parameter names
    """
    if r is None:
        r = R_TRUE
    theta, labels = [], []

    for i in range(5):
        theta.append(r[i])
        labels.append(f"r_{i+1}")

    mask = (A_TRUE != 0.0)
    for i in range(5):
        for j in range(5):
            if mask[i, j]:
                theta.append(A[i, j])
                labels.append(f"a_{i+1}{j+1}")

    return np.array(theta), labels


# ---------------------------------------------------------------------------
# 3.  ODE
# ---------------------------------------------------------------------------

def glv_rhs(t, x, r, A):
    """Generalised Lotka-Volterra RHS. Clips populations at zero."""
    x = np.maximum(x, 0.0)
    return x * (r + A @ x)


# ---------------------------------------------------------------------------
# 4.  Coexistence check
# ---------------------------------------------------------------------------

def check_coexistence(r, A, x0, T=500.0, tol=1e-2):
    """
    Integrates the noiseless ODE for time T and checks that all populations
    remain bounded and non-extinct.

    Note: a passing result indicates coexistence but does not distinguish
    between a stable fixed point and a stable limit cycle. Use
    glv_diagnostics.py for a definitive classification.

    Returns
    -------
    coexists : bool
    x_final  : (N,) state at time T
    """
    sol = solve_ivp(
        glv_rhs, [0, T], x0, args=(r, A),
        method='RK45', rtol=1e-8, atol=1e-10, max_step=0.1
    )
    x_final = sol.y[:, -1]
    coexists = (
        sol.success
        and np.all(x_final > tol)
        and np.all(x_final < 1e4)
    )
    return coexists, x_final


# ---------------------------------------------------------------------------
# 5.  Trajectory generation
# ---------------------------------------------------------------------------

def generate_trajectory(r, A, x0, t_span, dt, process_noise_std=0.0, rng=None):
    """
    Integrates the gLV ODE with optional log-space process noise injected at
    each observation step.

    Log-space noise: log(x) += N(0, sigma^2) keeps all populations positive
    and is a standard choice for multiplicative noise in population dynamics.

    Returns
    -------
    t_grid : (T,)    observation times
    X_true : (N, T)  true state at each observation time
    """
    if rng is None:
        rng = np.random.default_rng()

    t_start, t_end = t_span
    t_grid = np.arange(t_start, t_end + dt * 0.5, dt)
    T = len(t_grid)
    N = len(x0)

    X_true          = np.zeros((N, T))
    X_true[:, 0]    = x0.copy()
    x_current       = x0.copy()

    for k in range(1, T):
        sol = solve_ivp(
            glv_rhs,
            [t_grid[k-1], t_grid[k]],
            x_current,
            args=(r, A),
            method='RK45',
            rtol=1e-8, atol=1e-10,
            dense_output=False
        )
        x_next = np.maximum(sol.y[:, -1], 1e-8)

        if process_noise_std > 0.0:
            log_x  = np.log(x_next)
            log_x += rng.normal(0.0, process_noise_std, size=N)
            x_next = np.exp(log_x)

        X_true[:, k] = x_next
        x_current    = x_next.copy()

    return t_grid, X_true


# ---------------------------------------------------------------------------
# 6.  Observation generation
# ---------------------------------------------------------------------------

def generate_observations(X_true, obs_species, obs_noise_std, rng=None):
    """
    Generates partial, noisy observations.

    Observation model:  y_k = H x_k + xi_k,   xi_k ~ N(0, obs_noise_std^2 I)

    Returns
    -------
    Y : (M, T)  noisy observations
    H : (M, N)  observation operator
    """
    if rng is None:
        rng = np.random.default_rng()

    N, T = X_true.shape
    M    = len(obs_species)

    H = np.zeros((M, N))
    for i, s in enumerate(obs_species):
        H[i, s] = 1.0

    Y = H @ X_true + rng.normal(0.0, obs_noise_std, size=(M, T))
    Y = np.maximum(Y, 0.0)

    return Y, H


# ---------------------------------------------------------------------------
# 7.  Main generation routine
# ---------------------------------------------------------------------------

def generate_experiment(a_hidden, seed, dt=0.5, t_end=100.0,
                        process_noise_std=0.00, obs_noise_std=OBS_NOISE_STD,
                        save_dir='./data', observed_species=None):
    """
    Generates and saves one complete dataset for a given a_hidden value.

    Parameters
    ----------
    a_hidden          : hidden interaction strength (0.0 = well-specified)
    seed              : integer random seed for reproducibility
    dt                : time between observations
    t_end             : total simulation length
    process_noise_std : std of log-space process noise added to true trajectory
    obs_noise_std     : std of additive Gaussian observation noise
    save_dir          : directory to write output files
    observed_species  : list of species indices to observe (0..4). If None, uses
                        OBSERVED_SPECIES (default partial: [0,1,2]). Use
                        list(range(5)) or observe_all=True in CLI for all species.

    Output files
    ------------
    glv_ahidden<X>_truth.npz : t, X, r, A, A_base, a_hidden, theta,
                                theta_labels, x0, eps, process_noise_std
    glv_ahidden<X>_obs.npz   : t, Y, H, obs_noise_std, observed_species
    """
    if observed_species is None:
        observed_species = OBSERVED_SPECIES

    rng   = np.random.default_rng(seed)
    label = f"ahidden{a_hidden:.2f}".replace('.', 'p')

    A_true             = get_true_A(a_hidden)
    theta_true, labels = get_theta(A_true)
    x0                 = np.array([1.5, 1.2, 0.8, 0.7, 0.4])

    print(f"[a_hidden={a_hidden:.2f}] Checking coexistence...", end=' ')
    coexists, x_final = check_coexistence(R_TRUE, A_true, x0)
    status = (f"OK  (long-run state ~ {np.round(x_final, 3)})" if coexists
              else "WARNING: extinction or blow-up detected — check parameters.")
    print(status)

    print(f"[a_hidden={a_hidden:.2f}] Integrating ODE (T={t_end}, dt={dt})...")
    t_grid, X_true = generate_trajectory(
        R_TRUE, A_true, x0, (0.0, t_end), dt,
        process_noise_std=process_noise_std, rng=rng
    )

    Y, H = generate_observations(X_true, observed_species, obs_noise_std, rng=rng)

    os.makedirs(save_dir, exist_ok=True)
    truth_path = os.path.join(save_dir, f'glv_{label}_truth.npz')
    obs_path   = os.path.join(save_dir, f'glv_{label}_obs.npz')

    np.savez(truth_path,
             t=t_grid, X=X_true,
             r=R_TRUE, A=A_true, A_base=A_TRUE,
             a_hidden=a_hidden,
             theta=theta_true, theta_labels=np.array(labels),
             x0=x0, eps=EPS,
             process_noise_std=process_noise_std)

    np.savez(obs_path,
             t=t_grid, Y=Y, H=H,
             obs_noise_std=obs_noise_std,
             observed_species=np.array(observed_species))

    n_obs_species = len(observed_species)
    print(f"[a_hidden={a_hidden:.2f}] Saved: {truth_path}")
    print(f"[a_hidden={a_hidden:.2f}] Observed species: {n_obs_species}/5  {observed_species}")
    print(f"[a_hidden={a_hidden:.2f}] theta_true ({len(theta_true)} params):")
    for lbl, val in zip(labels, theta_true):
        print(f"    {lbl:8s} = {val: .4f}")

    return t_grid, X_true, Y, H, theta_true, labels


# ---------------------------------------------------------------------------
# 8.  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate gLV sweep data for EP6 misspecification experiment.'
    )
    parser.add_argument('--a_hidden', type=float, default=None,
                        help='Single a_hidden value. Omit to run the full sweep.')
    parser.add_argument('--observe_all', action='store_true',
                        help='Observe all 5 species (default: only P1, P2, H3).')
    parser.add_argument('--seed',     type=int,   default=0)
    parser.add_argument('--dt',       type=float, default=0.5)
    parser.add_argument('--t_end',    type=float, default=100.0)
    parser.add_argument('--save_dir', type=str,   default='./data')
    args = parser.parse_args()

    observed_species = list(range(N_SPECIES)) if args.observe_all else None

    sweep = [args.a_hidden] if args.a_hidden is not None else A_HIDDEN_SWEEP

    for a_h in sweep:
        generate_experiment(
            a_hidden=a_h,
            seed=args.seed,
            dt=args.dt,
            t_end=args.t_end,
            save_dir=args.save_dir,
            observed_species=observed_species,
        )

    print("\nDone. Load a dataset with:")
    print("    import numpy as np")
    print("    truth = np.load('data/glv_ahidden0p00_truth.npz')")
    print("    obs   = np.load('data/glv_ahidden0p00_obs.npz')")
    print("    X, Y  = truth['X'], obs['Y']")


if __name__ == '__main__':
    main()
