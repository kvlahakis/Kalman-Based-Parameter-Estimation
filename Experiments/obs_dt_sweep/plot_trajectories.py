"""Plot trajectory comparisons from a saved obs_dt_sweep run.

For a given pair of obs_dt settings (default: best and worst RMSE), plots
the ensemble mean ± 1σ band against the truth for each state component.

Usage
-----
  # best vs worst (default):
  python plot_trajectories.py

  # specific obs_dt values:
  python plot_trajectories.py --obs_dts 0.01 0.50

  # specific run:
  python plot_trajectories.py --run runs/2026-02-22/18-30-00

  # save figure to run folder:
  python plot_trajectories.py --save
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from paths import DATA_DIR


# ── helpers ───────────────────────────────────────────────────────────────────

def load_obs_pred(run_dir: Path, obs_dt: float) -> dict:
    """Load a single obs_pred_results file by obs_dt value."""
    fname = run_dir / "obs_pred_results" / f"obs_dt_{obs_dt:.4f}.pt"
    if not fname.exists():
        available = sorted((run_dir / "obs_pred_results").glob("obs_dt_*.pt"))
        raise FileNotFoundError(
            f"No result file for obs_dt={obs_dt:.4f}.\n"
            f"Available: {[f.name for f in available]}"
        )
    return torch.load(fname, weights_only=True)

def load_param_pred(run_dir: Path, obs_dt: float) -> dict:
    """Load a single obs_pred_results file by obs_dt value."""
    fname = run_dir / "param_pred_results" / f"obs_dt_{obs_dt:.4f}.pt"
    if not fname.exists():
        available = sorted((run_dir / "param_pred_results").glob("obs_dt_*.pt"))
        raise FileNotFoundError(
            f"No result file for obs_dt={obs_dt:.4f}.\n"
            f"Available: {[f.name for f in available]}"
        )
    return torch.load(fname, weights_only=True)


def load_truth(truth_file: str) -> [torch.Tensor, torch.Tensor]:
    payload = torch.load(truth_file, weights_only=True)
    return payload["data"], payload["metadata"]["parameters"]   # (T_full, n_state), (n_param,)

# ── plotting ──────────────────────────────────────────────────────────────────

def plot_trajectory(
    run_dir:  Path,
    obs_dt:   float,
) -> None:
    """
    Plot state and parameter trajectories for a given obs_dt.

    Args:
        run_dir:  path to the run folder
        obs_dt:  list of obs_dt values to plot (one column each)
        T_show:   max observation steps to show per panel
        save:     write PNG to run folder instead of displaying
    """
    config = torch.load(run_dir / "run_config.pt", weights_only=True)
    model_target = config.get("model", "unknown")
    model_name   = model_target.split(".")[-1] if "." in model_target else model_target

    n_cols = 2
    colors = plt.cm.tab10.colors

    state_payload  = load_obs_pred(run_dir, obs_dt)
    state_meta     = state_payload["metadata"]
    state_hist     = state_payload["data"]              # (T_obs, N_ens, n_state)
    state_mean     = state_hist.mean(dim=1)             # (T_obs, n_state)
    state_std      = state_hist.std(dim=1)              # (T_obs, n_state)
    param_payload  = load_param_pred(run_dir, obs_dt)
    param_hist     = param_payload["data"]              # (T_obs, N_ens, n_param)
    param_mean     = param_hist.mean(dim=1)             # (T_obs, n_param)
    param_std      = param_hist.std(dim=1)              # (T_obs, n_param)
    truth_hist, true_params = load_truth(state_meta["truth_file"])
    T_use = state_hist.shape[0]

    n_comp = state_hist.shape[-1]

    fig, axes = plt.subplots(
        n_comp, n_cols,
        figsize=(6 * n_cols, 3 * n_comp),
        sharex="col",
        squeeze=False,
    )
    fig.suptitle(
        f"{model_name} — Trajectory Comparison\n"
        f"obs_dt={obs_dt:.4f} — n_fc={state_meta['n_forecasts']} — RMSE={state_meta['rmse_total']:.4f}\n"
        "(shaded = ensemble ±1σ  |  solid = truth  |  dashed = ensemble mean)",
        fontsize=11,
    )

    for row, state in enumerate(["x", "y", "z"]):
        state_ax    = axes[row, 0]
        color = colors[row % len(colors)]
        t_ax = np.arange(T_use) * state_meta["dt"]

        state_ax.fill_between(
            t_ax,
            (state_mean[:T_use, row] - state_std[:T_use, row]).numpy(),
            (state_mean[:T_use, row] + state_std[:T_use, row]).numpy(),
            color=color, alpha=0.25, label="Ens ±1σ",
        )
        state_ax.plot(t_ax, truth_hist[:T_use, row].numpy(),
                "k-", lw=1.5, label="Truth")
        state_ax.plot(t_ax, state_mean[:T_use, row].numpy(),
                color=color, lw=1.5, ls="--", label="Ens mean")
        state_ax.set_ylabel(f"{state}", fontsize=11)
    
        if row == 0:
            state_ax.set_title(
                "State Trajectory",
                fontsize=10,
            )
        if row == n_comp - 1:
            state_ax.set_xlabel("Time", fontsize=10)
        
    for row, (param_name, true_value) in enumerate(true_params.items()):
        param_ax    = axes[row, 1]
        color = colors[row % len(colors)]
        t_ax = np.arange(T_use) * state_meta["dt"]

        param_ax.fill_between(
            t_ax,
            (param_mean[:T_use, row] - param_std[:T_use, row]).numpy(),
            (param_mean[:T_use, row] + param_std[:T_use, row]).numpy(),
            color=color, alpha=0.25, label="Ens ±1σ",
        )
        param_ax.axhline(true_value, color="k", lw=1.5, ls="--", label="Truth")
        param_ax.plot(t_ax, param_mean[:T_use, row].numpy(),
                color=color, lw=1.5, ls="--", label="Ens mean")

        
        param_ax.set_ylabel(f"{param_name}", fontsize=11)
        state_ax.grid(alpha=0.3)
        param_ax.grid(alpha=0.3)

        
        if row == 0:
            param_ax.set_title(
                "Parameter Trajectory\n",
                fontsize=10,
            )
        if row == n_comp - 1:
            param_ax.set_xlabel("Time", fontsize=10)
        
        fig.tight_layout()
        
        label = f"obs_dt_{obs_dt:.4f}"
        out   = run_dir / f"trajectories_{label}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

# ── entry point ───────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Plot trajectory comparisons from a saved run.",
    )
    p.add_argument(
        "--path_to_run", type=str, default="butterfly/state_aug_enkf_with_offset",
        help="Path to a run folder from runs folder.",
    )
    p.add_argument(
        "--obs_dt", type=float, default=0.2000,
        help=(
            "obs_dt values to plot (default: best and worst RMSE). "
            "E.g. --obs_dts 0.01 0.10 0.50"
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args     = _parse_args()
    run_dir = Path(__file__).parent / "runs" / args.path_to_run

    print(f"Loading run: {run_dir}")

    plot_trajectory(run_dir, args.obs_dt)
