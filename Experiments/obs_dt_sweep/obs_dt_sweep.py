"""Experiment: Effect of Observation Frequency on Forecasting Model Performance
===========================================================================

Any time-series forecasting model that implements BaseModel can be evaluated
here. The key independent variable is the observation time step:

    obs_dt = n_forecasts * model.time_step

The model is fully specified by the Hydra config and instantiated via
hydra.utils.instantiate — no model-specific code lives in this script.

To swap in a different model, run:

    python obs_dt_sweep.py --config-name <name>

where <name> matches a file in configs/.  Override individual values:

    python obs_dt_sweep.py model.time_step=0.005 N_ens=100
    python obs_dt_sweep.py sweep.n_forecasts=[1,5,10,50]

Outputs — saved to runs/<date>/<time>/ (configured via hydra.run.dir):
  run_config.pt
  obs_pred_results/obs_dt_<value>.pt   — state analysis ensemble per obs_dt
  param_pred_results/obs_dt_<value>.pt — parameter ensemble per obs_dt

Use plot_rmse.py and plot_trajectories.py to visualise saved results.
"""

from pathlib import Path

import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from enkf_ppe.Models import BaseModel
from enkf_ppe.Utils.initialisations import Initialisation
from paths import ROOT, DATA_DIR


@hydra.main(config_path="configs", config_name="enkf_butterly_init_offset", version_base=None)
def run_experiment(cfg: DictConfig) -> None:

    # ── load truth ──
    data_file = DATA_DIR / cfg.data_path
    payload   = torch.load(data_file, weights_only=True)
    truth     = payload["data"]       # (T_full, n_state)
    meta      = payload["metadata"]

    T_use    = min(truth.shape[0], cfg.experiment.T_use)
    N_ens    = cfg.N_ens
    n_state  = truth.shape[-1]
    n_param  = cfg.model.param_noise_cov.dim
    dt_model = cfg.model.time_step
    obs_std  = cfg.model.obs_noise_cov.std
    proc_std = cfg.model.process_noise_cov.std  

    theta_true = torch.tensor(list(meta["parameters"].values()))
    # Use config initial_theta if set (e.g. [12, 20, 5] for comparable AD-EnKF runs), else true params
    theta_center = torch.tensor(
        cfg.get("initial_theta", theta_true.tolist()),
        dtype=theta_true.dtype,
    ).unsqueeze(0)  # (1, p)

    initial_state: Initialisation = instantiate(cfg.state_init)
    initial_param: Initialisation = instantiate(cfg.param_init)

    # ── instantiate model entirely from config ──
    model: BaseModel = instantiate(cfg.model)

    model_id = OmegaConf.select(cfg, "model._target_") or "unknown"

    # ── hydra-managed run folder ──
    run_dir   = Path(HydraConfig.get().runtime.output_dir)
    obs_dir   = run_dir / "obs_pred_results"
    param_dir = run_dir / "param_pred_results"
    obs_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(parents=True, exist_ok=True)

    # ── obs_dt sweep ──
    n_forecasts_list = list(cfg.sweep.n_forecasts)
    obs_dt_list = [n_fc * dt_model for n_fc in n_forecasts_list]

    for n_fc in n_forecasts_list:
        torch.manual_seed(cfg.seed)
        obs_dt    = n_fc * dt_model
        fname_key = f"obs_dt_{obs_dt:.4f}"

        truth_sub = truth[:T_use:n_fc]                       # (T_obs, n_state)
        T_obs     = truth_sub.shape[0]

        X0 = initial_state(torch.tensor(list(meta["initial_state"])), N_ens)
        theta0 = initial_param(theta_center, N_ens)

        print(f"obs_dt={obs_dt:.3f}  (n_forecasts={n_fc:2d},  T_obs={T_obs}) ... ",
              end="", flush=True)

        X_hist, theta_hist = model.run(X0, theta0, truth_sub, dt=obs_dt)

        # RMSE — skip spin-up
        spin      = max(cfg.experiment.spinup_min,
                        int(T_obs * cfg.experiment.spinup_frac))
        mean_hist = X_hist[spin:].mean(dim=1)                # (T_valid, n_state)
        err       = mean_hist - truth[spin:T_use]
        rmse_total = err.pow(2).mean().sqrt().item()
        rmse_comp  = err.pow(2).mean(dim=0).sqrt().tolist()

        print(f"RMSE = {rmse_total:.4f}")

        # ── save state predictions ──
        torch.save(
            {
                "data": X_hist,
                "metadata": {
                    **meta,
                    "obs_dt":          obs_dt,
                    "n_forecasts":     n_fc,
                    "N_ens":           N_ens,
                    "T_obs":           T_obs,
                    "n_state":         n_state,
                    "dt_model":        dt_model,
                    "spin_steps":      spin,
                    "rmse_total":      rmse_total,
                    "rmse_components": rmse_comp,
                    "truth_file":      str(data_file),
                    "model":           model_id,
                }
            },
            obs_dir / f"{fname_key}.pt",
        )

        # ── save parameter predictions ──
        torch.save(
            {
                "data": theta_hist,
                "metadata": {
                    "obs_dt":      obs_dt,
                    "n_forecasts": n_fc,
                    "model":       model_id,
                    "param_dim":   n_param,
                    "theta_true":  list(meta["parameters"].values()),
                },
            },
            param_dir / f"{fname_key}.pt",
        )

    # ── save run-level config ──
    torch.save(
        {
            "model":            model_id,
            "data":             Path(cfg.data_path).stem,
            "n_forecasts_list": n_forecasts_list,
            "obs_dt_list":      obs_dt_list,
            "N_ens":            N_ens,
            "obs_std":          obs_std,
            "proc_std":         proc_std,
            "n_state":          n_state,
            "dt_model":         dt_model,
            "parameters":       meta["parameters"],
            "T_use":            T_use,
            "truth_file":       str(data_file),
            "hydra_cfg":        OmegaConf.to_container(cfg, resolve=True),
        },
        run_dir / "run_config.pt",
    )

    # ── summary table ──
    comp_header = "  ".join(f"{'RMSE_'+str(i):>8}" for i in range(n_state))
    print()
    print(f"{'obs_dt':>8}  {'n_fc':>5}  {'RMSE':>8}  {comp_header}")
    print("-" * (30 + 10 * n_state))
    for n_fc in n_forecasts_list:
        r = torch.load(
            obs_dir / f"obs_dt_{n_fc * dt_model:.4f}.pt",
            weights_only=True,
        )["metadata"]
        comp_str = "  ".join(f"{v:8.4f}" for v in r["rmse_components"])
        print(f"{r['obs_dt']:8.3f}  {n_fc:5d}  {r['rmse_total']:8.4f}  {comp_str}")

    print(f"\nRun saved to: {run_dir}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_experiment()
