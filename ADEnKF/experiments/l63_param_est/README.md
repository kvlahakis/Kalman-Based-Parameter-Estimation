# L63 parameter estimation (differentiable EnKF)

This experiment runs **torchEnKF** (auto-differentiable EnKF) for Lorenz 63 parameter estimation, with Hydra config aligned to **EnKF_PPE**’s standard EnKF so results can be compared.

## L63 parameter order (reconciled)

Both codebases use the same parameter order and dynamics:

- **Order:** `(sigma, rho, beta)` — same as `enkf_ppe.Dynamics.Lorentz63` and `torchEnKF.nn_templates.Lorenz63`.
- **Equations:** `dx = sigma*(y-x)`, `dy = x*(rho-z)-y`, `dz = x*y - beta*z`.

So the same true parameters (e.g. sigma=10, rho=28, beta=8/3) and initialisation can be used in both the standard EnKF (`Experiments/obs_dt_sweep` with `enkf_butterfly.yaml`) and this differentiable EnKF run.

## Running

From the **repo root** (EnKF_PPE_clone):

```bash
PYTHONPATH=.:torchEnKF python torchEnKF/experiments/l63_param_est/l63_param_est_run.py
```

- Uses `data_path` from config (same file as `enkf_butterfly`: `Lorentz63/sigma10.0000_rho28.0000_beta2.6667_dt0.0100.pt`) so both methods see the same truth and observation schedule.
- Override with Hydra, e.g.:
  - `python ... l63_param_est_run.py seed=0 N_ens=100`
  - `python ... l63_param_est_run.py data_path=null` — generate data in-memory instead of loading the file.

Outputs go to `runs/l63_param_est_torch/<date>/<time>/` (Hydra run dir): plot and `l63_param_est_results.pt`.

## Config alignment with enkf_butterfly

| Setting   | enkf_butterfly.yaml | l63_param_est.yaml |
|----------|----------------------|--------------------|
| seed     | 42                   | 42                 |
| N_ens    | 50                   | 50                 |
| data_path| Lorentz63/...pt      | same               |
| dt       | 0.01                 | 0.01               |
| obs_std  | 1.0                  | 1.0                |
| true (σ,ρ,β) | 10, 28, 8/3    | 10, 28, 8/3        |

Use the same data file and these settings to compare standard EnKF (state-augmented) and differentiable EnKF on the same problem.
