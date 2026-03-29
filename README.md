# Ensemble Kalman Filter (EnKF) for Physical Parameter Estimation (PPE)

This repository implements the **Ensemble Kalman Filter (EnKF)** for time-series forecasting and physical parameter estimation. It supports both **state-augmented EnKF** and **auto-differentiable EnKF (AD-EnKF)**, with experiments on Lorenz 63, Lorenz 96, and generalized Lotka–Volterra (gLV) systems.

---

## Setup

### Using uv

```bash
pip install uv
uv sync
```

### Python path for AD-EnKF experiments

Run from the **repo root** (`EnKF_PPE_clone`). For scripts under `ADEnKF/` use:

```bash
PYTHONPATH=.:ADEnKF python ADEnKF/experiments/...
```

For gLV (which imports `torchEnKF` from `ADEnKF`):

```bash
PYTHONPATH=.:ADEnKF python ADEnKF/experiments/glv_param_est/glv_param_est_run.py
```

---

## 1. Problem: state and parameter estimation

We consider a discrete-time nonlinear dynamical system. The goal is to estimate the state \(x_k \in \mathbb{R}^n\) and physical parameters \(\theta \in \mathbb{R}^p\) at time \(k\), given noisy observations \(y_{1:k}\).

- **State evolution:** \(x_k = \Psi(x_{k-1}; \theta) + \eta_k\), with process noise \(\eta_k \sim \mathcal{N}(0, \Sigma)\).
- **Observations:** \(y_k = h(x_k) + \xi_k\), with observation noise \(\xi_k \sim \mathcal{N}(0, \Gamma)\).

The objective is the posterior \(p(x_k, \theta \mid y_{1:k})\); in practice we estimate its mean and covariance.

---

## 2. Ensemble Kalman Filter (EnKF)

The EnKF represents the state distribution with an ensemble of \(N\) members \(\mathbf{X}_k = [x_k^{(1)}, \ldots, x_k^{(N)}]\).

- **Forecast:** Each member is evolved: \(\hat{x}_k^{(i)} = \Psi(x_{k-1}^{(i)}, \theta) + \eta_k^{(i)}\).
- **Analysis:** When \(y_k\) is observed, each member is updated via the Kalman gain \(K_k\):
  \[
  x_k^{(i)} = \hat{x}_k^{(i)} + K_k \bigl( y_k + \xi_k^{(i)} - h(\hat{x}_k^{(i)}) \bigr).
  \]
  The gain \(K_k\) uses the sample covariance of the forecast ensemble; it can be computed in ensemble space without forming full covariance matrices.

---

## 3. Two EnKF variants in this repo

### State-augmented EnKF

Parameters \(\theta\) are part of the state: augmented state \(z_k = [x_k; \theta_k]\). The filter estimates the joint distribution of state and parameters; the analysis step updates both using cross-correlations. (See `src/enkf_ppe/Models/ENKF/`.)

### Auto-differentiable EnKF (AD-EnKF)

Parameters \(\theta\) are learned by gradient-based optimization. The EnKF (forecast + analysis) is implemented in PyTorch so that the observation log-likelihood \(\ell(\theta)\) is differentiable. We minimize \(-\ell(\theta)\) (e.g. with Adam). This yields gradients that include the effect of the full particle history (Term A + Term B). The **EM-EnKF** variant detaches the ensemble after each analysis step so that only “Term A” (direct dependence on \(\theta\)) is used—useful for comparison and ablation. (See `ADEnKF/` and `ADEnKF/methods/em_enkf.py`.)

---

## 4. AD-EnKF: structure and usage

The **ADEnKF** package provides:

- **`torchEnKF`**: differentiable EnKF core (`da_methods.EnKF`), ODE templates (Lorenz63, Lorenz96, gLV), and noise models.
- **`methods/em_enkf.py`**: EM-style EnKF (Term A only) for comparison with AD-EnKF.
- **Experiments** (Hydra-configured):
  - **L63 parameter estimation** — `ADEnKF/experiments/l63_param_est/`
  - **gLV parameter estimation** — `ADEnKF/experiments/glv_param_est/`
  - **Gradient decomposition (Term A vs A+B)** — `ADEnKF/experiments/gradient_decomposition/`
  - **UQ comparison plots** — `ADEnKF/experiments/uq_plots.py`

Configs live under each experiment’s `configs/` (e.g. `l63_param_est.yaml`, `glv_param_est.yaml`). Key options include `filter_mode: ad` or `filter_mode: em`, `N_ens`, `n_obs`, `epochs`, `chunk_length`, `lr`, and `process_std`.

### L63 parameter estimation

- **Parameter order:** \((\sigma, \rho, \beta)\), same as `torchEnKF.nn_templates.Lorenz63` and the rest of the repo.
- **Run (AD-EnKF):**
  ```bash
  PYTHONPATH=.:ADEnKF python ADEnKF/experiments/l63_param_est/l63_param_est_run.py
  ```
- **Run (EM-EnKF):** add `filter_mode=em`.
- **Overrides:** e.g. `seed=0 N_ens=100`, or `data_path=null` to generate data in-memory.
- Outputs (in Hydra run dir): training plot, `l63_param_est_results.pt`, estimated trajectory, and Laplace UQ files (`ad_enkf_laplace_*.npy` or `em_enkf_laplace_*.npy`).

### gLV parameter estimation

- Uses synthetic gLV datasets under `Data/gLV/` (e.g. `glv_ahidden0p00_truth.npz` / `_obs.npz`).
- **Run:** `filter_mode` is in `glv_param_est.yaml`; run dir is chosen automatically (e.g. `runs/AD_glv_param_est_torch` or `runs/EM_glv_param_est_torch`).
  ```bash
  PYTHONPATH=.:ADEnKF python ADEnKF/experiments/glv_param_est/glv_param_est_run.py
  ```
- Outputs: training plot, `glv_param_est_results.pt`, estimated trajectory, and Laplace UQ files.

### UQ comparison plots

After running L63 or gLV (and optionally both AD and EM), aggregate posterior samples and plot:

```bash
PYTHONPATH=.:ADEnKF python ADEnKF/experiments/uq_plots.py --system l63 \
  --results_dir path/to/AD_run_dir --results_dir_em path/to/EM_run_dir --out_dir path/to/out
```

For gLV use `--system glv`. Plots include marginal posteriors (L63), coverage calibration, and gLV parameter-error bar charts. See `ADEnKF/experiments/uq_plots_explanation.md` for details.

---

## 5. Data

### Lorenz 63

- **Equations:** \(\dot{x} = \sigma(y-x)\), \(\dot{y} = x(\rho-z)-y\), \(\dot{z} = xy - \beta z\). Standard chaotic parameters: \(\sigma=10\), \(\rho=28\), \(\beta=8/3\).
- **Data generation:** `Data/Lorentz63/generate_data.py` (e.g. `--steps 5000 --dt 0.01`). Saves `.pt` with `data` (trajectory) and `metadata`.
- **Visualization:** `uv run Data/Lorentz63/visualize_dataset.py "Data/Lorentz63/...pt"`.

### gLV

- Truth and observation data are produced by `Data/gLV/glv_data_generator.py` and stored as `.npz` under `Data/gLV/data/`.

---

## 6. Other READMEs (reference)

- **`ADEnKF/README.md`** — Short AD-EnKF intro and demo list (L96 demos).
- **`ADEnKF/experiments/l63_param_est/README.md`** — L63 config alignment with `enkf_butterfly`.
- **`Data/Lorentz63/README.md`** — Lorenz 63 system and data scripts in detail.
- **`src/enkf_ppe/Models/ENKF/README.md`** — State-augmented vs autodiff EnKF formulation.
- **`ADEnKF/experiments/uq_plots_explanation.md`** — How each UQ plot is built and how to interpret it.

---

## 7. Repo layout (summary)

```
EnKF_PPE_clone/
├── README.md                 # This file
├── ADEnKF/                   # Auto-differentiable EnKF
│   ├── torchEnKF/            # EnKF core, templates, noise
│   ├── methods/              # e.g. em_enkf.py
│   ├── examples/             # Data generation, demos
│   └── experiments/         # l63_param_est, glv_param_est, gradient_decomposition, uq_plots
├── Data/
│   ├── Lorentz63/            # L63 data generation and scripts
│   └── gLV/                  # gLV data and generator
├── Experiments/              # Other EnKF_PPE experiments (e.g. obs_dt_sweep)
└── src/enkf_ppe/             # State-augmented EnKF and related
```

Hydra run directories (e.g. `ADEnKF/experiments/l63_param_est/runs/`, `.../glv_param_est/runs/`) and figure outputs are intended to be local and are listed in `.gitignore`.
