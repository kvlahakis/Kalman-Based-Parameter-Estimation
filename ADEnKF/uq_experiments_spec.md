# UQ Experiments Spec — AD-EnKF Parameter Estimation
## ACM 154 Final Project

---

## Context

The repo lives at `ADEnKF/`. Key existing files:
- `ADEnKF/experiments/glv_param_est/glv_param_est_run.py` — main training loop (AD-EnKF on gLV)
- `ADEnKF/experiments/glv_param_est/glv_param_est.yaml` — hydra config
- `ADEnKF/experiments/glv_param_est/aggregate_sweep.py` — results aggregation
- Lorenz-63 experiment files follow the same pattern (look in `ADEnKF/experiments/l63_*` or similar)
- The EnKF/AD-EnKF filter logic is in `ADEnKF/` (torchEnKF-style, PyTorch-based, autodiff through full filter graph)
- State-augmented EnKF appends θ to the state vector and runs standard EnKF on the extended system
- All models use `torchdiffeq` ODE solvers with adjoint backprop

**Goal:** Add UQ (posterior over θ, not just MAP) to two existing experiments:
1. **Lorenz-63** (3 params: σ, ρ, β — toy/visualization)
2. **gLV** (20 params: 5 growth rates + 15 interaction coefficients — high-dim benchmark)

---

## Experiment 1 — State-Augmented EnKF Posterior Samples

**What:** After the state-aug EnKF converges, the θ-slice of the ensemble IS an approximate posterior sample set. We just need to collect it.

**Where to add code:** In the Lorenz-63 state-aug EnKF script (wherever the augmented EnKF loop runs).

```python
# AFTER the EnKF loop finishes (post burn-in)
# Assume:
#   ensemble_history: list of (N_ens, d+p) tensors, one per assimilation step
#   d: state dimension (3 for L63, 5 for gLV)
#   p: param dimension (3 for L63, 20 for gLV)

import torch
import numpy as np

BURNIN = 200  # discard first 200 steps as transient

# Stack all post-burnin ensemble snapshots: shape (T_post * N_ens, p)
theta_samples = torch.stack(ensemble_history[BURNIN:])[:, :, d:].reshape(-1, p)
theta_samples_np = theta_samples.detach().cpu().numpy()

# Save for plotting
np.save("enkf_aug_theta_posterior.npy", theta_samples_np)
```

**Output file:** `enkf_aug_theta_posterior.npy` — shape `(N_samples, p)`

---

## Experiment 2 — AD-EnKF Laplace Approximation

**What:** After AD-EnKF converges to θ*, compute the Hessian of `ℓ_EnKF(θ)` at θ*. The Laplace posterior is `N(θ*, −H⁻¹)`.

**Where to add code:** At the end of `glv_param_est_run.py` (and the L63 equivalent), after the optimizer loop.

```python
import torch
from torch.autograd.functional import hessian

# theta_star: the converged parameter tensor from the optimizer
# Must be a flat 1D tensor of shape (p,) with requires_grad=False
theta_star = theta.detach().clone()  # adjust to your variable name

def neg_log_likelihood(theta_flat):
    """Re-run the EnKF forward pass with this theta and return -ℓ_EnKF."""
    # You need to reshape theta_flat back into whatever structure your model expects
    # e.g. alpha, beta = theta_flat[:n_alpha], theta_flat[n_alpha:]
    # then call your existing enkf_log_likelihood function
    return -enkf_log_likelihood(theta_flat, Y_obs)  # adjust to your function name

# Compute Hessian — shape (p, p)
H = hessian(neg_log_likelihood, theta_star)  # this is Hessian of -LL, so positive definite at max

# Posterior covariance: Sigma = H^{-1}  (H here is Hessian of neg-LL = -Hessian of LL)
nugget = 1e-5 * torch.eye(p)
Sigma_post = torch.linalg.inv(H + nugget)

# Posterior std per parameter
std_post = torch.sqrt(torch.diag(Sigma_post).clamp(min=0)).detach().cpu().numpy()

# Draw posterior samples
L_chol = torch.linalg.cholesky(Sigma_post + nugget)
z = torch.randn(p, 1000)
theta_samples = (theta_star.unsqueeze(1) + L_chol @ z).T  # (1000, p)
theta_samples_np = theta_samples.detach().cpu().numpy()

# Save
import numpy as np
np.save("ad_enkf_laplace_theta_posterior.npy", theta_samples_np)
np.save("ad_enkf_laplace_std.npy", std_post)
```

**Notes:**
- If `H` has non-positive diagonal entries, the Hessian is not at a proper maximum — print a warning and skip sampling. This is a valid negative result to report.
- For gLV with p=20, the Hessian is a 20×20 tensor — cheap to compute.
- For L63 with p=3, trivially cheap.
- The Hessian call reruns the forward pass with autograd twice — it can be slow for large N_ens/T. If too slow, reduce N_ens to 20 just for the Hessian call.

**Output files:** `ad_enkf_laplace_theta_posterior.npy` — shape `(1000, p)`, `ad_enkf_laplace_std.npy` — shape `(p,)`

---

## Experiment 3 — Neural ODE Laplace Baseline

**What:** Same Laplace trick but on the Neural ODE trajectory loss instead of ℓ_EnKF.

**Where to add:** At the end of your existing Neural ODE training script.

```python
from torch.autograd.functional import hessian

theta_star_node = neural_ode_theta.detach().clone()  # converged NeuralODE params

def neural_ode_neg_loss(theta_flat):
    """4DVar-style trajectory loss."""
    v_traj = integrate_ode(theta_flat, v0, t_span)  # your existing ODE solver call
    return ((H_obs @ v_traj - Y_obs) ** 2).mean()   # adjust H_obs, Y_obs to your names

H_node = hessian(neural_ode_neg_loss, theta_star_node)
Sigma_node = torch.linalg.inv(H_node + 1e-5 * torch.eye(p))
std_node = torch.sqrt(torch.diag(Sigma_node).clamp(min=0)).detach().cpu().numpy()

theta_samples_node = (theta_star_node.unsqueeze(1) +
                      torch.linalg.cholesky(Sigma_node) @ torch.randn(p, 1000)).T
np.save("neural_ode_laplace_theta_posterior.npy", theta_samples_node.detach().cpu().numpy())
np.save("neural_ode_laplace_std.npy", std_node)
```

**Output files:** `neural_ode_laplace_theta_posterior.npy` — shape `(1000, p)`, `neural_ode_laplace_std.npy` — shape `(p,)`

---

## Plotting Script

Create `ADEnKF/experiments/uq_plots.py`. Assumes all `.npy` files are in the same directory (or pass paths as args).

```python
"""
UQ comparison plots for L63 and gLV.
Usage:
    python uq_plots.py --system l63
    python uq_plots.py --system glv
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--system", choices=["l63", "glv"], required=True)
parser.add_argument("--out_dir", default=".")
args = parser.parse_args()

if args.system == "l63":
    param_names = [r"$\sigma$", r"$\rho$", r"$\beta$"]
    theta_true = np.array([10.0, 28.0, 8/3])
else:
    # gLV: 5 growth rates + 15 interaction coeffs (adjust to your ordering)
    r_names = [f"$r_{i}$" for i in range(1, 6)]
    a_names = [f"$a_{{13}}$", "$a_{14}$", "$a_{21}$", "$a_{23}$",
               "$a_{31}$", "$a_{33}$", "$a_{35}$", "$a_{42}$",
               "$a_{44}$", "$a_{45}$", "$a_{51}$", "$a_{52}$",
               "$a_{53}$", "$a_{54}$", "$a_{55}$"]
    param_names = r_names + a_names
    # Match your theta_true vector exactly
    theta_true = np.array([1.3, 1.1, -0.05, -0.3, -0.2,
                           -0.80, 0.0, 0.0, -0.70, 0.60,
                           0.0, -0.25, 0.0, 0.45, 0.0,
                           -0.2, 0.0, 0.15, 0.10, -0.10])

p = len(theta_true)

# Load posterior samples (handle missing files gracefully)
def load(fname):
    try:
        return np.load(fname)
    except FileNotFoundError:
        print(f"Warning: {fname} not found, skipping.")
        return None

enkf_samples   = load("enkf_aug_theta_posterior.npy")    # (N, p)
ad_samples     = load("ad_enkf_laplace_theta_posterior.npy")  # (1000, p)
node_samples   = load("neural_ode_laplace_theta_posterior.npy")  # (1000, p)

methods = []
if enkf_samples  is not None: methods.append(("EnKF-aug",    enkf_samples,  "C0"))
if ad_samples    is not None: methods.append(("AD-EnKF (Laplace)", ad_samples, "C1"))
if node_samples  is not None: methods.append(("NeuralODE (Laplace)", node_samples, "C2"))

# ----------------------------------------------------------------
# FIGURE 1: Marginal posteriors (L63 only — 3 params)
# ----------------------------------------------------------------
if args.system == "l63":
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, (ax, name, truth) in enumerate(zip(axes, param_names, theta_true)):
        for (label, samps, color) in methods:
            col = samps[:, i]
            ax.hist(col, bins=40, density=True, alpha=0.4, color=color, label=label)
            # KDE overlay
            kde = stats.gaussian_kde(col)
            xs = np.linspace(col.min(), col.max(), 200)
            ax.plot(xs, kde(xs), color=color, lw=2)
        ax.axvline(truth, color="red", lw=2, linestyle="--", label=r"$\theta^*$")
        ax.set_title(name, fontsize=14)
        ax.set_xlabel("Parameter value")
        if i == 0:
            ax.set_ylabel("Density")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=10)
    fig.suptitle("Lorenz-63: Marginal Posteriors", fontsize=15)
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/l63_marginal_posteriors.pdf", bbox_inches="tight")
    print("Saved l63_marginal_posteriors.pdf")

# ----------------------------------------------------------------
# FIGURE 2: Coverage plot (works for both systems)
# ----------------------------------------------------------------
alpha_levels = np.linspace(0.05, 0.95, 19)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")

for (label, samps, color) in methods:
    coverages = []
    mean = samps.mean(0)
    std  = samps.std(0) + 1e-12
    z = np.abs((theta_true - mean) / std)
    for alpha in alpha_levels:
        z_crit = stats.norm.ppf((1 + alpha) / 2)
        coverages.append((z < z_crit).mean())
    ax.plot(alpha_levels, coverages, "o-", color=color, label=label, markersize=5)

ax.set_xlabel("Nominal coverage", fontsize=13)
ax.set_ylabel("Empirical coverage", fontsize=13)
ax.set_title(f"{'L63' if args.system=='l63' else 'gLV'}: Calibration (Coverage Plot)", fontsize=14)
ax.legend(fontsize=11)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(f"{args.out_dir}/{args.system}_coverage.pdf", bbox_inches="tight")
print(f"Saved {args.system}_coverage.pdf")

# ----------------------------------------------------------------
# FIGURE 3 (gLV only): Parameter recovery bar chart with UQ error bars
# ----------------------------------------------------------------
if args.system == "glv":
    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(p)
    width = 0.25
    offsets = [-width, 0, width]

    for offset, (label, samps, color) in zip(offsets, methods):
        means = samps.mean(0)
        stds  = samps.std(0)
        err   = np.abs(means - theta_true)
        ax.bar(x + offset, err, width, yerr=2*stds, capsize=3,
               label=label, color=color, alpha=0.75, error_kw={"elinewidth": 1.5})

    ax.axhline(0.05, color="red", linestyle="--", lw=1.5, label="0.05 tolerance")
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(r"$|\hat{\theta} - \theta^*|$ ± 2σ_post", fontsize=12)
    ax.set_title("gLV: Parameter Error with Posterior Uncertainty", fontsize=14)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/glv_uq_bar_chart.pdf", bbox_inches="tight")
    print("Saved glv_uq_bar_chart.pdf")

plt.close("all")
print("Done.")
```

---

## Summary of Files to Create / Modify

| File | Action | What to do |
|---|---|---|
| `experiments/l63_*/l63_enkf_aug_run.py` | **Modify** | Save `ensemble_history[-200:, :, d:]` to `enkf_aug_theta_posterior.npy` |
| `experiments/glv_param_est/glv_param_est_run.py` | **Modify** | Add Hessian block at end → save `ad_enkf_laplace_*.npy` |
| `experiments/l63_*/l63_ad_enkf_run.py` | **Modify** | Same Hessian block → save L63 version |
| Your Neural ODE training script | **Modify** | Add Hessian block at end → save `neural_ode_laplace_*.npy` |
| `experiments/uq_plots.py` | **Create** | Full plotting script above |

---

## Notes for the Coding Agent

1. **Variable name mapping** — the snippets use generic names (`theta`, `enkf_log_likelihood`, `Y_obs`). You need to replace these with whatever they are actually called in each script. Look for the optimizer step line (e.g. `optimizer.step()`) to find the live `theta` tensor, and for the EnKF log-likelihood accumulation to find the function/value to differentiate.

2. **Hessian memory** — `torch.autograd.functional.hessian` builds a full p×p matrix. For p=20 (gLV) this is trivial. For p=3 (L63) also trivial. No special handling needed.

3. **Detach before Hessian** — make sure `theta_star` has `requires_grad=False` when passed in; enable grad inside `neg_log_likelihood` by creating `theta_flat = theta_flat.requires_grad_(True)` at the top of that function if needed.

4. **ensemble_history** — if the existing code doesn't accumulate ensemble snapshots, add `ensemble_history.append(ensemble.detach().clone())` inside the assimilation loop.

5. **Run order** — run the training scripts first (they already work), then run `uq_plots.py` pointing at the saved `.npy` files.
