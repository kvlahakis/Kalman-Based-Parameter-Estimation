"""
L63 parameter estimation with differentiable EnKF, driven by Hydra config.

Aligned with Experiments/obs_dt_sweep (enkf_butterfly.yaml) so the same
data_path, seed, N_ens, true params, and obs settings can be used to compare
the auto-differentiable EnKF (torchEnKF) with the standard state-augmented EnKF.

Run from repo root (EnKF_PPE_clone) with:

  PYTHONPATH=.:ADEnKF python ADEnKF/experiments/l63_param_est/l63_param_est_run.py

Override config, e.g.:

  python ADEnKF/experiments/l63_param_est/l63_param_est_run.py seed=0 N_ens=100
  python ADEnKF/experiments/l63_param_est/l63_param_est_run.py data_path=null
"""

from pathlib import Path
import sys

# Repo root = EnKF_PPE_clone; ensure we can import `paths`, `examples`, and `torchEnKF`
_script_dir = Path(__file__).resolve().parent          # ADEnKF/experiments/l63_param_est
_ad_enkf_dir = _script_dir.parent.parent              # ADEnKF
_repo_root = _ad_enkf_dir.parent                      # repo root
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_ad_enkf_dir))

from examples import generate_data
from torchEnKF import da_methods, nn_templates, noise
from paths import DATA_DIR
from torchdiffeq import odeint
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
import torch
import random

from methods.em_enkf import EnKF_EM
from fractions import Fraction

def _setup_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_data_from_file(data_path: str, n_forecasts: int, n_obs: int, obs_std: float, device):
    """Load truth from EnKF_PPE data file; subsample at obs times and add observation noise."""
    full_path = DATA_DIR / data_path
    payload = torch.load(full_path, weights_only=True)
    data = payload["data"]  # (T, 3)
    meta = payload["metadata"]
    # Subsample: one observation every n_forecasts steps
    obs_dt_steps = n_forecasts
    truth = data[::obs_dt_steps][:n_obs]  # (n_obs, 3)
    if truth.shape[0] < n_obs:
        raise ValueError(
            f"Data has {truth.shape[0]} obs after subsampling; need at least n_obs={
                n_obs}. "
            "Increase dataset length or reduce n_obs / n_forecasts."
        )
    truth = truth.to(device)
    # Observations: truth + noise (match enkf_butterfly obs_noise_cov std)
    torch.manual_seed(42)  # reproducible obs noise
    y_obs = truth + obs_std * torch.randn_like(truth)
    # Initial state from metadata (for consistency; we use single trajectory from file)
    initial_state = torch.tensor(
        meta["initial_state"], dtype=torch.float32, device=device)
    # True params (sigma, rho, beta) from metadata
    params = meta["parameters"]
    true_coeff = torch.tensor(
        [params["sigma"], params["rho"], params["beta"]],
        device=device,
    )
    dt = meta["dt"]
    obs_dt = dt * n_forecasts
    t_obs = obs_dt * torch.arange(1, n_obs + 1, device=device)
    return truth, y_obs, true_coeff, initial_state, t_obs, dt


def _generate_data_in_memory(cfg: DictConfig, device):
    """Generate truth and observations in-memory (no data file)."""
    x_dim = cfg.x_dim
    true_coeff = torch.tensor(
        [cfg.true_params.sigma, cfg.true_params.rho, cfg.true_params.beta],
        device=device,
    )
    true_ode_func = nn_templates.Lorenz63(true_coeff).to(device)
    train_size = 4
    init_cov = torch.diag(torch.tensor(cfg.init_cov_diag))
    with torch.no_grad():
        x0_warmup = torch.distributions.MultivariateNormal(
            torch.zeros(x_dim), covariance_matrix=init_cov
        ).sample().to(device)
        t_warmup = (40 * torch.arange(0.0, train_size + 1)).to(device)
        x0 = odeint(
            true_ode_func, x0_warmup, t_warmup,
            method="rk4", options=dict(step_size=cfg.ode_step_size),
        )[1:]
    n_obs = cfg.n_obs
    obs_dt = cfg.n_forecasts * cfg.dt
    t_obs = obs_dt * torch.arange(1, n_obs + 1, device=device)
    H = torch.eye(x_dim, device=device)
    true_obs_func = nn_templates.Linear(x_dim, x_dim, H=H).to(device)
    noise_R = noise.AddGaussian(x_dim, torch.tensor(
        cfg.obs_std), param_type="scalar").to(device)
    with torch.no_grad():
        x_truth, y_obs = generate_data.generate(
            true_ode_func, true_obs_func, t_obs, x0,
            None, noise_R, device=device,
            ode_method=cfg.ode_method, ode_options=dict(step_size=cfg.ode_step_size), tqdm=tqdm,
        )
    initial_state = x0[0]
    return x_truth, y_obs, true_coeff, initial_state, t_obs, cfg.dt, train_size


@hydra.main(config_path="configs", config_name="l63_param_est", version_base=None)
def run(cfg: DictConfig) -> None:
    device = _setup_device()
    print(f"device: {device}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    x_dim = cfg.x_dim
    N_ensem = cfg.N_ens
    n_obs = cfg.n_obs
    L = cfg.chunk_length
    t0 = 0.0

    if cfg.get("data_path") and str(cfg.data_path).lower() not in ("none", "null"):
        truth, y_obs, true_coeff, initial_state, t_obs, dt = _load_data_from_file(
            cfg.data_path, cfg.n_forecasts, n_obs, cfg.obs_std, device
        )
        train_size = 1
        # y_obs from file is (n_obs, 3); we need (n_obs, train_size, y_dim) for da_methods
        y_obs = y_obs.unsqueeze(1)
        truth = truth.unsqueeze(1)
    else:
        out = _generate_data_in_memory(cfg, device)
        x_truth, y_obs, true_coeff, initial_state, t_obs, dt, train_size = out
        truth = x_truth

    # true_ode_func = nn_templates.Lorenz63(true_coeff).to(device)
    H_true = torch.eye(x_dim, device=device)
    true_obs_func = nn_templates.Linear(x_dim, x_dim, H=H_true).to(device)
    noise_R_true = noise.AddGaussian(x_dim, torch.tensor(
        cfg.obs_std), param_type="scalar").to(device)

    # When using data file, perturb around file's initial state (match EnKF_PPE); else around zero
    if cfg.get("data_path") and str(cfg.data_path).lower() not in ("none", "null"):
        init_m = initial_state
    else:
        init_m = torch.zeros(x_dim, device=device)
    init_C_param = noise.AddGaussian(
        x_dim,
        torch.diag(torch.tensor(cfg.init_cov_diag, device=device)),
        "full",
    ).to(device)

    init_coeff = torch.tensor(
        [cfg.init_params.sigma, cfg.init_params.rho, cfg.init_params.beta],
        device=device,
    )
    learned_ode_func = nn_templates.Lorenz63(init_coeff).to(device)
    # Process noise Q: per-dimension std (separate from obs_std). Match EnKF process_noise_cov.std when comparing.
    q_scale = float(cfg.get("process_std", cfg.get("init_Q_diag_scale", 0.05)))
    init_Q = q_scale * torch.ones(x_dim, device=device)
    learned_model_Q = noise.AddGaussian(x_dim, init_Q, "diag").to(device)

    optimizer = torch.optim.Adam([
        {"params": learned_ode_func.parameters(), "lr": cfg.lr},
        {"params": learned_model_Q.parameters(), "lr": cfg.lr},
    ])
    epoch_start = cfg.get("lr_lambda_epoch_start", 30)
    expo = cfg.get("lr_lambda_exponent", -0.4)

    def lr_lambda(epoch):
        return (epoch - 9) ** expo if epoch >= epoch_start else 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[lr_lambda, lr_lambda],
    )

    monitor = []
    filter_mode = str(cfg.get("filter_mode", "ad")).lower()  # "ad" (AD-EnKF) or "em" (EM-EnKF)
    for epoch in tqdm(range(cfg.epochs), desc="Training", leave=False):
        train_log_likelihood = torch.zeros(train_size, device=device)
        t_start = t0
        X = init_C_param(init_m.expand(train_size, N_ensem, x_dim))

        for start in range(0, n_obs, L):
            optimizer.zero_grad()
            end = min(start + L, n_obs)
            if filter_mode == "ad":
                # AD-EnKF: full adjoint backprop through EnKF
                X, _, log_likelihood = da_methods.EnKF(
                    learned_ode_func,
                    true_obs_func,
                    t_obs[start:end],
                    y_obs[start:end],
                    N_ensem,
                    init_m,
                    init_C_param,
                    learned_model_Q,
                    noise_R_true,
                    device,
                    save_filter_step={},
                    t0=t_start,
                    init_X=X,
                    ode_method=cfg.ode_method,
                    ode_options=dict(step_size=cfg.ode_step_size),
                    adjoint=True,
                    adjoint_method=cfg.adjoint_method,
                    adjoint_options=dict(step_size=cfg.adjoint_step_size),
                    tqdm=None,
                )
            else:
                # EM-EnKF: particles detached after each analysis step (Term A only)
                X, _, log_likelihood = EnKF_EM(
                    learned_ode_func,
                    true_obs_func,
                    t_obs[start:end],
                    y_obs[start:end],
                    N_ensem,
                    init_m,
                    init_C_param,
                    learned_model_Q,
                    noise_R_true,
                    device,
                    init_X=X,
                    ode_method=cfg.ode_method,
                    ode_options=dict(step_size=cfg.ode_step_size),
                    t0=float(t_start),
                    compute_likelihood=True,
                    tqdm=None,
                )
            t_start = t_obs[end - 1]
            (-log_likelihood).mean().backward()
            train_log_likelihood += log_likelihood.detach().clone()
            optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            sigma, rho, beta = learned_ode_func.coeff.data.cpu()
            tqdm.write(
                f"Epoch {epoch} | LL: {
                    train_log_likelihood.mean().item():.2f} | "
                f"sigma={sigma:.3f} (true=10.000) | "
                f"rho={rho:.3f} (true=28.000) | "
                f"beta={beta:.3f} (true=2.667)"
            )
        with torch.no_grad():
            q_scale = torch.sqrt(torch.trace(learned_model_Q.full()) / x_dim)
            s, r, b = learned_ode_func.coeff.tolist()
            monitor.append(
                [s, r, b, q_scale.item(), train_log_likelihood.mean().item()])

    # Shape: (n_epochs, 5) with columns [sigma, rho, beta, q_scale, log_likelihood]
    monitor = np.asarray(monitor)
    true_vals = [to_float(cfg.true_params.sigma),
                 to_float(cfg.true_params.rho), 
                 to_float(cfg.true_params.beta)]
    param_names = ["sigma", "rho", "beta"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    for i, (name, true_val) in enumerate(zip(param_names, true_vals)):
        line, = ax.plot(monitor[:, i], label=f"{name} (learned)")
        ax.axhline(true_val, color=line.get_color(), linestyle="--",
                   alpha=0.5, label=f"{name} true={true_val:.2f}")
    ax.set_title("Parameter convergence")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    axes[1].plot(monitor[:, 3])
    axes[1].set_title("Model noise scale (q)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("q")
    axes[2].plot(monitor[:, 4])
    axes[2].set_title("Training log-likelihood")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Log-likelihood")
    plt.tight_layout()
    from hydra.core.hydra_config import HydraConfig
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    plot_path = Path(cfg.get("plot_path", "l63_parameter_estimation.png"))
    if not plot_path.is_absolute():
        plot_path = run_dir / plot_path
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_path}")

    # ------------------------------------------------------------------
    # Save results in a format compatible with visualize_dataset.py:
    #   - 'data': (N, 3) tensor visualised as a 3D trajectory, here the
    #             learned (sigma, rho, beta) over training epochs.
    #   - 'metadata': free-form dict used only for plot title.
    # Avoid storing NumPy arrays so that torch.load(..., weights_only=True)
    # works without unsafe NumPy pickling.
    # ------------------------------------------------------------------
    monitor_list = monitor.tolist()
    data = torch.tensor([[row[0], row[1], row[2]]
                        for row in monitor_list], dtype=torch.float32)

    results = {
        "data": data,
        "metadata": {
            "description": "L63 parameter estimates trajectory (sigma, rho, beta) over training epochs",
            "param_names": param_names,
            "true_vals": true_vals,
            "epochs": cfg.epochs,
        },
        "monitor": monitor_list,
        "param_names": param_names,
        "true_vals": true_vals,
        "hydra_cfg": OmegaConf.to_container(cfg, resolve=True),
    }

    results_path = run_dir / "l63_param_est_results.pt"
    torch.save(results, results_path)
    print(f"Results saved to {results_path}")

    # ------------------------------------------------------------------
    # Additionally, generate a Lorenz '63 trajectory using the FINAL
    # estimated parameters and save it in the same (N, 3) + metadata
    # format used by Data/Lorentz63/generate_data.py so it can be
    # visualised directly with visualize_dataset.py.
    # ------------------------------------------------------------------
    with torch.no_grad():
        sigma_est, rho_est, beta_est = learned_ode_func.coeff.detach().cpu().tolist()

        total_time = cfg.n_forecasts * cfg.n_obs * cfg.dt
        t_eval = torch.arange(0.0, total_time + cfg.dt, cfg.dt, device=device)
        traj = odeint(
            learned_ode_func,
            initial_state,
            t_eval,
            method=cfg.ode_method,
            options=dict(step_size=cfg.ode_step_size),
        )

    # traj: (T, 3) states [x, y, z]
    traj_cpu = traj.detach().cpu()
    est_dataset = {
        "data": traj_cpu,
        "metadata": {
            "sigma": sigma_est,
            "rho": rho_est,
            "beta": beta_est,
            "dt": cfg.dt,
            "initial_state": initial_state.detach().cpu().tolist(),
            "description": "Lorenz '63 trajectory generated with estimated parameters from l63_param_est_run",
        },
    }

    est_path = run_dir / "l63_estimated_traj.pt"
    torch.save(est_dataset, est_path)
    print(f"Estimated trajectory dataset saved to {est_path}")

    # ------------------------------------------------------------------
    # Laplace approximation for EnKF-based posterior over (sigma, rho, beta)
    # (files are named according to filter_mode: AD-EnKF vs EM-EnKF)
    # ------------------------------------------------------------------
    from torch.autograd.functional import hessian

    # Final parameter estimate (3,)
    theta_star = learned_ode_func.coeff.detach().clone()
    p_dim = theta_star.numel()
    laplace_prefix = "ad_enkf" if filter_mode == "ad" else "em_enkf"
    
    _hessian_seed = cfg.seed + 9999
    
    class _L63Direct(torch.nn.Module):
        """Lorenz-63 ODE that holds coeff as a plain tensor (not nn.Parameter)
        so that torch_hessian's tracked theta_flat stays in the graph."""
        def __init__(self, coeff_tensor: torch.Tensor):
            super().__init__()
            self._coeff = coeff_tensor  # NOT nn.Parameter — preserves graph

        def forward(self, t, u):
            sigma, rho, beta = self._coeff
            return torch.stack((
                sigma * (u[..., 1] - u[..., 0]),
                u[..., 0] * (rho - u[..., 2]) - u[..., 1],
                u[..., 0] * u[..., 1] - beta * u[..., 2],
            ), dim=-1)


    def enkf_neg_log_likelihood(theta_flat: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(_hessian_seed)
        theta_flat = theta_flat.to(device)

        ode_func_h = _L63Direct(theta_flat)   # <-- no nn.Parameter wrapping
        q_scale_h = float(cfg.get("process_std", cfg.get("init_Q_diag_scale", 0.05)))
        model_Q_h = noise.AddGaussian(
            x_dim, q_scale_h * torch.ones(x_dim, device=device), "diag"
        ).to(device)

        X_h = init_C_param(init_m.expand(train_size, N_ensem, x_dim))
        t_start_h = t0
        log_lik_h = torch.zeros(train_size, device=device)

        for start in range(0, n_obs, L):
            end = min(start + L, n_obs)
            X_h, _, ll = da_methods.EnKF(
                ode_func_h, true_obs_func,
                t_obs[start:end], y_obs[start:end],
                N_ensem, init_m, init_C_param, model_Q_h, noise_R_true, device,
                save_filter_step={}, t0=t_start_h, init_X=X_h,
                ode_method=cfg.ode_method,
                ode_options=dict(step_size=cfg.ode_step_size),
                adjoint=False,
                tqdm=None,
            )
            t_start_h = t_obs[end - 1]
            log_lik_h = log_lik_h + ll

        return -(log_lik_h.mean())

    try:
        H_raw = hessian(enkf_neg_log_likelihood, theta_star)  # (p, p)
        H_raw = H_raw.detach()

        # 1. Symmetrize — eliminates floating-point asymmetry from the two
        #    graph traversals used internally by torch_hessian.
        H_sym = (H_raw + H_raw.T) / 2.0

        # 2. Nearest positive-definite projection via eigenvalue clipping.
        #    We expect H to be PD at the MLE but the stochastic EnKF estimate
        #    will have some small negative eigenvalues from finite-ensemble noise.
        #    Clipping to a small floor (1e-4) is the standard Laplace fix and is
        #    equivalent to adding a weak prior that limits posterior variance
        #    in poorly-identified directions.
        eigvals, eigvecs = torch.linalg.eigh(H_sym)   # eigh guarantees real output
        min_eig = eigvals.min().item()
        max_eig = eigvals.max().item()
        print(f"Hessian eigenvalues — min: {min_eig:.3e}, max: {max_eig:.3e}")

        CLIP_FLOOR = 1e-4   # governs max posterior std in unidentified directions
        n_clipped = int((eigvals < CLIP_FLOOR).sum().item())
        if n_clipped > 0:
            print(
                f"Note: {n_clipped}/{p_dim} eigenvalue(s) clipped to {CLIP_FLOOR}. "
                "This is normal for a stochastic EnKF Hessian — not a failure."
            )
        eigvals_pd = eigvals.clamp(min=CLIP_FLOOR)
        H_pd = eigvecs @ torch.diag(eigvals_pd) @ eigvecs.T

        # 3. Posterior covariance and samples
        Sigma_post = torch.linalg.inv(H_pd)                    # (p, p)
        std_post = torch.sqrt(torch.diag(Sigma_post)).detach().cpu().numpy()

        L_chol = torch.linalg.cholesky(Sigma_post)
        z = torch.randn(p_dim, 1000, device=device)
        theta_samples = (theta_star.to(device).unsqueeze(1) + L_chol @ z).T  # (1000, p)
        theta_samples_np = theta_samples.detach().cpu().numpy()

        np.save(run_dir / f"{laplace_prefix}_laplace_theta_posterior.npy", theta_samples_np)
        np.save(run_dir / f"{laplace_prefix}_laplace_std.npy", std_post)

        for name, mu, s in zip(["sigma", "rho", "beta"],
                                theta_star.tolist(), std_post.tolist()):
            print(f"  {name}: {mu:.4f} ± {s:.4f}  (2σ = [{mu-2*s:.4f}, {mu+2*s:.4f}])")

        print(f"Saved L63 {laplace_prefix.upper().replace('_', '-')} Laplace posterior to run directory.")

    except RuntimeError as exc:
        print(f"Warning: Hessian computation failed: {exc}")

def to_float(x):
    if isinstance(x, str):
        return float(Fraction(x))
    return float(x)

if __name__ == "__main__":
    _SCRIPT_DIR = Path(__file__).resolve().parent
    sys.argv.append(f"hydra.run.dir={_SCRIPT_DIR}/runs/EM_l63_param_est_torch")
    run()
