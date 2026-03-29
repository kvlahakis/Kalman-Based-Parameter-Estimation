from tqdm import tqdm
import os
import sys
sys.path.append(os.getcwd())

from torchEnKF import da_methods, nn_templates, noise
from examples import generate_data, utils

import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

######### Define reference model (sigma, rho, beta — same order as EnKF_PPE Lorentz63) #########
x_dim = 3
true_coeff = torch.tensor([10., 28., 8/3], device=device)  # sigma, rho, beta
true_ode_func = nn_templates.Lorenz63(true_coeff).to(device)

######### Draw x0 from the L63 attractor (short warmup) #########
train_size = 4
with torch.no_grad():
    x0_warmup = torch.distributions.MultivariateNormal(
        torch.zeros(x_dim),
        covariance_matrix=torch.diag(torch.tensor([25., 25., 50.]))
    ).sample().to(device)
    t_warmup = 40 * torch.arange(0., train_size + 1).to(device)
    x0 = odeint(
        true_ode_func, x0_warmup, t_warmup,
        method='rk4', options=dict(step_size=0.01)
    )[1:]  # Shape: (train_size, x_dim)

######### Generate training data #########
t0 = 0.
t_obs_step = 0.05
n_obs = 400
t_obs = t_obs_step * torch.arange(1, n_obs + 1).to(device)

model_Q_true = None

indices = [0, 1, 2]
y_dim = len(indices)
H_true = torch.eye(x_dim)[indices]
true_obs_func = nn_templates.Linear(x_dim, y_dim, H=H_true).to(device)
noise_R_true = noise.AddGaussian(y_dim, torch.tensor(1.), param_type='scalar').to(device)

with torch.no_grad():
    x_truth, y_obs = generate_data.generate(
        true_ode_func, true_obs_func, t_obs, x0,
        model_Q_true, noise_R_true, device=device,
        ode_method='rk4', ode_options=dict(step_size=0.01), tqdm=tqdm
    )  # Shapes: (n_obs, train_size, x_dim), (n_obs, train_size, y_dim)

######### Parameter estimation #########
N_ensem = 50

init_m = torch.zeros(x_dim, device=device)
init_C_param = noise.AddGaussian(
    x_dim,
    torch.diag(torch.tensor([25., 25., 50.])),
    'full'
).to(device)

init_coeff = torch.tensor([5., 15., 1.], device=device)  # sigma, rho, beta (deliberately off true)
learned_ode_func = nn_templates.Lorenz63(init_coeff).to(device)

init_Q = 0.05 * torch.ones(x_dim)
learned_model_Q = noise.AddGaussian(x_dim, init_Q, 'diag').to(device)

optimizer = torch.optim.Adam([
    {'params': learned_ode_func.parameters(), 'lr': 1e-1},
    {'params': learned_model_Q.parameters(), 'lr': 1e-1}
])
lambda_sched = lambda epoch: (epoch - 9) ** (-0.4) if epoch >= 30 else 1
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=[lambda_sched, lambda_sched]
)

L = 20
monitor = []

for epoch in tqdm(range(150), desc="Training", leave=False):
    train_log_likelihood = torch.zeros(train_size, device=device)
    t_start = t0  # reset time cursor each epoch
    X = init_C_param(init_m.expand(train_size, N_ensem, x_dim))

    for start in range(0, n_obs, L):
        optimizer.zero_grad()
        end = min(start + L, n_obs)

        X, res, log_likelihood = da_methods.EnKF(
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
            ode_method='rk4',
            ode_options=dict(step_size=0.01),
            adjoint=True,              # use adjoint for memory-efficient backprop
            adjoint_method='rk4',
            adjoint_options=dict(step_size=0.05),
            # No localization_radius — meaningless in 3D
            tqdm=None
        )
        t_start = t_obs[end - 1]
        (-log_likelihood).mean().backward()
        train_log_likelihood += log_likelihood.detach().clone()
        optimizer.step()

    scheduler.step()

    if epoch % 10 == 0:
        sigma, rho, beta = learned_ode_func.coeff.data.cpu()
        tqdm.write(
            f"Epoch {epoch} | LL: {train_log_likelihood.mean().item():.2f} | "
            f"sigma={sigma:.3f} (true=10.000) | "
            f"rho={rho:.3f} (true=28.000) | "
            f"beta={beta:.3f} (true=2.667)"
        )

    with torch.no_grad():
        q_scale = torch.sqrt(torch.trace(learned_model_Q.full()) / x_dim)
        sigma, rho, beta = learned_ode_func.coeff.tolist()
        monitor.append([sigma, rho, beta, q_scale.item(), train_log_likelihood.mean().item()])

######### Plot results #########
monitor = np.asarray(monitor)
true_vals = [10., 28., 8/3]  # sigma, rho, beta
param_names = ['sigma', 'rho', 'beta']

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
for i, (name, true_val) in enumerate(zip(param_names, true_vals)):
    line, = ax.plot(monitor[:, i], label=f'{name} (learned)')
    ax.axhline(true_val, color=line.get_color(), linestyle='--', alpha=0.5,
               label=f'{name} true={true_val:.2f}')
ax.set_title('Parameter convergence')
ax.set_xlabel('Epoch')
ax.legend(fontsize=8)

axes[1].plot(monitor[:, 3])
axes[1].set_title('Model noise scale (q)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('q')

axes[2].plot(monitor[:, 4])
axes[2].set_title('Training log-likelihood')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Log-likelihood')

plt.tight_layout()
plt.savefig('l63_parameter_estimation.png', dpi=150)
plt.show()