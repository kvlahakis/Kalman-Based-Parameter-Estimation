import torch

def _lorenz_deriv(state, sigma=10.0, rho=28.0, beta=8/3):
    """
    Calculates the derivatives of the Lorenz '63 system.

    Args:
        state (torch.Tensor): The current state [x, y, z].
        sigma (float): Prandtl number.
        rho (float): Rayleigh number.
        beta (float): Parameter related to the layer dimensions.
        
    Returns:
        torch.Tensor: The derivatives [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return torch.stack([dx, dy, dz], dim=-1)

def _rk4_step(X, Theta, *, dt):
    """
    One Runge-Kutta 4 integration step matching the Ψ signature expected by
    StateAugEnKF.  Partially initialise with dt to obtain psi:

        psi = functools.partial(rk4_step, dt=0.01)
        # psi(X, Theta) -> X_next

    Args:
        X     (torch.Tensor): State      [..., n].
        Theta (torch.Tensor): Parameters [..., 3]  – [sigma, rho, beta].
        dt    (float):        Integration time step.

    Returns:
        torch.Tensor: Next state [..., n].
    """
    sigma, rho, beta = Theta[..., 0], Theta[..., 1], Theta[..., 2]
    k1 = _lorenz_deriv(X,                  sigma, rho, beta)
    k2 = _lorenz_deriv(X + 0.5 * dt * k1, sigma, rho, beta)
    k3 = _lorenz_deriv(X + 0.5 * dt * k2, sigma, rho, beta)
    k4 = _lorenz_deriv(X +       dt * k3, sigma, rho, beta)
    return X + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def forward(X, Theta, *, dt):
    """
    Forward pass of the Lorenz '63 system.

    Args:
        X     (torch.Tensor): State      [..., n].
        Theta (torch.Tensor): Parameters [..., 3]  – [sigma, rho, beta].
        dt    (float):        Integration time step.

    Returns:
        torch.Tensor: Next state [..., n].
    """
    return _rk4_step(X, Theta, dt=dt)