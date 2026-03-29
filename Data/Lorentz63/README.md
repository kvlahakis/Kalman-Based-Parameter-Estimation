# Lorenz '63 System

The Lorenz '63 system is a simplified mathematical model for atmospheric convection, originally developed by Edward Lorenz in 1963. It is a system of three ordinary differential equations (ODEs) that is famous for having chaotic solutions for certain parameter values and initial conditions.

The system is a classic benchmark for data assimilation and filtering algorithms like the Ensemble Kalman Filter (EnKF) because it exhibits nonlinear, chaotic dynamics.

## Governing Equations

The state of the system is defined by three variables $(x, y, z)$, and its evolution is governed by:

$$ \frac{dx}{dt} = \sigma (y - x) $$
$$ \frac{dy}{dt} = x (\rho - z) - y $$
$$ \frac{dz}{dt} = xy - \beta z $$

### Parameters
The system behavior is determined by three physical parameters:
- $\sigma$ (Prandtl number): Related to the ratio of fluid viscosity to thermal conductivity.
- $\rho$ (Rayleigh number): Related to the temperature difference driving the convection.
- $\beta$: Related to the physical dimensions of the layer.

The standard values that lead to the famous "butterfly" chaotic attractor are:
- $\sigma = 10$
- $\rho = 28$
- $\beta = 8/3$

## Significance in PPE
In the context of Physical Parameter Estimation (PPE), the Lorenz '63 system is used to test how well an EnKF can recover the "true" values of $\sigma$, $\rho$, and $\beta$ when only noisy observations of $(x, y, z)$ (or a subset of them) are available.

## Data Generation

To generate or extend a Lorenz '63 dataset, use the `generate_data.py` script. This script uses the Runge-Kutta 4 (RK4) integration scheme and saves the resulting trajectory as a PyTorch tensor (`.pt` file).

### How it Works:
- **State Persistence**: If the specified file already exists, the script loads the last state and continues the simulation from that point, ensuring a continuous trajectory.
- **RK4 Scheme**: The integration uses a fixed-step RK4 solver for better numerical stability compared to the simple Euler method.
- **Vectorized PyTorch**: The data is stored and processed as PyTorch tensors for easy model training.

### Usage:
The script automatically generates a filename based on the simulation parameters (e.g., `sigma10.0000_rho28.0000_beta2.6667_dt0.0100.pt`).

```bash
python Data/Lorentz63/generate_data.py --steps 5000 --dt 0.01
```

**Common Arguments:**
- `--steps`: Number of steps to generate (default: `10000`).
- `--dt`: Integration time step size (default: `0.01`).
- `--sigma`, `--rho`, `--beta`: Parameters of the Lorenz system.
- `--x0`: Initial state $[x, y, z]$ if starting a new dataset (default: `1.0 1.0 1.0`).

### Saved Data Format:
The script saves a dictionary with the following keys:
- `data`: A PyTorch tensor of shape `(total_steps, 3)` containing the $(x, y, z)$ trajectory.
- `metadata`: A dictionary containing the simulation parameters (`num_steps`, `sigma`, `rho`, `beta`, `dt`, `initial_state`).

## Data Visualization

To visualize a generated Lorenz '63 dataset, use the `visualize_dataset.py` script. This script opens an interactive 3D window showing the trajectory using **Plotly**.

### Usage:

```bash
uv run Data/Lorentz63/visualize_dataset.py "Data/Lorentz63/sigma10.0000_rho28.0000_beta2.6667_dt0.0100.pt"
```

**Key Features:**
- **Interactive 3D**: Rotate, zoom, and pan to explore the chaotic attractor.
- **Time-Based Coloring**: The trajectory is colored by the time step using the Viridis colorscale.
- **Hover Information**: Hover over any point to see its $(x, y, z)$ coordinates and the corresponding time step.
- **Visual Cues**: Clearly marks the starting point (green) and ending point (red) of the trajectory.
