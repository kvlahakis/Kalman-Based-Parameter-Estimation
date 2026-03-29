# EnKF Models

This folder contains two different implementations of the Ensemble Kalman Filter (EnKF) for physical parameter estimation. Both models use the notation established in the root [README](../../README.md).

## 1. State Augmented EnKF (`state_aug_enkf`)

In this implementation, the physical parameters $\theta$ are treated as part of the state vector. We define an augmented state $z_k$:

$$z_k = \begin{bmatrix} x_k \\ \theta_k \end{bmatrix}$$

The filter then estimates the joint distribution of the system state and the parameters.

### How it Works:
- **Augmented Transition**: The transition model for the augmented state becomes:
  $$\begin{bmatrix} x_k \\ \theta_k \end{bmatrix} = \begin{bmatrix} \Psi(x_{k-1}, \theta_{k-1}) \\ \theta_{k-1} \end{bmatrix} + \begin{bmatrix} \eta_k \\ \zeta_k \end{bmatrix}$$
  where $\zeta_k$ is a small artificial parameter noise used to maintain ensemble spread and prevent parameter collapse.
- **Augmented Observation**: The observation model is applied to the augmented state:
  $$y_k = h(x_k, \theta_k) + \xi_k$$
- **Joint Update**: During the analysis step, the cross-correlations between the state and the parameters in the augmented forecast covariance $P_k^f$ allow the Kalman update to adjust the parameters based on the observed data $y_k$.

## 2. Autodiff EnKF (`autodiff_enkf`)

This implementation treats the physical parameters $\theta$ as external variables that are learned through gradient-based optimization (e.g., Stochastic Gradient Descent or Adam). This requires the entire EnKF process (forecast and update) to be differentiable.

### How it Works:
- **Differentiable EnKF**: Each step of the EnKF (forecast, Kalman gain calculation, and analysis update) is implemented using an automatic differentiation framework (like PyTorch or JAX).
 - **Objective Function**: We define a loss function $L(\theta)$ that measures the discrepancy between the filter's predictions and the observations. Common choices include:
  - **Mean Squared Error (MSE)**:
    $$L_{MSE}(\theta) = \sum_k \|y_k - h(\bar{\hat{x}}_k(\theta))\|^2$$
  - **Negative Log-Likelihood (NLL)**:
    $$L_{NLL}(\theta) = \sum_k \left( \log|\mathbf{S}_k(\theta)| + (y_k - h(\bar{\hat{x}}_k(\theta)))^T \mathbf{S}_k(\theta)^{-1} (y_k - h(\bar{\hat{x}}_k(\theta))) \right)$$
    where $\mathbf{S}_k = \mathcal{H} P_k^f \mathcal{H}^T + \Gamma$ is the innovation covariance.
- **Gradient Descent**: The parameters $\theta$ are updated by backpropagating through the EnKF operations:
  $$\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)$$
  where $\alpha$ is the learning rate.

This approach is particularly useful when the parameter space is high-dimensional or when the relationship between parameters and observations is complex.
