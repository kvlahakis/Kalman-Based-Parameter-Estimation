### UQ plots for AD-EnKF experiments

This document explains what each UQ plot produced by `uq_plots.py` shows, how it is constructed from the saved posterior samples, and how to interpret it.

The script expects posterior samples saved as `.npy` files in a results directory:

- `enkf_aug_theta_posterior.npy` — samples from the state-augmented EnKF posterior over parameters (if available)
- `ad_enkf_laplace_theta_posterior.npy` — samples from the AD-EnKF Laplace approximation
- `neural_ode_laplace_theta_posterior.npy` — samples from a Neural ODE Laplace approximation (if available)

It then compares these methods on the Lorenz-63 (L63) and gLV experiments.

---

### Lorenz‑63 marginal posterior plots (`l63_marginal_posteriors`)

- **Files**: `l63_marginal_posteriors.pdf`, `l63_marginal_posteriors.png`
- **System**: Lorenz‑63 (`--system l63`)

For L63 there are three parameters, \(\sigma, \rho, \beta\). The plotting script:

1. **Loads posterior samples**  
   From each method, it loads an array of shape \((N, 3)\):
   - column 0: samples of \(\sigma\)
   - column 1: samples of \(\rho\)
   - column 2: samples of \(\beta\)

2. **Builds marginal histograms and KDEs**  
   For each parameter separately (three subplots):
   - Plots a **histogram** of the samples for each method (normalized to a density).
   - Overlays a **Gaussian kernel density estimate (KDE)** for each method to give a smooth approximation of the marginal posterior.

3. **Overlays the true parameter value**  
   A red dashed vertical line marks the true parameter value \(\theta^*\) (e.g. \(\sigma = 10, \rho = 28, \beta = 8/3\)).

4. **Interpretation**  
   - The **center** of each distribution (mean/median) shows the point estimate for that parameter.
   - The **spread** shows posterior uncertainty; narrower distributions indicate higher confidence.
   - If the red truth line lies near the high-density region, that method is **well calibrated** on that parameter; if it falls in a low-density tail, the method systematically misestimates that parameter.

This figure is primarily about **shape** and **bias** of marginals: are the posteriors centered at the truth and how wide are they?

---

### Coverage calibration plot (`*_coverage`)

- **Files**:
  - `l63_coverage.pdf`, `l63_coverage.png` for L63 (`--system l63`)
  - `glv_coverage.pdf`, `glv_coverage.png` for gLV (`--system glv`)
- **Systems**: Both L63 and gLV

This figure checks how well the posterior uncertainty is **calibrated** across parameters using a simple Gaussian approximation to each marginal.

For each method:

1. **Compute posterior mean and std per parameter**  
   Given samples \(\theta^{(s)} \in \mathbb{R}^p\), the script computes:
   - \(\mu_j = \text{mean}(\theta_j^{(s)})\)
   - \(\sigma_j = \text{std}(\theta_j^{(s)})\)

2. **Convert truth to z‑scores**  
   For each parameter \(j\), it forms:
   \[
   z_j = \frac{|\theta^{\*,j} - \mu_j|}{\sigma_j + \varepsilon}
   \]
   where \(\theta^{\*,j}\) is the true parameter and \(\varepsilon\) is a small constant for numerical stability.

3. **Nominal vs empirical coverage**  
   For a grid of nominal coverage levels \(\alpha \in [0.05, 0.95]\):
   - Compute the corresponding Gaussian critical value \(z_\text{crit}(\alpha)\) so that
     \[
     \mathbb{P}(|Z| \le z_\text{crit}(\alpha)) = \alpha, \quad Z \sim \mathcal{N}(0,1).
     \]
   - Count the **fraction of parameters** whose \(z_j\) falls inside this interval:
     \[
     \hat{c}(\alpha) = \frac{1}{p} \sum_{j=1}^p \mathbf{1}\{z_j < z_\text{crit}(\alpha)\}.
     \]
   - Plot the curve \(\alpha \mapsto \hat{c}(\alpha)\).

4. **Interpretation**  
   - The diagonal line is **ideal calibration**: empirical coverage = nominal coverage.
   - Curves **below** the diagonal: credible intervals are too narrow (under‑dispersed, overconfident).
   - Curves **above** the diagonal: intervals are too wide (over‑dispersed, conservative).
   - This plot aggregates calibration across **all parameters** for a given method, for both L63 and gLV.

In short: the closer a method’s curve lies to the diagonal, the better calibrated its uncertainty estimates are.

---

### gLV parameter error with UQ bars (`glv_uq_bar_chart`)

- **Files**: `glv_uq_bar_chart.pdf`, `glv_uq_bar_chart.png`
- **System**: gLV (`--system glv`)

The gLV model has \(p = 20\) parameters (5 growth rates \(r_i\) and 15 interaction terms \(a_{ij}\)). This figure summarizes **parameter estimation error** and **posterior uncertainty** for each parameter and method.

For each method:

1. **Compute posterior means and stds**  
   From samples \(\theta^{(s)} \in \mathbb{R}^{20}\) it computes:
   - \(\mu_j = \text{mean}(\theta_j^{(s)})\)
   - \(\sigma_j = \text{std}(\theta_j^{(s)})\)

2. **Compute absolute estimation error**  
   For each parameter \(j\), it forms:
   \[
   e_j = |\mu_j - \theta^{\*,j}|
   \]
   which is the distance between the posterior mean and the true parameter.

3. **Plot grouped bar chart with error bars**  
   For each parameter index on the x‑axis:
   - Draw **one bar per method** showing \(e_j\).
   - Add **vertical error bars** of size \(2 \sigma_j\) (approximate 95% credible radius under a Gaussian).
   - Overlay a horizontal line at error = 0.05 as a rough performance threshold.

4. **Interpretation**  
   - **Bar height**: bias or estimation error for that parameter.
   - **Error bar length**: posterior uncertainty; larger error bars mean more uncertainty.
   - A method is doing well on parameter \(j\) if:
     - The bar is low (small bias), and
     - The 2σ interval is small and ideally includes zero error.
   - Comparing methods parameter‑wise lets you see where AD‑EnKF vs EnKF‑aug vs Neural ODE are strong or weak.

This figure focuses on **parameter recovery accuracy** and how well the posterior width matches the actual estimation error.

