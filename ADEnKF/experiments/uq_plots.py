"""
UQ comparison plots for L63 and gLV.

Usage:
    PYTHONPATH=.:ADEnKF python ADEnKF/experiments/uq_plots.py --system l63
    PYTHONPATH=.:ADEnKF python ADEnKF/experiments/uq_plots.py --system glv
"""

import argparse
from pathlib import Path

import matplotlib.gridspec as gridspec  # noqa: F401  (kept for easy extensions)
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["l63", "glv"], required=True)
    parser.add_argument(
        "--results_dir",
        type=str,
        default="ADEnKF/experiments/l63_param_est/runs/AD_l63_param_est_torch",
        help="Directory containing the saved *.npy posterior files (e.g. AD-EnKF run).",
    )
    parser.add_argument(
        "--results_dir_em",
        type=str,
        default="ADEnKF/experiments/l63_param_est/runs/EM_l63_param_est_torch",
        help="Optional second directory (e.g. EM-EnKF run) to overlay in the same plots.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="ADEnKF/experiments/l63_param_est/uq_results",
        help="Output directory.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir_em = Path(args.results_dir_em) if args.results_dir_em is not None else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.system == "l63":
        param_names = [r"$\sigma$", r"$\rho$", r"$\beta$"]
        theta_true = np.array([10.0, 28.0, 8.0 / 3.0], dtype=float)
    else:
        # gLV: 5 growth rates + 15 interaction coeffs (ordering matches uq_experiments_spec)
        r_names = [f"$r_{i}$" for i in range(1, 6)]
        a_names = [
            "$a_{13}$",
            "$a_{14}$",
            "$a_{21}$",
            "$a_{23}$",
            "$a_{31}$",
            "$a_{33}$",
            "$a_{35}$",
            "$a_{42}$",
            "$a_{44}$",
            "$a_{45}$",
            "$a_{51}$",
            "$a_{52}$",
            "$a_{53}$",
            "$a_{54}$",
            "$a_{55}$",
        ]
        param_names = r_names + a_names
        theta_true = np.array(
            [
                1.3,
                1.1,
                -0.05,
                -0.3,
                -0.2,
                -0.80,
                0.0,
                0.0,
                -0.70,
                0.60,
                0.0,
                -0.25,
                0.0,
                0.45,
                0.0,
                -0.2,
                0.0,
                0.15,
                0.10,
                -0.10,
            ],
            dtype=float,
        )

    p = len(theta_true)

    def load(fname: Path):
        try:
            return np.load(fname)
        except FileNotFoundError:
            print(f"Warning: {fname} not found, skipping.")
            return None

    # Primary run (typically AD-EnKF)
    enkf_samples = load(results_dir / "enkf_aug_theta_posterior.npy")
    ad_samples = load(results_dir / "ad_enkf_laplace_theta_posterior.npy")
    node_samples = load(results_dir / "neural_ode_laplace_theta_posterior.npy")

    # Optional EM-EnKF run
    em_samples = None
    if results_dir_em is not None:
        em_samples = load(results_dir_em / "em_enkf_laplace_theta_posterior.npy")

    methods = []
    if enkf_samples is not None:
        methods.append(("EnKF-aug", enkf_samples, "C3"))
    if ad_samples is not None:
        methods.append(("AD-EnKF (Laplace)", ad_samples, "C0"))
    if em_samples is not None:
        methods.append(("EM-EnKF (Laplace)", em_samples, "C1"))
    if node_samples is not None:
        methods.append(("NeuralODE (Laplace)", node_samples, "C2"))

    if not methods:
        print("No posterior sample files found; nothing to plot.")
        return

    # ----------------------------------------------------------------
    # FIGURE 1: Marginal posteriors (L63 only — 3 params)
    # ----------------------------------------------------------------
    if args.system == "l63":
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, (ax, name, truth) in enumerate(zip(axes, param_names, theta_true)):
            for (label, samps, color) in methods:
                col = samps[:, i]
                ax.hist(col, bins=40, density=True, alpha=0.4, color=color, label=label)
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
        pdf_path = out_dir / "l63_marginal_posteriors.pdf"
        png_path = out_dir / "l63_marginal_posteriors.png"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, bbox_inches="tight", dpi=150)
        print(f"Saved {pdf_path} and {png_path}")

    # ----------------------------------------------------------------
    # FIGURE 2: Coverage plot (works for both systems)
    # ----------------------------------------------------------------
    alpha_levels = np.linspace(0.05, 0.95, 19)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")

    for (label, samps, color) in methods:
        coverages = []
        mean = samps.mean(0)
        std = samps.std(0) + 1e-12
        z = np.abs((theta_true - mean) / std)
        for alpha in alpha_levels:
            z_crit = stats.norm.ppf((1 + alpha) / 2)
            coverages.append((z < z_crit).mean())
        ax.plot(alpha_levels, coverages, "o-", color=color, label=label, markersize=5)

    ax.set_xlabel("Nominal coverage", fontsize=13)
    ax.set_ylabel("Empirical coverage", fontsize=13)
    ax.set_title(
        f"{'L63' if args.system == 'l63' else 'gLV'}: Calibration (Coverage Plot)",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    cov_pdf_path = out_dir / f"{args.system}_coverage.pdf"
    cov_png_path = out_dir / f"{args.system}_coverage.png"
    fig.savefig(cov_pdf_path, bbox_inches="tight")
    fig.savefig(cov_png_path, bbox_inches="tight", dpi=150)
    print(f"Saved {cov_pdf_path} and {cov_png_path}")

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
            stds = samps.std(0)
            err = np.abs(means - theta_true)
            ax.bar(
                x + offset,
                err,
                width,
                yerr=2 * stds,
                capsize=3,
                label=label,
                color=color,
                alpha=0.75,
                error_kw={"elinewidth": 1.5},
            )

        ax.axhline(0.05, color="red", linestyle="--", lw=1.5, label="0.05 tolerance")
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(r"$|\hat{\theta} - \theta^*|$ ± 2σ_post", fontsize=12)
        ax.set_title("gLV: Parameter Error with Posterior Uncertainty", fontsize=14)
        ax.legend(fontsize=10)
        fig.tight_layout()
        bar_pdf_path = out_dir / "glv_uq_bar_chart.pdf"
        bar_png_path = out_dir / "glv_uq_bar_chart.png"
        fig.savefig(bar_pdf_path, bbox_inches="tight")
        fig.savefig(bar_png_path, bbox_inches="tight", dpi=150)
        print(f"Saved {bar_pdf_path} and {bar_png_path}")

    plt.close("all")
    print("Done.")


if __name__ == "__main__":
    main()

