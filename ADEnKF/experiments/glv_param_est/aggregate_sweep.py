"""
Aggregate gLV AD-EnKF sweep results from a Hydra multirun directory.

Each run should have produced run_summary.yaml (final_l2_error, final_ll, and
sweep config). This script collects them, sorts by your choice of metric, and
prints a table and optionally writes a CSV for comparison.

Usage (from repo root):
  python ADEnKF/experiments/glv_param_est/aggregate_sweep.py <multirun_dir> [--sort-by l2|ll] [--csv]

Example:
  python ADEnKF/experiments/glv_param_est/aggregate_sweep.py multirun/2026-02-25/11-26-01 --sort-by l2 --csv
"""

import argparse
import sys
from pathlib import Path

import yaml


def load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate glv_param_est run_summary.yaml files from a Hydra multirun."
    )
    parser.add_argument(
        "multirun_dir",
        type=Path,
        help="Path to the multirun root (e.g. multirun/2026-02-25/11-26-01)",
    )
    parser.add_argument(
        "--sort-by",
        choices=("l2", "ll"),
        default="l2",
        help="Sort runs by: l2 = best (lowest) ||θ-θ_true||_2 first; ll = best (highest) log-likelihood first",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Write sweep_results.csv into the multirun directory",
    )
    args = parser.parse_args()

    multirun_dir = args.multirun_dir.resolve()
    if not multirun_dir.is_dir():
        print(f"Error: not a directory: {multirun_dir}", file=sys.stderr)
        sys.exit(1)

    # Find all run_summary.yaml under multirun_dir (e.g. 0/run_summary.yaml, 1/run_summary.yaml)
    summary_files = sorted(multirun_dir.glob("*/run_summary.yaml"))
    if not summary_files:
        print(f"No run_summary.yaml found under {multirun_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for p in summary_files:
        try:
            s = load_summary(p)
            s["_job_dir"] = str(p.parent)
            s["_job_id"] = p.parent.name
            rows.append(s)
        except Exception as e:
            print(f"Warning: could not load {p}: {e}", file=sys.stderr)

    if not rows:
        sys.exit(1)

    # Sort: by L2 ascending (best first) or by LL descending (best first)
    if args.sort_by == "l2":
        rows.sort(key=lambda r: r.get("final_l2_error", float("inf")))
    else:
        rows.sort(key=lambda r: r.get("final_ll", float("-inf")), reverse=True)

    # Columns to show (order and headers)
    key_cols = [
        ("_job_id", "job"),
        ("final_l2_error", "||θ-θ_true||_2"),
        ("final_ll", "log_likelihood"),
        ("best_ll", "best_LL"),
        ("best_ll_epoch", "best_LL_epoch"),
        ("init_theta_scale", "init_θ_scale"),
        ("lr", "lr"),
        ("process_std", "process_std"),
        ("N_ens", "N_ens"),
        ("chunk_length", "chunk_len"),
        ("data_truth_path", "data"),
        ("_job_dir", "run_dir"),
    ]

    # Print table (run_dir gets extra width for path)
    col_widths = [40 if h == "run_dir" else max(len(h), 10) for _, h in key_cols]
    header = "  ".join(h.center(w) for h, w in zip([k[1] for k in key_cols], col_widths))
    sep = "-" * len(header)

    print()
    print(f"Aggregated gLV sweep: {multirun_dir}")
    print(f"Runs found: {len(rows)}  (sorted by {'L2 error ↑' if args.sort_by == 'l2' else 'log-likelihood ↓'})")
    print(sep)
    print(header)
    print(sep)

    for r in rows:
        cells = []
        for (key, _), w in zip(key_cols, col_widths):
            val = r.get(key, "")
            if key == "data_truth_path" and isinstance(val, str) and len(val) > 20:
                val = "..." + val[-17:]
            if key == "_job_dir" and isinstance(val, str) and len(val) > w:
                val = "..." + val[-(w - 3) :]
            if isinstance(val, float):
                if "l2" in key or "error" in key:
                    cells.append(f"{val:.4e}".center(w))
                else:
                    cells.append(f"{val:.4f}".center(w))
            else:
                cells.append(str(val).center(w)[:w])
        print("  ".join(cells))

    print(sep)
    print(f"Best run by {args.sort_by}: job {rows[0].get('_job_id', '?')}  ({rows[0].get('_job_dir', '')})")
    print()

    if args.csv:
        import csv
        csv_path = multirun_dir / "sweep_results.csv"
        fieldnames = [k for k, _ in key_cols]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
