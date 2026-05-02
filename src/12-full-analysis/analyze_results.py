"""
Task 12: Analyze simulation results for all floorplans.

Expects a CSV produced by one of the simulate_*.py scripts with columns:
    building_id, mean_temp, std_temp, pct_above_18, pct_below_15

Usage:
    python src/12-full-analysis/analyze_results.py results.csv [--output-dir outputs/12-full-analysis]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def print_summary(df: pd.DataFrame) -> None:
    print(f"Total buildings processed: {len(df)}")
    print()

    avg_mean = df["mean_temp"].mean()
    avg_std = df["std_temp"].mean()
    print(f"12b) Average mean temperature:        {avg_mean:.4f} °C")
    print(f"12c) Average temperature std dev:     {avg_std:.4f} °C")

    n_above_18 = (df["pct_above_18"] >= 50).sum()
    n_below_15 = (df["pct_below_15"] >= 50).sum()
    print(f"12d) Buildings with ≥50% area above 18°C: {n_above_18} ({n_above_18 / len(df) * 100:.1f}%)")
    print(f"12e) Buildings with ≥50% area below 15°C: {n_below_15} ({n_below_15 / len(df) * 100:.1f}%)")


def plot_histograms(df: pd.DataFrame, output_dir: Path) -> None:
    mpl.use("Agg")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].hist(df["mean_temp"], bins=50, color="#4c72b0", edgecolor="white")
    axes[0, 0].set_xlabel("Mean temperature (°C)")
    axes[0, 0].set_ylabel("Number of buildings")
    axes[0, 0].set_title("Distribution of mean interior temperatures")
    axes[0, 0].axvline(
        df["mean_temp"].mean(), color="red", linestyle="--", label=f"Mean = {df['mean_temp'].mean():.2f}°C"
    )
    axes[0, 0].legend()

    axes[0, 1].hist(df["std_temp"], bins=50, color="#dd8452", edgecolor="white")
    axes[0, 1].set_xlabel("Std dev of temperature (°C)")
    axes[0, 1].set_ylabel("Number of buildings")
    axes[0, 1].set_title("Distribution of temperature standard deviation")

    axes[1, 0].hist(df["pct_above_18"], bins=50, color="#55a868", edgecolor="white")
    axes[1, 0].axvline(50, color="red", linestyle="--", label="50% threshold")
    axes[1, 0].set_xlabel("% area above 18°C")
    axes[1, 0].set_ylabel("Number of buildings")
    axes[1, 0].set_title("Distribution of % area above 18°C")
    axes[1, 0].legend()

    axes[1, 1].hist(df["pct_below_15"], bins=50, color="#c44e52", edgecolor="white")
    axes[1, 1].axvline(50, color="red", linestyle="--", label="50% threshold")
    axes[1, 1].set_xlabel("% area below 15°C")
    axes[1, 1].set_ylabel("Number of buildings")
    axes[1, 1].set_title("Distribution of % area below 15°C")
    axes[1, 1].legend()

    fig.suptitle("Wall Heating Simulation — Full Dataset Analysis", fontsize=13)
    fig.savefig(output_dir / "histograms.png", dpi=160)
    plt.close(fig)
    print(f"Histograms saved to {output_dir / 'histograms.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 12: analyze full simulation results")
    parser.add_argument("results_csv", help="CSV file with simulation results")
    parser.add_argument("--output-dir", default="outputs/12-full-analysis")
    args = parser.parse_args()

    df = load_results(Path(args.results_csv))
    output_dir = Path(args.output_dir)

    print_summary(df)
    plot_histograms(df, output_dir)


if __name__ == "__main__":
    main()
