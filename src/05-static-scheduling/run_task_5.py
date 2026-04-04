"""
Task 5: Benchmark static scheduling parallelization.

Measures speedup of multiprocessing Pool-based static scheduling
across worker counts for a subset of floorplans.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt

WORKER_COUNTS = (1, 2, 4, 8, 16, 32, 64)


def time_simulation(n_floorplans: int, n_workers: int, data_dir: str) -> float:
    """Run simulate_static_scheduling.py and return wall-clock seconds."""
    script = Path(__file__).parent / "simulate_static_scheduling.py"
    cmd = [sys.executable, str(script), str(n_floorplans), str(n_workers), data_dir]
    t0 = time.perf_counter()
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return time.perf_counter() - t0


def compute_speedups(timings: dict[int, float]) -> dict[int, float]:
    """Compute speedup S(P) = T(1) / T(P) for each worker count."""
    t_serial = timings[1]
    return {p: t_serial / t for p, t in timings.items()}


def save_timing_csv(
    timings: dict[int, float],
    speedups: dict[int, float],
    output_path: Path,
) -> None:
    """Write timing results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["workers", "time_seconds", "speedup"])
        for p in sorted(timings):
            writer.writerow([p, f"{timings[p]:.6f}", f"{speedups[p]:.4f}"])


def plot_speedup(speedups: dict[int, float], output_path: Path) -> None:
    """Plot speedup vs worker count and save to PNG."""
    mpl.use("Agg")

    workers = sorted(speedups.keys())
    speedup_vals = [speedups[p] for p in workers]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(workers, speedup_vals, "o-", label="Measured speedup", linewidth=2)
    ax.plot(workers, workers, "--", color="gray", alpha=0.5, label="Ideal (linear)")
    ax.set_xlabel("Number of workers")
    ax.set_ylabel("Speedup S(P) = T(1) / T(P)")
    ax.set_title("Static Scheduling Speedup")
    ax.set_xticks(workers)
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 5: benchmark static scheduling parallelization")
    parser.add_argument(
        "--subset",
        type=int,
        default=10,
        help="Number of floorplans to simulate",
    )
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument(
        "--output-dir",
        default="outputs/05-static-scheduling",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    subset = args.subset

    # Only test worker counts that make sense for the subset size
    worker_counts = [p for p in WORKER_COUNTS if p <= subset]

    timings: dict[int, float] = {}

    for p in worker_counts:
        print(f"Running with {p} worker(s)...")
        elapsed = time_simulation(subset, p, args.data_dir)
        timings[p] = elapsed
        print(f"  {p} workers: {elapsed:.3f} s")

    speedups = compute_speedups(timings)

    save_timing_csv(timings, speedups, output_dir / "timing_results.csv")
    plot_speedup(speedups, output_dir / "speedup.png")

    # Print summary
    print("\nSpeedup summary:")
    for p in sorted(speedups):
        print(f"  P={p:>2}: T={timings[p]:.3f}s, S={speedups[p]:.2f}x")

    # Estimate for all floorplans using fastest config
    best_p = max(speedups, key=speedups.get)  # type: ignore  # noqa: PGH003
    best_time = timings[best_p]
    secs_per_fp = best_time / subset
    print(f"\nBest: P={best_p} ({best_time:.3f}s for {subset} floorplans)")
    print(f"Per-floorplan: {secs_per_fp:.3f}s")


if __name__ == "__main__":
    main()
