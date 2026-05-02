"""
Task 10: Benchmark CuPy optimized.

Batched convergence check vs original and compare nsys profiles.
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


def load_reference_timing(ref_csv: Path) -> float | None:
    try:
        with ref_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                return float(row["seconds_per_floorplan"])
    except (FileNotFoundError, KeyError, StopIteration):
        return None
    return None


def time_simulation(script: Path, n_floorplans: int, data_dir: str) -> tuple[float, float]:
    cmd = [sys.executable, str(script), str(n_floorplans), data_dir]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    wall = time.perf_counter() - t0

    internal = wall
    for line in result.stderr.splitlines():
        if line.startswith("TIMING_SECONDS="):
            internal = float(line.split("=", 1)[1])
            break
    return wall, internal


def count_all_floorplans(data_dir: str) -> int:
    ids_file = Path(data_dir) / "building_ids.txt"
    try:
        return sum(1 for line in ids_file.read_text().splitlines() if line.strip())
    except FileNotFoundError:
        return len(list(Path(data_dir).glob("*_domain.npy")))


def save_timing_csv(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(output_path: Path, labels: list[str], spfs: list[float]) -> None:
    mpl.use("Agg")
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
    bars = ax.barh(labels, spfs, color=colors[: len(labels)])
    ax.bar_label(bars, fmt="%.3f s", padding=4)
    ax.set_xlabel("Seconds per floorplan")
    ax.set_title("CuPy Jacobi: original vs optimized (batched convergence)")
    ax.set_xlim(0, max(spfs) * 1.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 10: benchmark CuPy optimized vs original")
    parser.add_argument("--subset", type=int, default=10)
    parser.add_argument("--data-dir", default="/dtu/projects/02613_2025/data/modified_swiss_dwellings/")
    parser.add_argument("--output-dir", default="outputs/10-nsys-cupy")
    parser.add_argument(
        "--reference-csv",
        default="outputs/09-cupy/timing_summary.csv",
        help="Task 9 timing CSV to compare against",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    here = Path(__file__).parent

    original_spf = load_reference_timing(Path(args.reference_csv))

    print(f"Running CuPy OPTIMIZED simulation on N={args.subset} floorplans...")
    wall_opt, internal_opt = time_simulation(here / "simulate_cupy_optimized.py", args.subset, args.data_dir)
    spf_opt = internal_opt / args.subset
    n_all = count_all_floorplans(args.data_dir)

    print(f"  Wall time:      {wall_opt:.3f} s")
    print(f"  Internal time:  {internal_opt:.3f} s")
    print(f"  Per-floorplan:  {spf_opt:.3f} s")
    print(f"  Est. all {n_all}: {spf_opt * n_all:.0f} s (~{spf_opt * n_all / 60:.1f} min)")

    if original_spf is not None:
        speedup = original_spf / spf_opt
        print("\nComparison vs Task 9 CuPy (original):")
        print(f"  Task 9 original:  {original_spf:.3f} s/floorplan")
        print(f"  Task 10 optimized:{spf_opt:.3f} s/floorplan")
        print(f"  Speedup:          {speedup:.2f}x")

    rows = [
        {
            "variant": "optimized",
            "subset_size": args.subset,
            "wall_seconds": f"{wall_opt:.6f}",
            "internal_seconds": f"{internal_opt:.6f}",
            "seconds_per_floorplan": f"{spf_opt:.6f}",
            "estimated_all_seconds": f"{spf_opt * n_all:.2f}",
        }
    ]
    if original_spf is not None:
        rows.insert(
            0,
            {
                "variant": "original (Task 9)",
                "subset_size": args.subset,
                "wall_seconds": "N/A",
                "internal_seconds": "N/A",
                "seconds_per_floorplan": f"{original_spf:.6f}",
                "estimated_all_seconds": f"{original_spf * n_all:.2f}",
            },
        )

    save_timing_csv(output_dir / "timing_comparison.csv", rows)

    labels = ["CuPy optimized (Task 10)"]
    spfs = [spf_opt]
    if original_spf is not None:
        labels.insert(0, "CuPy original (Task 9)")
        spfs.insert(0, original_spf)
    plot_comparison(output_dir / "timing_comparison.png", labels, spfs)
    print(f"\nResults written to {output_dir}/")


if __name__ == "__main__":
    main()
