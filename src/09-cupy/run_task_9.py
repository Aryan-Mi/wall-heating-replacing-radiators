"""
Task 9: Benchmark CuPy Jacobi solver and compare to NumPy reference.
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


def time_simulation(n_floorplans: int, data_dir: str) -> tuple[float, float]:
    script = Path(__file__).parent / "simulate_cupy.py"
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


def save_timing_csv(
    output_path: Path,
    subset: int,
    wall: float,
    internal: float,
    n_all: int,
    ref_spf: float | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spf = internal / subset
    estimated_all = spf * n_all
    row: dict[str, object] = {
        "subset_size": subset,
        "wall_seconds": f"{wall:.6f}",
        "internal_seconds": f"{internal:.6f}",
        "seconds_per_floorplan": f"{spf:.6f}",
        "num_all_floorplans": n_all,
        "estimated_all_seconds": f"{estimated_all:.2f}",
    }
    if ref_spf is not None:
        row["reference_seconds_per_floorplan"] = f"{ref_spf:.6f}"
        row["speedup_vs_reference"] = f"{ref_spf / spf:.2f}"

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def plot_comparison(output_path: Path, cupy_spf: float, ref_spf: float | None) -> None:
    mpl.use("Agg")
    labels = ["CuPy (Task 9)"]
    values = [cupy_spf]
    if ref_spf is not None:
        labels.insert(0, "NumPy reference (Task 2)")
        values.insert(0, ref_spf)

    fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
    bars = ax.barh(labels, values, color=["#4c72b0", "#dd8452"][: len(labels)][::-1])
    ax.bar_label(bars, fmt="%.3f s", padding=4)
    ax.set_xlabel("Seconds per floorplan")
    ax.set_title("Jacobi solver: per-floorplan timing")
    ax.set_xlim(0, max(values) * 1.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 9: benchmark CuPy Jacobi")
    parser.add_argument("--subset", type=int, default=10)
    parser.add_argument("--data-dir", default="/dtu/projects/02613_2025/data/modified_swiss_dwellings/")
    parser.add_argument("--output-dir", default="outputs/09-cupy")
    parser.add_argument(
        "--reference-csv",
        default="outputs/02-reference-benchmark-and-visualization/timing_summary.csv",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ref_spf = load_reference_timing(Path(args.reference_csv))

    print(f"Running CuPy simulation on N={args.subset} floorplans...")
    wall, internal = time_simulation(args.subset, args.data_dir)
    spf = internal / args.subset
    n_all = count_all_floorplans(args.data_dir)
    estimated_all = spf * n_all

    print(f"  Wall time (incl. CuPy JIT compile): {wall:.3f} s")
    print(f"  Internal time (post-warmup):         {internal:.3f} s")
    print(f"  Per-floorplan:                       {spf:.3f} s")
    print(f"  Estimated for all {n_all}:           {estimated_all:.0f} s (~{estimated_all / 60:.1f} min)")

    if ref_spf is not None:
        speedup = ref_spf / spf
        print(f"\nComparison vs reference (NumPy, Task 2):")
        print(f"  Reference:  {ref_spf:.3f} s/floorplan")
        print(f"  CuPy:       {spf:.3f} s/floorplan")
        print(f"  Speedup:    {speedup:.1f}x")

    save_timing_csv(output_dir / "timing_summary.csv", args.subset, wall, internal, n_all, ref_spf)
    plot_comparison(output_dir / "timing_comparison.png", spf, ref_spf)
    print(f"\nResults written to {output_dir}/")


if __name__ == "__main__":
    main()
