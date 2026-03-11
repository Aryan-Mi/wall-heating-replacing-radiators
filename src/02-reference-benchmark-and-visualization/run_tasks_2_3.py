"""
Task 2 + 3: benchmark reference simulation and visualize outputs.

This script keeps the reference Jacobi implementation from simulate.py,
runs it on a subset of floorplans, times repeated batch runs for reliability,
estimates runtime for all floorplans, and saves simulation plots.
"""

from __future__ import annotations

import argparse
import csv
from os.path import join
from pathlib import Path
import statistics
import time

import numpy as np


def load_data(load_dir: str, bid: str) -> tuple[np.ndarray, np.ndarray]:
    size = 512
    u = np.zeros((size + 2, size + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(
    u: np.ndarray, interior_mask: np.ndarray, max_iter: int, atol: float = 1e-6
) -> np.ndarray:
    # Reference implementation logic from simulate.py.
    u = np.copy(u)

    for _ in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    return u


def summary_stats(u: np.ndarray, interior_mask: np.ndarray) -> dict[str, float]:
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        "mean_temp": float(mean_temp),
        "std_temp": float(std_temp),
        "pct_above_18": float(pct_above_18),
        "pct_below_15": float(pct_below_15),
    }


def discover_building_ids(data_dir: Path) -> list[str]:
    ids_file = data_dir / "building_ids.txt"
    if ids_file.exists():
        return ids_file.read_text().splitlines()

    domain_ids = {
        p.name.removesuffix("_domain.npy") for p in data_dir.glob("*_domain.npy")
    }
    interior_ids = {
        p.name.removesuffix("_interior.npy") for p in data_dir.glob("*_interior.npy")
    }
    return sorted(domain_ids & interior_ids)


def run_reference_batch(
    all_u0: np.ndarray,
    all_interior_mask: np.ndarray,
    max_iter: int,
    atol: float,
) -> np.ndarray:
    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        all_u[i] = jacobi(u0, interior_mask, max_iter, atol)
    return all_u


def building_bbox(
    domain: np.ndarray, interior_mask: np.ndarray
) -> tuple[int, int, int, int, np.ndarray]:
    wall_mask = np.isclose(domain, 5.0) | np.isclose(domain, 25.0)
    building_mask = interior_mask | wall_mask

    rows, cols = np.where(building_mask)
    if rows.size == 0:
        return 0, domain.shape[0], 0, domain.shape[1], building_mask

    pad = 6
    r0 = max(int(rows.min()) - pad, 0)
    r1 = min(int(rows.max()) + pad + 1, domain.shape[0])
    c0 = max(int(cols.min()) - pad, 0)
    c1 = min(int(cols.max()) + pad + 1, domain.shape[1])
    return r0, r1, c0, c1, building_mask


def save_simulation_plots(
    building_ids: list[str],
    all_u0: np.ndarray,
    all_interior_mask: np.ndarray,
    all_u: np.ndarray,
    plot_count: int,
    output_dir: Path,
) -> None:
    if plot_count <= 0:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    count = min(plot_count, len(building_ids))

    for i in range(count):
        bid = building_ids[i]
        domain = all_u0[i][1:-1, 1:-1]
        interior_mask = all_interior_mask[i]
        sim = all_u[i][1:-1, 1:-1]

        r0, r1, c0, c1, building_mask = building_bbox(domain, interior_mask)
        region_mask = building_mask[r0:r1, c0:c1]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

        input_crop = np.where(region_mask, domain[r0:r1, c0:c1], np.nan)
        im0 = axes[0].imshow(input_crop, cmap="coolwarm", vmin=0, vmax=25)
        axes[0].set_title(f"Input domain {bid}")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        sim_crop = np.where(region_mask, sim[r0:r1, c0:c1], np.nan)
        im1 = axes[1].imshow(sim_crop, cmap="coolwarm", vmin=0, vmax=25)
        axes[1].set_title(f"Simulated temperature {bid}")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        fig.colorbar(im1, ax=axes, shrink=0.82, label="Temperature [C]")
        out_path = output_dir / f"{bid}_simulation.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)


def write_stats_csv(
    building_ids: list[str],
    all_u: np.ndarray,
    all_interior_mask: np.ndarray,
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    stat_keys = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["building_id", *stat_keys])
        for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
            stats = summary_stats(u, interior_mask)
            writer.writerow([bid, *(stats[k] for k in stat_keys)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 2+3 reference benchmark and visualization"
    )
    parser.add_argument(
        "--subset", type=int, default=10, help="Number of floorplans to simulate"
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Number of repeated batch timings"
    )
    parser.add_argument(
        "--plot-count", type=int, default=3, help="How many floorplans to plot"
    )
    parser.add_argument("--max-iter", type=int, default=20_000)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--output-dir",
        default="outputs/02-reference-benchmark-and-visualization",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    all_ids = discover_building_ids(data_dir)
    if not all_ids:
        msg = f"No floorplans found in {data_dir}"
        raise ValueError(msg)

    subset_n = min(args.subset, len(all_ids))
    subset_ids = all_ids[:subset_n]

    all_u0 = np.empty((subset_n, 514, 514))
    all_interior_mask = np.empty((subset_n, 512, 512), dtype=bool)
    for i, bid in enumerate(subset_ids):
        u0, interior_mask = load_data(str(data_dir), bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    batch_times: list[float] = []
    all_u = np.empty_like(all_u0)

    for run_idx in range(args.repeats):
        t0 = time.perf_counter()
        all_u = run_reference_batch(all_u0, all_interior_mask, args.max_iter, args.atol)
        elapsed = time.perf_counter() - t0
        batch_times.append(elapsed)
        print(f"Run {run_idx + 1}/{args.repeats}: {elapsed:.3f} s")

    mean_batch = statistics.mean(batch_times)
    stdev_batch = statistics.stdev(batch_times) if len(batch_times) > 1 else 0.0
    seconds_per_floorplan = mean_batch / subset_n
    estimate_all = seconds_per_floorplan * len(all_ids)

    save_simulation_plots(
        building_ids=subset_ids,
        all_u0=all_u0,
        all_interior_mask=all_interior_mask,
        all_u=all_u,
        plot_count=args.plot_count,
        output_dir=output_dir / "plots",
    )

    write_stats_csv(
        building_ids=subset_ids,
        all_u=all_u,
        all_interior_mask=all_interior_mask,
        output_csv=output_dir / "stats_subset.csv",
    )

    timing_csv = output_dir / "timing_summary.csv"
    timing_csv.parent.mkdir(parents=True, exist_ok=True)
    with timing_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "subset_size",
                "repeats",
                "mean_batch_seconds",
                "stdev_batch_seconds",
                "seconds_per_floorplan",
                "num_all_floorplans",
                "estimated_all_seconds",
            ]
        )
        writer.writerow(
            [
                subset_n,
                args.repeats,
                f"{mean_batch:.6f}",
                f"{stdev_batch:.6f}",
                f"{seconds_per_floorplan:.6f}",
                len(all_ids),
                f"{estimate_all:.6f}",
            ]
        )

    print("\nTiming summary")
    print(f"Subset size: {subset_n}")
    print(f"Total floorplans available: {len(all_ids)}")
    print(f"Batch runs: {args.repeats}")
    print(f"Batch times [s]: {', '.join(f'{t:.3f}' for t in batch_times)}")
    print(f"Mean batch time [s]: {mean_batch:.3f}")
    print(f"Std. dev. [s]: {stdev_batch:.3f}")
    print(f"Estimated time for all floorplans [s]: {estimate_all:.3f}")


if __name__ == "__main__":
    main()
