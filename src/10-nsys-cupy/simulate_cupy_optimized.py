"""
Task 10: CuPy Jacobi solver with batched convergence check to eliminate
per-iteration GPU→CPU synchronizations.

CLI: python simulate_cupy_optimized.py N [data_dir]

Prints CSV rows to stdout, timing to stderr as TIMING_SECONDS=<value>.
"""

from glob import glob
import sys
import time
from os.path import join

import cupy as cp
import numpy as np

MAX_ITER = 20_000
ABS_TOL = 1e-4
CHECK_INTERVAL = 100  # only sync GPU→CPU every this many iterations


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        "mean_temp": mean_temp,
        "std_temp": std_temp,
        "pct_above_18": pct_above_18,
        "pct_below_15": pct_below_15,
    }


def discover_building_ids(data_dir):
    ids_file = join(data_dir, "building_ids.txt")
    try:
        with open(ids_file) as f:
            return f.read().splitlines()
    except FileNotFoundError:
        domain_files = glob(join(data_dir, "*_domain.npy"))
        return sorted(f.split("/")[-1].removesuffix("_domain.npy") for f in domain_files)


def jacobi_cupy_optimized(u, interior_mask, max_iter, atol=1e-4, check_interval=CHECK_INTERVAL):
    u_gpu = cp.asarray(u)
    mask_gpu = cp.asarray(interior_mask)

    for i in range(max_iter):
        u_new = 0.25 * (
            u_gpu[1:-1, :-2] + u_gpu[1:-1, 2:]
            + u_gpu[:-2, 1:-1] + u_gpu[2:, 1:-1]
        )

        # Only transfer delta to CPU every check_interval iterations.
        # This reduces GPU→CPU synchronizations from up to 20,000 down to ~200,
        # keeping the GPU pipeline full between checks.
        if i % check_interval == check_interval - 1:
            delta = float(cp.abs(u_new[mask_gpu] - u_gpu[1:-1, 1:-1][mask_gpu]).max())
            converged = delta < atol
        else:
            converged = False

        u_gpu[1:-1, 1:-1] = cp.where(mask_gpu, u_new, u_gpu[1:-1, 1:-1])

        if converged:
            break

    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(u_gpu)


if __name__ == "__main__":
    DEFAULT_DATA_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"

    N = 1 if len(sys.argv) < 2 else int(sys.argv[1])
    LOAD_DIR = DEFAULT_DATA_DIR if len(sys.argv) < 3 else sys.argv[2]

    building_ids = discover_building_ids(LOAD_DIR)[:N]

    all_u0 = []
    all_masks = []
    for bid in building_ids:
        u0, mask = load_data(LOAD_DIR, bid)
        all_u0.append(u0)
        all_masks.append(mask)

    # Warmup: trigger CuPy kernel compilation before the timed section
    jacobi_cupy_optimized(all_u0[0], all_masks[0], 1)

    t0 = time.perf_counter()
    results = []
    for bid, u0, mask in zip(building_ids, all_u0, all_masks):
        u_final = jacobi_cupy_optimized(u0, mask, MAX_ITER, ABS_TOL)
        stats = summary_stats(u_final, mask)
        results.append((bid, stats))
    elapsed = time.perf_counter() - t0

    print(f"TIMING_SECONDS={elapsed:.6f}", file=sys.stderr)

    stat_keys = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]
    print("building_id," + ",".join(stat_keys))
    for bid, stats in results:
        print(f"{bid}," + ",".join(str(stats[k]) for k in stat_keys))
