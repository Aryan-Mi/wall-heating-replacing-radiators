"""
Task 8: Jacobi solver with a custom Numba CUDA kernel (single GPU).

CLI: python simulate_numba_cuda.py N [data_dir]

Prints CSV rows to stdout, timing to stderr as TIMING_SECONDS=<value>.
The kernel performs one Jacobi iteration per launch; the Python helper
calls it for a fixed number of iterations (no early stopping).
"""

import math
from glob import glob
import sys
import time
from os.path import join

import numpy as np
from numba import cuda

MAX_ITER = 20_000

_TPB = 16  # threads per block along each dimension


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


@cuda.jit
def _jacobi_kernel(u, u_new, interior_mask_u8):
    """One Jacobi iteration: each thread updates one interior cell."""
    i, j = cuda.grid(2)
    if i < 512 and j < 512:
        # uint8 mask avoids potential bool device-array instability on some stacks.
        if interior_mask_u8[i, j] != 0:
            u_new[i + 1, j + 1] = 0.25 * (
                u[i + 1, j] + u[i + 1, j + 2] + u[i, j + 1] + u[i + 2, j + 1]
            )
        # Non-interior cells (walls, exterior) are never written: both buffers were
        # initialised from the same u, and boundary conditions are fixed, so the
        # double-buffer swap preserves their values without any explicit copy.


def jacobi_numba_cuda(u, interior_mask, max_iter):
    # Keep dtypes/layout explicit to reduce CUDA JIT/runtime edge cases.
    u = np.ascontiguousarray(u, dtype=np.float32)
    interior_mask_u8 = np.ascontiguousarray(interior_mask, dtype=np.uint8)

    # Both device buffers initialised from u so ghost rows are preserved after swaps.
    d_u = cuda.to_device(u)
    d_u_new = cuda.to_device(u.copy())
    d_mask_u8 = cuda.to_device(interior_mask_u8)

    bpg = (math.ceil(512 / _TPB), math.ceil(512 / _TPB))
    tpb = (_TPB, _TPB)

    for _ in range(max_iter):
        _jacobi_kernel[bpg, tpb](d_u, d_u_new, d_mask_u8)
        d_u, d_u_new = d_u_new, d_u  # double-buffer swap

    cuda.synchronize()
    return d_u.copy_to_host()


if __name__ == "__main__":
    DEFAULT_DATA_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"

    N = 1 if len(sys.argv) < 2 else int(sys.argv[1])
    LOAD_DIR = DEFAULT_DATA_DIR if len(sys.argv) < 3 else sys.argv[2]

    building_ids = discover_building_ids(LOAD_DIR)[:N]
    if not building_ids:
        raise SystemExit(f"No floorplans found in data dir: {LOAD_DIR}")

    if not cuda.is_available():
        print(
            "Numba CUDA is not available in this environment. "
            "Run on a GPU node with a visible CUDA driver (e.g. via module load + uv run).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    all_u0 = []
    all_masks = []
    for bid in building_ids:
        u0, mask = load_data(LOAD_DIR, bid)
        all_u0.append(u0)
        all_masks.append(mask)

    # Warmup: trigger CUDA JIT compilation before the timed section
    jacobi_numba_cuda(all_u0[0], all_masks[0], 1)

    t0 = time.perf_counter()
    results = []
    for bid, u0, mask in zip(building_ids, all_u0, all_masks):
        u_final = jacobi_numba_cuda(u0, mask, MAX_ITER)
        stats = summary_stats(u_final, mask)
        results.append((bid, stats))
    elapsed = time.perf_counter() - t0

    print(f"TIMING_SECONDS={elapsed:.6f}", file=sys.stderr)

    stat_keys = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]
    print("building_id," + ",".join(stat_keys))
    for bid, stats in results:
        print(f"{bid}," + ",".join(str(stats[k]) for k in stat_keys))
