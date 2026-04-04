from glob import glob
from multiprocessing import Pool
from os.path import join
import sys

import numpy as np


def load_data(load_dir, bid):
    """
    Load the floor plan domain and interior mask for a building ID.

    and interior mask for a given building ID.

    Args:
        load_dir (str): The directory of the data folder.
        bid (str): The building ID to load.

    Returns:
        u: Shape: (514, 514): The floor plan domain (prepadded for jacobi iterations).
        interior_mask (512, 512): The mask indicating where the interior lies.

    """
    SIZE = 512

    u = np.zeros((SIZE + 2, SIZE + 2))  # Pre-pad with a 1-pixel border to simplify Jacobi iterations.
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))

    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))

    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    """
    Perform Jacobi iterations for the simulation.

    Update each interior point to the average of its four neighbors
    (left, right, up, down).

    Args:
        u: Shape: (514, 514): Prepadded floor plan domain
        interior_mask: Shape: (512, 512): The mask indicating where the interior lies.
        max_iter: The maximum number of Jacobi iterations to perform.
        atol: The absolute tolerance for convergence.

    Returns:
        u: The updated floor plan domain after Jacobi iterations.

    """
    u = np.copy(u)

    # Jacobi iterations: Iterating up to max_iter times
    for _ in range(max_iter):
        # 1. Compute average of left, right, up and down neighbors, see the second equation in the pdf
        # Note that u is prepadded, hence the 1:-1 indexing to skip the padding.
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])

        # 2. Calculate delta for stopping criterion
        u_new_interior = u_new[interior_mask]  # Consider only interior points; boundary points remain fixed.
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()

        # 3. Update u at the interior points with the new values.
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        # 4. Stopping criterion: If the maximum change in temperature at any interior point
        # is less than the specified absolute tolerance (atol),
        # we consider the solution to have converged and stop.
        if delta < atol:
            break

    return u


def summary_stats(u, interior_mask):
    """
    Compute summary statistics for interior temperatures after convergence.

    Args:
        u: Shape: (514, 514): The floor plan domain after Jacobi iterations.
        interior_mask: Shape: (512, 512): The mask indicating where the interior lies.

    Returns:
        stats (dict): A dictionary containing the summary statistics.

    """
    # 1. Restricts the analysis to the interior points of the floor plan (where interior_mask is True)
    u_interior = u[1:-1, 1:-1][interior_mask]

    # 2. Computes the mean and standard deviation of the interior temperatures,
    # as well as the percentage of interior points above 18 degrees and below 15 degrees.
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
    """
    Discover available building IDs from the data directory.

    Reads building_ids.txt if present, otherwise globs *_domain.npy files.

    Args:
        data_dir (str): Path to the data directory.

    Returns:
        list[str]: Sorted list of building ID strings.

    """
    ids_file = join(data_dir, "building_ids.txt")
    try:
        with open(ids_file) as f:
            return f.read().splitlines()
    except FileNotFoundError:
        domain_files = glob(join(data_dir, "*_domain.npy"))
        return sorted(f.split("/")[-1].removesuffix("_domain.npy") for f in domain_files)


def process_building(args):
    """
    Worker function: load, solve, and summarize a single building.

    Each worker independently loads data from disk, runs the Jacobi solver,
    and computes summary statistics. This avoids pickling large numpy arrays.

    Args:
        args: Tuple of (load_dir, bid, max_iter, atol).

    Returns:
        Tuple of (bid, stats_dict).

    """
    load_dir, bid, max_iter, atol = args
    u, interior_mask = load_data(load_dir, bid)
    u = jacobi(u, interior_mask, max_iter, atol)
    stats = summary_stats(u, interior_mask)
    return (bid, stats)


if __name__ == "__main__":
    # 1. Parse arguments: N (floorplan count), P (worker count), DATA_DIR
    DEFAULT_DATA_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"

    N = 1 if len(sys.argv) < 2 else int(sys.argv[1])
    P = 1 if len(sys.argv) < 3 else int(sys.argv[2])
    LOAD_DIR = DEFAULT_DATA_DIR if len(sys.argv) < 4 else sys.argv[3]

    # 2. Discover and select building IDs
    building_ids = discover_building_ids(LOAD_DIR)
    building_ids = building_ids[:N]

    # 3. Build work items (lightweight tuples — no large arrays cross process boundaries)
    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    work_items = tuple((LOAD_DIR, bid, MAX_ITER, ABS_TOL) for bid in building_ids)

    # 4. Run simulation: sequential or parallel with dynamic scheduling
    #    Dynamic scheduling uses chunksize=1 so that each floorplan is dispatched
    #    individually to the next available worker. This handles varying convergence
    #    times better than static scheduling.
    if P <= 1:
        results = [process_building(item) for item in work_items]
    else:
        with Pool(processes=P) as pool:
            results = pool.map(process_building, work_items, chunksize=1)

    # 5. Print summary statistics in CSV format (order preserved by pool.map)
    stat_keys = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]
    print("building_id, " + ", ".join(stat_keys))
    for bid, stats in results:
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
