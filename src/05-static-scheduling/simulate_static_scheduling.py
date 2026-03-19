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
    # 1. Initialize: Input image width / height
    SIZE = 512

    # 2. Prepad floor plan domain
    u = np.zeros((SIZE + 2, SIZE + 2))  # Pre-pad with a 1-pixel border to simplify Jacobi iterations.
    u[1:-1, 1:-1] = np.load(
        join(load_dir, f"{bid}_domain.npy")
    )  # Overlay the domain image on u. U is now the padded domain image

    # 3. Load interior mask
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


if __name__ == "__main__":
    # 1. Preparation
    # 1.1 Load data
    LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
    with open(join(LOAD_DIR, "building_ids.txt")) as f:
        building_ids = f.read().splitlines()

    N = 1 if len(sys.argv) < 2 else int(sys.argv[1])
    building_ids = building_ids[:N]

    # 1.2. Load floor plans and interior masks for all building IDs.
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype="bool")
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # 2. Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    # all_u shares the same shape as all_u0, but will store the updated floor plan domains after Jacobi iterations.
    all_u = np.empty_like(all_u0)

    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u

    # 3. Print summary statistics in CSV format
    stat_keys = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]
    print("building_id, " + ", ".join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
