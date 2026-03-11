# Task 1: Visualize floor plans

"""Visualize input data (domain and interior mask) for all floor plans."""

from os.path import join
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "data"
OUTPUT_DIR = "outputs/01-visualize-floorplans"


def load_data(load_dir: str, bid: str) -> tuple[np.ndarray, np.ndarray]:
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def visualize_floorplan(
    bid: str, u: np.ndarray, interior_mask: np.ndarray, output_dir: str
) -> None:
    domain = u[1:-1, 1:-1]  # 512x512 inner grid

    # Build an RGB image encoding three regions:
    #   outside  (domain == 0 and not interior): dark grey
    #   wall/boundary (domain > 0):
    #       cold boundary (5 °C): blue-ish
    #       heated boundary (25 °C): red-ish
    #   interior (free cells): white
    rgb = np.ones((*domain.shape, 3), dtype=float)  # start white

    outside_mask = (domain == 0) & ~interior_mask
    cold_wall_mask = np.isclose(domain, 5.0) & ~interior_mask
    hot_wall_mask = np.isclose(domain, 25.0) & ~interior_mask

    rgb[outside_mask] = [0.15, 0.15, 0.15]  # dark grey - outside
    rgb[cold_wall_mask] = [0.2, 0.4, 0.9]  # blue - cold boundary (5 °C)
    rgb[hot_wall_mask] = [0.9, 0.2, 0.2]  # red - heated boundary (25 °C)
    # interior cells remain white

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Floor plan {bid}", fontsize=14, fontweight="bold")

    # Left: colour-coded domain regions
    axes[0].imshow(rgb, origin="upper")
    axes[0].set_title("Domain (boundary conditions)")
    axes[0].axis("off")
    legend_patches = [
        mpatches.Patch(color=(0.15, 0.15, 0.15), label="Outside"),
        mpatches.Patch(color=(0.2, 0.4, 0.9), label="Cold wall (5 °C)"),
        mpatches.Patch(color=(0.9, 0.2, 0.2), label="Heated wall (25 °C)"),
        mpatches.Patch(color=(1.0, 1.0, 1.0), label="Interior (free)"),
    ]
    axes[0].legend(
        handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.8
    )

    # Right: interior mask
    axes[1].imshow(interior_mask, cmap="gray", origin="upper")
    axes[1].set_title("Interior mask (white = free cells)")
    axes[1].axis("off")

    plt.tight_layout()
    out_path = Path(output_dir) / f"{bid}_floorplan.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    # Discover all building IDs from files present in DATA_DIR
    data_path = Path(DATA_DIR)
    building_ids = sorted(
        p.stem.replace("_domain", "") for p in data_path.glob("*_domain.npy")
    )

    if not building_ids:
        print(f"No floor plan data found in '{DATA_DIR}'.")
        return

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Found {len(building_ids)} floor plans: {', '.join(building_ids)}")

    for bid in building_ids:
        u, interior_mask = load_data(DATA_DIR, bid)
        visualize_floorplan(bid, u, interior_mask, OUTPUT_DIR)

    print(f"\nAll visualizations saved to '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
