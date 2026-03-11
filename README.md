# 02613 Mini-Project: Wall Heating!

This repository contains solution for the **02613 Python and High Performance Computing** mini-project on *Wall Heating*.

The goal is to simulate steady-state heat distribution in 2D building floorplans and then optimize the implementation using various HPC techniques (parallelization, Numba, CUDA, CuPy). Finally, the results are used to evaluate the viability of the Wall Heating concept.

## Project overview

We model building floorplans as 2D grids where:

- Load-bearing walls are kept at 5°C.  
- Interior walls are kept at 25°C.  
- Room interiors start at 0°C and evolve according to the steady-state heat equation (Laplace’s equation).

The temperature field $u(x, y)$ is discretized on a 512 × 512 grid (with a 1-cell padding border), and the steady-state solution is approximated with the **Jacobi method**:

$$
u_{i,j}^{(k+1)} = \frac{1}{4}\left(u_{i-1,j}^{(k)} + u_{i+1,j}^{(k)} + u_{i,j-1}^{(k)} + u_{i,j+1}^{(k)}\right)
$$

Only interior (room) points are updated; wall and outside cells remain fixed.

For each building, we compute:

- Mean interior temperature  
- Standard deviation of interior temperature  
- Percentage of interior area above 18°C  
- Percentage of interior area below 15°C  

These metrics are then used to assess how well Wall Heating works across thousands of buildings.

## Data

Each building in the dataset is identified by an ID and has two `.npy` files:

- `<id>_domain.npy`  
  - Shape: `(512, 512)`  
  - Contains the initial temperature grid:
    - 5 for load-bearing walls  
    - 25 for interior walls  
    - 0 for interior room cells  

- `<id>_interior.npy`  
  - Shape: `(512, 512)`  
  - Boolean mask:
    - 1 (True) for interior room cells  
    - 0 (False) for walls and outside the building

There is also a `building_ids.txt` listing all building IDs.

**Note:** Paths and access to the dataset are set according to the course environment; adjust `LOAD_DIR` in the scripts to match your system.