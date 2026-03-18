# Task 2 and 3 - Reference Benchmark + Simulation Visualization

This folder contains a reimplementation of task 2 and task 3 based on the reference solver logic in `simulate.py`.

## What this does

The script [run_tasks_2_3.py](run_tasks_2_3.py) performs the following steps:

1. Discovers floorplans from `data/` by matching `<id>_domain.npy` and `<id>_interior.npy`.
2. Loads a subset (default: 10 floorplans).
3. Runs the **reference Jacobi implementation** with:
   - `max_iter = 20000`
   - `atol = 1e-4`
4. Times the simulation as a **batch job** repeated multiple times (default: 3 runs) for reliable timing.
5. Estimates total runtime for all available floorplans using:

   estimated_all_seconds = mean_batch_seconds / subset_size * total_floorplans

6. Saves outputs:
   - `output/timing_summary.csv`: timing and estimate summary
   - `output/stats_subset.csv`: simulation summary stats per floorplan
   - `output/plots/*_simulation.png`: visualization for a few floorplans

## Run

From repository root:

```bash
python3 src/02-reference-benchmark-and-visualization/run_tasks_2_3.py --subset 10 --repeats 3 --plot-count 3
```
    
Optional arguments:

- `--data-dir` (default: `data`)
- `--output-dir` (default: `outputs/02-reference-benchmark-and-visualization`)
- `--max-iter` (default: `20000`)
- `--atol` (default: `1e-4`)

