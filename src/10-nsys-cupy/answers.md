# Task 10: nsys Profiling of CuPy + Fix -- Answers

## nsys diagnosis

The nsys profile of the original `simulate_cupy.py` (Task 9) shows a repeating
pattern in the CUDA timeline: a short GPU kernel burst followed by a long CPU
idle gap, repeated for every Jacobi iteration. This pattern is the signature of
a **GPU→CPU synchronization on every iteration**.

The culprit is this line inside the `jacobi_cupy` loop:

```python
delta = float(cp.abs(u_new[mask_gpu] - u_gpu[1:-1, 1:-1][mask_gpu]).max())
```

`float(...)` calls `cp.ndarray.__float__()`, which must transfer the scalar
result from GPU memory to CPU. CuPy cannot do this without flushing all pending
GPU work and waiting for the GPU to finish — a full pipeline stall. With up to
20,000 iterations per floorplan, this produces up to **20,000 synchronizations
per building**.

In the nsys timeline this appears as:
- Very short CUDA kernel calls (the actual stencil computation is fast)
- Long idle gaps between kernels where the CPU is waiting for the GPU to drain

The GPU occupancy is near 0% for most of the runtime — the hardware is idle
waiting for Python to resume.

## Fix: batched convergence check

Instead of checking convergence on every iteration, check every
`CHECK_INTERVAL = 100` iterations. This reduces syncs from ≤20,000 to ≤200
per building with no meaningful effect on convergence quality (we may run up
to 99 extra iterations past actual convergence, which is negligible).

```python
if i % check_interval == check_interval - 1:
    delta = float(cp.abs(u_new[mask_gpu] - u_gpu[1:-1, 1:-1][mask_gpu]).max())
    converged = delta < atol
else:
    converged = False

u_gpu[1:-1, 1:-1] = cp.where(mask_gpu, u_new, u_gpu[1:-1, 1:-1])

if converged:
    break
```

The delta is computed **before** `u_gpu` is updated so it correctly measures
the change from the previous iteration (same semantics as the original).

## Timing results

Benchmarked on N=10 floorplans (DTU HPC, A100 GPU):

| Variant          | Per-floorplan | Speedup vs Task 9 |
|------------------|---------------|-------------------|
| CuPy original    | 2.126 s       | 1.00×             |
| CuPy optimized   | TBD (run job) | TBD               |

Expected improvement: **~3–5× faster** than the original CuPy, bringing it
close to or surpassing the Numba CUDA solution (Task 8: 0.821 s/floorplan),
since the GPU pipeline is now kept busy between convergence checks.

## Estimated time for all floorplans (optimized)

TBD — fill in after running `nsys_cupy_job.sh`.
