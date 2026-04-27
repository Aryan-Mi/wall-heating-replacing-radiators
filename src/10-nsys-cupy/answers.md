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

| Variant                   | Per-floorplan | Speedup vs Task 9 |
|---------------------------|---------------|-------------------|
| CuPy original (Task 9)    | 2.126 s       | 1.00×             |
| CuPy optimized (Task 10)  | 0.828 s       | **2.57×**         |
| Numba CUDA (Task 8)       | 0.821 s       | —                 |

The optimized CuPy is **2.57× faster** than the original, and matches the
Numba CUDA kernel (Task 8: 0.821 s) almost exactly. This confirms the nsys
diagnosis: once the per-iteration sync is eliminated, CuPy's high-level
array operations perform as well as a hand-written CUDA kernel for this workload.

Note: the output values differ slightly between original and optimized
(e.g. mean 14.0123 vs 14.0131). This is expected — the batched check may run
up to 99 additional iterations past the convergence point, refining the solution
marginally further.

## Estimated time for all floorplans (optimized)

$$0.828 \text{ s} \times 4{,}571 \text{ floorplans} = 3{,}785 \text{ s} \approx 63.1 \text{ min}$$

This matches the Numba CUDA estimate from Task 8 (~62.6 min), confirming that
both GPU implementations are compute-bound rather than memory-bound at this
problem size.
