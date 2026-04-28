# Task 7: Numba JIT CPU -- Answers

## 7a) Timing and comparison to reference

Benchmarked on N = 10 floorplans (single core, DTU HPC `hpc` queue):

| Metric | Value |
|---|---|
| Wall time (incl. JIT compile) | 38.901 s |
| Internal time (post-warmup) | 36.432 s |
| Per-floorplan (internal) | 3.643 s |
| Reference per-floorplan (NumPy, Task 2) | 8.625 s |
| **Speedup vs reference** | **2.37×** |

The Numba JIT solution is **2.37× faster** than the NumPy reference. The one-time JIT compilation cost is small (~2.5 s, measured by comparing wall time to post-warmup internal time) and amortises quickly across many floorplans.

## 7b) Function explanation and cache access pattern

The `jacobi_numba` function is decorated with `@numba.njit`, which compiles it to native machine code ahead of any call. This eliminates the Python interpreter overhead and NumPy vectorisation overhead that limits the reference implementation.

The solver uses two explicit nested loops — outer over rows (`i`), inner over columns (`j`) — and applies the five-point stencil directly:

```python
val = 0.25 * (u[i+1, j] + u[i+1, j+2] + u[i, j+1] + u[i+2, j+1])
```

**Cache access pattern:** NumPy arrays are stored in C-contiguous (row-major) order, meaning elements within the same row are adjacent in memory. By iterating `i` in the outer loop and `j` in the inner loop, all four stencil neighbours are accessed in a sequential left-to-right sweep across rows `i`, `i+1`, and `i+2`. As `j` increments by 1 each step, each memory access moves forward by one element within its row, maximising cache-line reuse and avoiding strided or random-access patterns. The three active rows fit comfortably in L2/L3 cache together, so the cross-row accesses (`u[i, j+1]` and `u[i+2, j+1]`) also hit warm cache lines.

Convergence tracking uses a plain scalar `delta` variable updated in-loop, avoiding any NumPy reduction calls that would break out of the JIT-compiled region.

## 7c) Estimated time for all floorplans

Using the per-floorplan internal time:

$$3.643 \text{ s} \times 4{,}571 \text{ floorplans} = 16{,}653 \text{ s} \approx 277.6 \text{ min} \approx 4.6 \text{ hours}$$

This is roughly 2.4× faster than the NumPy reference estimate, but still impractical for a single serial core. Multi-core parallelism (Tasks 5/6) or GPU acceleration (Tasks 8/9) are needed to bring the total to a manageable runtime.
