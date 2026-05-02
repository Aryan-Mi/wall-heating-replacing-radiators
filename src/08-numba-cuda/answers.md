# Task 8: Numba CUDA Kernel -- Answers

## 8a) Timing and comparison to reference

Benchmarked on N = 10 floorplans (DTU HPC, A100 GPU, run twice for consistency):

| Metric | Run 1 | Run 2 (used in CSV) |
|---|---|---|
| Wall time (incl. CUDA JIT compile) | 9.751 s | 9.604 s |
| Internal time (post-warmup) | 8.354 s | 8.214 s |
| Per-floorplan (internal) | 0.835 s | 0.821 s |
| Reference per-floorplan (NumPy, Task 2) | 8.625 s | 8.625 s |
| **Speedup vs reference** | **10.3×** | **10.5×** |

The two runs are highly consistent (~1.7% variation), confirming reliable measurements. The Numba CUDA kernel is **~10.4× faster** than the NumPy reference. The JIT compilation overhead is small (~1.4 s) and amortises quickly across floorplans.

## 8b) Function description

The solution consists of a CUDA kernel and a Python helper function.

**`_jacobi_kernel`** is decorated with `@cuda.jit`. The GPU launches a 2D grid of 16×16-thread blocks covering the full 512×512 interior grid. Each thread obtains its cell index via `cuda.grid(2)` and performs one Jacobi iteration: if the cell is interior (`interior_mask_u8[i, j] != 0`), it writes the five-point stencil average to `u_new[i+1, j+1]`; otherwise it copies the current value unchanged, preserving wall and fixed-boundary conditions. The mask is stored as `uint8` rather than `bool` to avoid device-array instability on some CUDA driver versions.

**`jacobi_numba_cuda`** manages two device arrays (`d_u`, `d_u_new`) initialised from the input so ghost rows are present in both buffers from the start. After each kernel launch the two references are swapped (double-buffering): the "new" values become the read source for the next iteration. This gives correct Jacobi semantics — every thread reads from the previous step's values. Kernel completion serves as a global barrier between iterations; no intra-kernel synchronisation is needed because each thread writes only to its own output cell. The loop runs for a fixed `MAX_ITER = 20,000` iterations with no early stopping, as required.

## 8c) Estimated time for all floorplans

$$0.821 \text{ s} \times 4{,}571 \text{ floorplans} = 3{,}754 \text{ s} \approx 62.6 \text{ min} \approx 1.0 \text{ hour}$$

This is a **4.4× improvement** over the Numba CPU solution (Task 7: ~277.6 min) and makes processing the full dataset in a single GPU job practical.
