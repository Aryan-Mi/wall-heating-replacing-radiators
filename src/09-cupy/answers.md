# Task 9: CuPy -- Answers

## 9a) Timing and comparison to reference

Benchmarked on N = 10 floorplans (DTU HPC, A100 GPU, LSF batch job 28298176):

| Metric | Value |
|---|---|
| Wall time (incl. CuPy JIT compile) | 26.409 s |
| Internal time (post-warmup) | 21.260 s |
| Per-floorplan (internal) | 2.126 s |
| Reference per-floorplan (NumPy, Task 2) | 8.625 s |
| **Speedup vs reference** | **4.1×** |

The large warmup overhead (~5.1 s) reflects CuPy compiling all its built-in GPU kernels on first use. The post-warmup throughput is the relevant metric: **4.1× faster** than the NumPy reference.

## 9b) Estimated time for all floorplans

$$2.126 \text{ s} \times 4{,}571 \text{ floorplans} = 9{,}718 \text{ s} \approx 162.0 \text{ min} \approx 2.7 \text{ hours}$$

## 9c) Surprising performance observation

Despite running on a GPU, CuPy achieves only **4.1× speedup** — far less than the Numba CUDA kernel (Task 8: 10.5×) on the same hardware, and even slower than a 64-core CPU run (Task 5: 17.7×). This is surprising given that CuPy offloads all array operations to highly optimised cuBLAS/cuDNN kernels on the A100.

The bottleneck is the **per-iteration convergence check**:

```python
delta = float(cp.abs(u_new[mask_gpu] - u_gpu[1:-1, 1:-1][mask_gpu]).max())
```

The `float(...)` call transfers a single scalar from GPU memory to CPU memory, which forces a **GPU→CPU synchronisation** at every iteration. With up to 20,000 iterations per floorplan, this produces up to 20,000 pipeline stalls per building — each one requiring the GPU to flush its work queue and wait for the CPU to acknowledge the result. The actual GPU arithmetic is fast; the synchronisation overhead dominates.

The Numba CUDA solution avoids this entirely by removing the convergence check and running a fixed number of iterations without any CPU interaction mid-loop. Task 10 uses `nsys` profiling to confirm this diagnosis and explores fixes (e.g. batching iterations between checks, or removing the convergence criterion altogether).
