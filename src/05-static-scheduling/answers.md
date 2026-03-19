# Task 5: Static Scheduling -- Answers

## 5a) Speedup Plot

![Speedup](../../outputs/05-static-scheduling/speedup.png)

| Workers | Time (s) | Speedup |
|---------|----------|---------|
| 1       | _TBD_    | 1.00    |
| 2       | _TBD_    | _TBD_   |
| 4       | _TBD_    | _TBD_   |
| 8       | _TBD_    | _TBD_   |
| 16      | _TBD_    | _TBD_   |
| 32      | _TBD_    | _TBD_   |

## 5b) Parallel Fraction (Amdahl's Law)

**Amdahl's law**
$$S(P) = 1 / ((1 - f) + f/P)$$

Rearranging to estimate f from measured speedup S at P workers:

$$f = (1 - 1/S) / (1 - 1/P)$$

Using the measurement at P = _TBD_ workers, S = _TBD_:

f = (1 - 1/_TBD_) / (1 - 1/_TBD_) = _TBD_

Approximately _TBD_% of the computation is parallelizable.

## 5c) Theoretical Maximum Speedup

$$S_{max} = 1 / (1 - f) = 1 / (1 - \text{TBD}) = \text{TBD}$$

Achieved speedup: _TBD_x at P = _TBD_ workers.
This is _TBD_% of the theoretical maximum.

Diminishing returns become visible beyond P = _TBD_ workers.

## 5d) Estimate for All Floorplans

Using the fastest configuration (P = _TBD_):

- Time for 100 floorplans: _TBD_ s
- Per-floorplan time: _TBD_ s
- Total floorplans available: _TBD_
- Estimated time for all floorplans: _TBD_ s (~_TBD_ min)
