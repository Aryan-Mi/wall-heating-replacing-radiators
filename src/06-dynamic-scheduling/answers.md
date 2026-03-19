# Task 6: Dynamic Scheduling -- Answers

## 6a) Did it get faster? By how much?

| Workers | Static (s) | Dynamic (s) | Difference |
|---------|-----------|-------------|------------|
| 1       | _TBD_     | _TBD_       | baseline   |
| 2       | _TBD_     | _TBD_       | _TBD_      |
| 4       | _TBD_     | _TBD_       | _TBD_      |
| 8       | _TBD_     | _TBD_       | _TBD_      |
| 16      | _TBD_     | _TBD_       | _TBD_      |
| 32      | _TBD_     | _TBD_       | _TBD_      |

_TBD_: Analysis of whether dynamic scheduling was faster and by how much.

## 6b) Did the speed-up improve or worsen?

![Speedup comparison](../../outputs/06-dynamic-scheduling/speedup.png)

| Workers | Static Speedup | Dynamic Speedup |
|---------|---------------|-----------------|
| 1       | 1.00          | 1.00            |
| 2       | _TBD_         | _TBD_           |
| 4       | _TBD_         | _TBD_           |
| 8       | _TBD_         | _TBD_           |
| 16      | _TBD_         | _TBD_           |
| 32      | _TBD_         | _TBD_           |

_TBD_: Analysis of speedup comparison. Dynamic scheduling should show better speedup at higher worker counts because it handles load imbalance from varying convergence times -- workers that finish a fast floorplan immediately pick up the next one, rather than waiting idle while another worker processes a slow floorplan.
