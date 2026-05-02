#!/bin/bash
#BSUB -J wallheat_nsys_cupy
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:30
#BSUB -u s254355@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo outputs/10-nsys-cupy/%J.out
#BSUB -eo outputs/10-nsys-cupy/%J.err

set -euo pipefail
cd "$LS_SUBCWD"
mkdir -p outputs/10-nsys-cupy

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

DATA_DIR=/dtu/projects/02613_2025/data/modified_swiss_dwellings/

# 1. Profile the ORIGINAL CuPy script with nsys to confirm the sync bottleneck
echo "=== nsys profile: CuPy ORIGINAL (Task 9) ==="
nsys profile \
  --output outputs/10-nsys-cupy/nsys_original \
  --force-overwrite true \
  python src/09-cupy/simulate_cupy.py 1 "$DATA_DIR"

echo "=== nsys stats: CuPy ORIGINAL ==="
nsys stats outputs/10-nsys-cupy/nsys_original.nsys-rep

# 2. Profile the OPTIMIZED script to confirm syncs are eliminated
echo "=== nsys profile: CuPy OPTIMIZED (Task 10) ==="
nsys profile \
  --output outputs/10-nsys-cupy/nsys_optimized \
  --force-overwrite true \
  python src/10-nsys-cupy/simulate_cupy_optimized.py 1 "$DATA_DIR"

echo "=== nsys stats: CuPy OPTIMIZED ==="
nsys stats outputs/10-nsys-cupy/nsys_optimized.nsys-rep

# 3. Benchmark: optimized vs original on N=10 floorplans
echo "=== Timing benchmark: N=10 ==="
python src/10-nsys-cupy/run_task_10.py \
  --subset 10 \
  --data-dir "$DATA_DIR" \
  --output-dir outputs/10-nsys-cupy \
  --reference-csv outputs/09-cupy/timing_summary.csv
