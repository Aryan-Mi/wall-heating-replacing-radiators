#!/bin/bash
#BSUB -J wallheat_numba_jit_cpu
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:20
#BSUB -oo outputs/07-numba-jit-cpu/%J.out
#BSUB -eo outputs/07-numba-jit-cpu/%J.err

set -euo pipefail
cd "$LS_SUBCWD"
mkdir -p outputs/07-numba-jit-cpu

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python src/07-numba-jit-cpu/run_task_7.py \
  --subset 10 \
  --data-dir /dtu/projects/02613_2025/data/modified_swiss_dwellings/ \
  --output-dir outputs/07-numba-jit-cpu \
  --reference-csv outputs/02-reference-benchmark-and-visualization/timing_summary.csv
