#!/bin/bash
#BSUB -J wallheat_cupy
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:20
#BSUB -u s254355@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo outputs/09-cupy/%J.out
#BSUB -eo outputs/09-cupy/%J.err

set -eo pipefail
cd "$LS_SUBCWD"
mkdir -p outputs/09-cupy

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python src/09-cupy/run_task_9.py \
  --subset 10 \
  --data-dir /dtu/projects/02613_2025/data/modified_swiss_dwellings/ \
  --output-dir outputs/09-cupy \
  --reference-csv outputs/02-reference-benchmark-and-visualization/timing_summary.csv
