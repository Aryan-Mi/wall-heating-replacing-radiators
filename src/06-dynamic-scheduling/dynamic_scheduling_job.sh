#!/bin/bash
#BSUB -J wallheat_dynamic
#BSUB -q hpc
#BSUB -n 64
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 01:00
#BSUB -oo outputs/06-dynamic-scheduling/%J.out
#BSUB -eo outputs/06-dynamic-scheduling/%J.err

set -euo pipefail

cd "$LS_SUBCWD"
mkdir -p outputs/06-dynamic-scheduling

python src/06-dynamic-scheduling/run_task_6.py \
  --subset 100 \
  --data-dir /dtu/projects/02613_2025/data/modified_swiss_dwellings/ \
  --output-dir outputs/06-dynamic-scheduling \
  --static-csv outputs/05-static-scheduling/timing_results.csv
