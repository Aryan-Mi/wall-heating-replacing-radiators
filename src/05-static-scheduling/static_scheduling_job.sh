#!/bin/bash
#BSUB -J wallheat_static
#BSUB -q hpc
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 01:00
#BSUB -oo outputs/05-static-scheduling/%J.out
#BSUB -eo outputs/05-static-scheduling/%J.err
#BSUB -R "select[model==XeonGold6226R]"

set -euo pipefail

cd "$LS_SUBCWD"
mkdir -p outputs/05-static-scheduling

# Init Python environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python src/05-static-scheduling/run_task_5.py \
  --subset 100 \
  --data-dir /dtu/projects/02613_2025/data/modified_swiss_dwellings/ \
  --output-dir outputs/05-static-scheduling
