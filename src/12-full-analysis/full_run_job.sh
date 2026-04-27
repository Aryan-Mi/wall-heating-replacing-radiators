#!/bin/bash
#BSUB -J wallheat_full_run
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 02:00
#BSUB -u s254355@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo outputs/12-full-analysis/%J.out
#BSUB -eo outputs/12-full-analysis/%J.err

set -euo pipefail
cd "$LS_SUBCWD"
mkdir -p outputs/12-full-analysis

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

DATA_DIR=/dtu/projects/02613_2025/data/modified_swiss_dwellings/
RESULTS_CSV=outputs/12-full-analysis/all_results.csv

# Run optimized CuPy on all 4571 buildings and save CSV
python src/10-nsys-cupy/simulate_cupy_optimized.py 4571 "$DATA_DIR" > "$RESULTS_CSV"

# Analyze the results
python src/12-full-analysis/analyze_results.py "$RESULTS_CSV" \
  --output-dir outputs/12-full-analysis
