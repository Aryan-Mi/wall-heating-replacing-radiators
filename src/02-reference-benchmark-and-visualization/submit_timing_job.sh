#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p outputs/02-reference-benchmark-and-visualization/lsf
bsub < src/02-reference-benchmark-and-visualization/timing_job.lsf
