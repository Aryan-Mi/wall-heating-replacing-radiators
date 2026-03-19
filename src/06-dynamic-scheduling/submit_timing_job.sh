#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p outputs/06-dynamic-scheduling/lsf
bsub < src/06-dynamic-scheduling/timing_job.lsf
