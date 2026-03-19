#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p outputs/05-static-scheduling/lsf
bsub < src/05-static-scheduling/timing_job.lsf
