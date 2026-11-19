#!/bin/bash
# Push-T and Wall expansion first (they were level 19-19 and 10-10, so more seeds
# are needed there to say anything), then the remaining ablations.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
bash writeup/run_rawl2.sh
echo "=== rawl2 expansion done $(date +%H:%M) ==="
bash writeup/run_abl_stream.sh ess    ess_target_frac  0.0 0.25 0.75
echo "=== ess done $(date +%H:%M) ==="
bash writeup/run_abl_stream.sh budget opt_steps        25 50 100 300
echo "=== QUEUE COMPLETE $(date +%H:%M) ==="
