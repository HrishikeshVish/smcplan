#!/bin/bash
# Wait for the GPU1 reacher resume to finish, then run pointmaze + pusht on GPU1.
cd /home/hviswan/Documents/grasp
while kill -0 3026315 2>/dev/null; do sleep 60; done
echo "=== CHAIN: reacher done, starting pointmaze+pusht ($(date +%T)) ==="
GPU=1 bash writeup/.run_sweep_gpu1.sh pointmaze pusht
echo "=== CHAIN: ALL DONE ($(date +%T)) ==="
