#!/bin/bash
# GPU0: after the running smoothed-GD (gds12) worker finishes, GRASP-12's remaining
# cells (GD-12 dropped -- gds12 replaces it), then CEM-12.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
while pgrep -f "PLANNERS=gds12" > /dev/null || pgrep -f "run_baselines_ms.sh" > /dev/null; do sleep 60; done
GPU=0 PLANNERS=grasp12 bash writeup/run_baselines_ms.sh >> sweep/baselines_ms_gpu0.log 2>&1
GPU=0 bash writeup/run_cem12.sh >> sweep/cem12_gpu0.log 2>&1
