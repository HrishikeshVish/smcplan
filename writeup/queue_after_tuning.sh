#!/bin/bash
# GPU0, after the PDE tuning finishes: smoothed GD (gds12) on all cells, then the
# remaining GD-12/GRASP-12 cells, then CEM-12.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
while pgrep -f "tune_pde.sh" > /dev/null; do sleep 60; done
GPU=0 PLANNERS=gds12 bash writeup/run_baselines_ms.sh >> sweep/gds12_gpu0.log 2>&1
GPU=0 bash writeup/run_baselines_ms.sh >> sweep/baselines_ms_gpu0.log 2>&1
GPU=0 bash writeup/run_cem12.sh >> sweep/cem12_gpu0.log 2>&1
