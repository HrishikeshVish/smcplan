#!/bin/bash
# After the 25-seed tuning set: same compute-matched pathwise config on all 100
# Push-T H=5 table seeds (finished seeds are skipped), for a paired comparison
# against GD-12 and the current JALA-EM.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
until grep -q "=== pathwise_s1 H5 DONE" sweep_emsmc/tune_s1.log 2>/dev/null; do sleep 30; done
GPU=1 CHUNK=4 H=5 SEEDS="$(seq 1 100)" NEED_MIB=6000 TAG=pathwise_s1 \
  OVR="planner.m_step=pathwise planner.m_smooth_samples=1" bash writeup/run_tune_pusht.sh
