#!/bin/bash
# Per GPU: wait for its GD-12/GRASP-12 worker to finish, then CEM-12, then the
# kernel ablation (replaces the ablation step of launch_baselines_then_ablation.sh).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
for g in 0 1; do
  ( until grep -q "BASELINES WORKER DONE" sweep/baselines_ms_gpu$g.log 2>/dev/null; do sleep 60; done
    GPU=$g bash writeup/run_cem12.sh >> sweep/cem12_gpu$g.log 2>&1
    for K in mala rwm hmc; do
      GPU=$g KERNEL=$K ARM=k_$K HORIZONS=10 NEED_MIB=12000 bash writeup/run_emsmc_table.sh \
        >> sweep_emsmc/table/kernel_worker_gpu$g.log 2>&1
      GPU=$g KERNEL=$K ARM=k_$K bash writeup/run_emsmc_cube.sh \
        >> sweep_emsmc/table/kernel_worker_gpu$g.log 2>&1
    done ) &
done
wait
