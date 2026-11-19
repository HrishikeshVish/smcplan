#!/bin/bash
# Per GPU: 12-candidate baselines (GD-12, GRASP-12) first, then resume the kernel
# ablation (claim queues skip everything already finished).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
for g in ${GPUS:-0 1}; do
  ( GPU=$g bash writeup/run_baselines_ms.sh >> sweep/baselines_ms_gpu$g.log 2>&1
    for K in mala rwm hmc; do
      GPU=$g KERNEL=$K ARM=k_$K HORIZONS=10 NEED_MIB=12000 bash writeup/run_emsmc_table.sh \
        >> sweep_emsmc/table/kernel_worker_gpu$g.log 2>&1
      GPU=$g KERNEL=$K ARM=k_$K bash writeup/run_emsmc_cube.sh \
        >> sweep_emsmc/table/kernel_worker_gpu$g.log 2>&1
    done ) &
done
wait
