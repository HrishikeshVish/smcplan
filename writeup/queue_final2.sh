#!/bin/bash
# GPU0, after the running smoothed-GD (gds12) worker finishes:
#   0. mop-up pass for gds12 seeds that crashed (torch.hub timeout) or OOMed
#   1. fixed JALA-EM (main table: 4 envs x H{5,10,15} x 100 seeds, then cube 50)
#   2. GRASP-12 remaining cells   3. CEM-12
#   4. kernel ablation for the appendix under the fixed setting (H=10 + cube; mala, rwm, hmc)
cd /home/hviswan/Documents/grasp
while pgrep -f "run_baselines_ms.sh" > /dev/null; do sleep 60; done
GPU=0 PLANNERS=gds12 ENVS="pusht cube" bash writeup/run_baselines_ms.sh >> sweep/gds12_gpu0.log 2>&1
GPU=0 NEED_MIB=12000 bash writeup/run_emsmc_final.sh >> sweep_emsmc/table/final_gpu0.log 2>&1
GPU=0 ARM=pdefix bash writeup/run_emsmc_cube.sh >> sweep_emsmc/table/final_gpu0.log 2>&1
GPU=0 PLANNERS=grasp12 bash writeup/run_baselines_ms.sh >> sweep/baselines_ms_gpu0.log 2>&1
GPU=0 bash writeup/run_cem12.sh >> sweep/cem12_gpu0.log 2>&1
for K in mala rwm hmc; do
  GPU=0 KERNEL=$K ARM=kfix_$K HORIZONS=10 NEED_MIB=12000 bash writeup/run_emsmc_final.sh \
    >> sweep_emsmc/table/kfix_gpu0.log 2>&1
  GPU=0 KERNEL=$K ARM=kfix_$K bash writeup/run_emsmc_cube.sh >> sweep_emsmc/table/kfix_gpu0.log 2>&1
done
