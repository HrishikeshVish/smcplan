#!/bin/bash
# Second GPU0 worker (GPU1 is reserved for another project). Shares the claim
# queues with queue_final2.sh, so nothing is run twice:
#   1. CEM-12 remaining cells (Push-T H15, cube)
#   2. the 3 fixed-JALA-EM seeds that failed (hub 502 / stale claim)
#   3. kernel ablation on the LeWM envs only (Reacher H10, cube) per kernel --
#      ~1 GB each, so they pair with the main worker's DINO-WM runs.
cd /home/hviswan/Documents/grasp
GPU=0 NEED_MIB=4000 ENVS="pusht cube" bash writeup/run_cem12.sh >> sweep/cem12_gpu0_w2.log 2>&1
GPU=0 ENVS=pointmaze HORIZONS=5 SEEDS="58 59" CHUNK=8 NEED_MIB=10000 bash writeup/run_emsmc_final.sh >> sweep_emsmc/table/final_gpu0_w2.log 2>&1
GPU=0 ARM=pdefix SEEDS=3 NEED_MIB=4000 bash writeup/run_emsmc_cube.sh >> sweep_emsmc/table/final_gpu0_w2.log 2>&1
for K in mala rwm hmc; do
  GPU=0 KERNEL=$K ARM=kfix_$K ENVS=reacher HORIZONS=10 CHUNK=4 NEED_MIB=4000 bash writeup/run_emsmc_final.sh \
    >> sweep_emsmc/table/kfix_gpu0_w2.log 2>&1
  GPU=0 KERNEL=$K ARM=kfix_$K NEED_MIB=4000 bash writeup/run_emsmc_cube.sh >> sweep_emsmc/table/kfix_gpu0_w2.log 2>&1
done
