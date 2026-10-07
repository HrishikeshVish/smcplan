#!/bin/bash
# Second GPU0 worker, alongside the smoothed-GD queue: fixed JALA-EM on the LeWM
# envs only (single-token latents, small rollouts) -- Reacher, then Cube. Gated at
# 6 GB free with the rollout chunk halved (exact; memory only). Shares the claim
# queues with queue_final2.sh, which skips anything finished here.
cd /home/hviswan/Documents/grasp
GPU=0 ENVS=reacher CHUNK=4 NEED_MIB=6000 bash writeup/run_emsmc_final.sh >> sweep_emsmc/table/final_gpu0_w2.log 2>&1
GPU=0 ARM=pdefix NEED_MIB=6000 bash writeup/run_emsmc_cube.sh >> sweep_emsmc/table/final_gpu0_w2.log 2>&1
