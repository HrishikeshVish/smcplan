#!/bin/bash
# Round 2: compute-matched pathwise M-step (S=1 reuses the kernel's cached grad C;
# no extra rollouts), queued after round 1 on GPU1.
cd /home/hviswan/Documents/grasp
until grep -q "=== beta100 H5 DONE" sweep_emsmc/tune_r1.log 2>/dev/null; do sleep 60; done
S="4 8 11 15 17 18 20 21 22 25 28 33 36 37 39 45 55 57 60 62 65 69 72 81 99"
GPU=1 CHUNK=4 H=5 SEEDS="$S" NEED_MIB=6000 TAG=pathwise_s1 OVR="planner.m_step=pathwise planner.m_smooth_samples=1" bash writeup/run_tune_pusht.sh
