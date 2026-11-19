#!/bin/bash
# Compute-matched pathwise JALA-EM on the Push-T seeds where GD-12 FAILED.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
O="planner.m_step=pathwise planner.m_smooth_samples=1"
GPU=1 CHUNK=4 H=5 SEEDS="2 47 58 73" NEED_MIB=6000 TAG=pathwise_s1 OVR="$O" bash writeup/run_tune_pusht.sh
GPU=1 CHUNK=2 H=10 SEEDS="2 4 5 6 8 9 10 13 14 15 17 18 20 21 22 23 26 27 28 31 33 34 37 38 39 40 41 42 43 49 52 53 57 58 59 62 63 64 65 67 69 70 73 74 75 80 81 84 88 90 93 94 99" NEED_MIB=6000 TAG=pathwise_s1 OVR="$O" bash writeup/run_tune_pusht.sh
