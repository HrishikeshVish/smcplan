#!/bin/bash
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
S="4 8 11 15 17 18 20 21 22 25 28 33 36 37 39 45 55 57 60 62 65 69 72 81 99"
GPU=1 CHUNK=4 H=5 SEEDS="$S" NEED_MIB=6000 TAG=pathwise OVR="planner.m_step=pathwise" bash writeup/run_tune_pusht.sh
GPU=1 CHUNK=4 H=5 SEEDS="$S" NEED_MIB=6000 TAG=beta10 OVR="planner.beta_schedule=geometric planner.beta_init=1.0 planner.beta_max=10.0" bash writeup/run_tune_pusht.sh
GPU=1 CHUNK=4 H=5 SEEDS="$S" NEED_MIB=6000 TAG=beta100 OVR="planner.beta_schedule=geometric planner.beta_init=1.0 planner.beta_max=100.0" bash writeup/run_tune_pusht.sh
