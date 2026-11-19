#!/bin/bash
# Compute-matched pathwise M-step WITH the calibrated Push-T PDE cost (k=64, c=10,
# t=c/median(D), trajectory-weighted Vhat, auto lambda_V, surrogate 3000 steps).
# Order: GD-12 failures at H5, then the 25 GD-12-only seeds at H5, then GD-12
# failures at H10.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
P=/tmp/claude-1002/-home-hviswan-Documents-grasp-langevin/665af950-b32b-4966-b53a-0925538356b0/scratchpad
O="planner.m_step=pathwise planner.m_smooth_samples=1 ++planner.v_traj=all ++planner.v_base=2.0 ++planner.heat_t_rule=degree ++planner.heat_c=10.0 planner.graph_k_neighbors=64 ++planner.surrogate_steps=3000"
export GPU=0 NEED_MIB=6000 TAG=pde_pathwise_s1 LAMBDA_V=auto OVR="$O"
CHUNK=4 H=5 SEEDS="$(cat $P/gd12fail_H5.txt)" bash writeup/run_tune_pusht.sh
CHUNK=4 H=5 SEEDS="$(cat $P/tune_pusht_H5.txt)" bash writeup/run_tune_pusht.sh
CHUNK=2 H=10 SEEDS="$(cat $P/gd12fail_H10.txt)" bash writeup/run_tune_pusht.sh
