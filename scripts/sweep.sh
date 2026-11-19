#!/bin/bash
# Launch a GPU sweep. Each cell of the paper is 100 seeds of one plan.py call,
# so these run for days, not minutes. Every launcher claims each seed with a lock
# directory, so several workers share a sweep and an interrupted run resumes by
# skipping any seed whose result.txt exists.
#
#   bash scripts/sweep.sh table1          # ours, 4 envs x H in {5,10,15}   -> sweep_emsmc/table
#   bash scripts/sweep.sh baselines       # GD, GRASP, CEM, MPPI at 12 particles -> sweep/
#   bash scripts/sweep.sh cube            # cube under LeFlow's protocol    -> sweep_cube_leflow/
#   bash scripts/sweep.sh ablations       # appendix ablations and sensitivity
#   bash scripts/sweep.sh kernels         # MALA, RWM, HMC against JALA
#   bash scripts/sweep.sh pde-baselines   # the roadmap cost added to GD, CEM, MPPI
#
# GPUS picks the cards (default "0 1"); ENVS narrows table1. MUJOCO_GL is set per
# environment inside the launchers: egl for Reacher and Cube, unset elsewhere.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_ROOT"
[ $# -ge 1 ] || { sed -n 2,16p "$0"; exit 1; }
case $1 in
  table1)        bash writeup/launch_emsmc_table.sh ;;
  baselines)     bash writeup/launch_baselines_then_ablation.sh ;;
  cube)          bash writeup/run_cube_leflow.sh ;;
  ablations)     bash writeup/run_ablations.sh ;;
  kernels)       bash writeup/run_emsmc_kernels.sh ;;
  pde-baselines) bash writeup/run_pde_baselines.sh ;;
  *) echo "unknown sweep $1"; exit 1 ;;
esac
