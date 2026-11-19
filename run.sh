#!/usr/bin/env bash
# One planning run, at the protocol of the paper's main table.
#
#   ./run.sh <env> <planner> <horizon> [n_evals] [extra hydra overrides...]
#
#   env       reacher | pointmaze | wall | pusht | cube
#   planner   ours | ours-l2 | gd | grasp | cem | mppi
#   horizon   5 | 10 | 15   (cube: 5)
#   n_evals   number of trials, default 1
#
# Every planner gets 12 candidate action sequences and 150 iterations, the
# matched search budget of Table 1. Three things are easy to get wrong by hand
# and are set here for you:
#
#   MUJOCO_GL   egl for the LeWM environments, and UNSET for the DINO-WM ones,
#               where egl renders silently wrong frames.
#   goal_source differs per environment: a sampled state, or a state from the
#               dataset a fixed number of steps ahead.
#   eval_chunk  caps peak memory. The DINO-WM rollout retained for autograd
#               grows superlinearly with the horizon and runs a 16 GB card out
#               of memory at H=15 without it. Chunking is exact, since the rows
#               are independent.
#
# Anything after n_evals is passed to plan.py untouched, so
#   ./run.sh pointmaze ours 10 1 planner.kernel=mala
# swaps the propagation kernel.
set -euo pipefail

usage() { sed -n '2,27p' "$0" | sed 's/^# \?//'; exit "${1:-0}"; }
[ $# -ge 3 ] || usage 1
case "${1:-}" in -h|--help|help) usage;; esac

ENV=$1; PLANNER=$2; H=$3; N_EVALS=${4:-1}; shift $(( $# >= 4 ? 4 : 3 ))

case $ENV in
  reacher)   MODEL=lewm_reacher; OURS=emsmc_reacher;   GOAL=random_state; GL=egl;;
  pointmaze) MODEL=point_maze;   OURS=emsmc_pointmaze; GOAL=random_state; GL=none;;
  wall)      MODEL=wall_single;  OURS=emsmc_wall;      GOAL=random_state; GL=none;;
  pusht)     MODEL=pusht;        OURS=emsmc_pusht;     GOAL=dset;         GL=none;;
  cube)      MODEL=lewm_cube;    OURS=emsmc_cube;      GOAL=dset;         GL=egl;;
  *) echo "unknown env: $ENV" >&2; usage 1;;
esac

# Rollout chunk. Only the DINO-WM environments need a smaller one at depth.
case $ENV:$H in
  pointmaze:10|pusht:10) CHUNK=4;;
  pointmaze:15|pusht:15) CHUNK=2;;
  *) CHUNK=8;;
esac
CHUNK=${CHUNK_OVERRIDE:-$CHUNK}

# 150 iterations and 12 candidates everywhere. The flag that carries the
# candidate count is the one each planner's own config uses.
case $PLANNER in
  ours)    ARGS="planner=$OURS planner.num_particles=12 planner.opt_steps=150 planner.eval_every=1 planner.lambda_v=auto planner.eval_chunk=$CHUNK";;
  ours-l2) ARGS="planner=$OURS planner.num_particles=12 planner.opt_steps=150 planner.eval_every=1 planner.lambda_v=0.0 planner.eval_chunk=$CHUNK";;
  gd)      GDP=gd; [ "$ENV" = cube ] && GDP=gd_cube
           ARGS="planner=$GDP planner.opt_steps=150 planner.eval_every=1 planner.hjb_tau=0.0 planner.num_samples=12 planner.var_actions=1.0 ++planner.num_candidates=1 ++planner.cand_chunk=$CHUNK";;
  grasp)   ARGS="planner=grasp planner.opt_steps=150 planner.eval_every=1 ++planner.num_parallel_trajectories=12";;
  cem)     ARGS="planner=cem planner.opt_steps=150 planner.num_samples=12 planner.topk=3 planner.eval_every=1";;
  mppi)    ARGS="planner=mppi planner.opt_steps=150 planner.num_samples=12 planner.eval_every=1";;
  *) echo "unknown planner: $PLANNER" >&2; usage 1;;
esac

if [ "$GL" = egl ]; then export MUJOCO_GL=egl; unset DISPLAY || true
else unset MUJOCO_GL || true; export DISPLAY=${DISPLAY:-:0}; fi

set -x
exec python plan.py model_name="$MODEL" goal_H="$H" goal_source="$GOAL" \
  n_evals="$N_EVALS" save_images=False $ARGS "$@"
