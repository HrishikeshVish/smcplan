#!/bin/bash
# Roadmap vs raw-L2 on seeds chosen WITHOUT reference to either arm's outcome.
#
# Why this exists: writeup/slow_seeds.json was built by filtering a prior FK
# sweep to seeds the roadmap SOLVES in 5-120 iterations. On the deterministic
# envs the roadmap therefore re-succeeds by construction -- 0 failures in 232
# non-Reacher pairs -- so the discordant count from that sweep is one-sided and
# its absolute success rates are meaningless. Reacher was the only env where the
# filter did not bind (its goals do not reproduce), and there the arms tie 12-12.
# This sweep removes the filter: plain seeds 1..N, nothing conditioned.
#
# Both arms run fresh rather than pairing against sweep_fk/table, because the
# particle count that sweep used at H15 is not recoverable from its logs (the
# PARTICLES override is recorded nowhere), and a silent config mismatch would be
# the same class of error this sweep exists to fix.
#
# Cells: pusht H15 dropped -- the roadmap solves 5/100 unselected seeds there, so
# nearly every pair would be both-fail and carry no information.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=$PWD/sweep_aft/unbiased
CK=ckpt_base_path=$PWD/checkpoints
N=${N:-40}

run () {   # env H arm gl seed args...
  local env=$1 H=$2 arm=$3 gl=$4 s=$5; shift 5
  local OUT=$ROOT/$arm/$env/H$H/seed_$s
  [ -s "$OUT/result.txt" ] && return
  mkdir -p $OUT
  if [ "$gl" = egl ]; then export MUJOCO_GL=egl; unset DISPLAY; else unset MUJOCO_GL; export DISPLAY=:2; fi
  local A=""; [ "$arm" = rawl2 ] && A="++planner.v_null=objective_raw"
  CUDA_VISIBLE_DEVICES=${GPU:-1} GRASP_RESULTS_DIR=$OUT \
    GRASP_FK_DUMP=$OUT/trace.npz GRASP_TRAJ_DUMP=$OUT/traj.npz \
    timeout 3600 python plan.py n_evals=1 seed=$s save_images=False \
    planner.opt_steps=150 planner.eval_every=1 "$@" $A $CK > $OUT/run.log 2>&1
  if grep -q "Success rate:" $OUT/run.log; then
    local ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
    echo "$ok $(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')" > $OUT/result.txt
    echo "[$arm $env H$H s$s] success=$ok"
  else
    local why=OOM; grep -qi "out of memory" $OUT/run.log || why=CRASH
    echo "[$arm $env H$H s$s] $why -- not recorded"
  fi
}
np () { [ "$1" = 15 ] && echo 2 || echo 3; }
cell () {  # env H gl args...
  local env=$1 H=$2 gl=$3; shift 3
  for s in $(seq 1 $N); do
    for arm in roadmap rawl2; do run $env $H $arm $gl $s "$@"; done
  done
  echo "=== CELL DONE $env H$H $(date +%H:%M) ==="
}
for H in 10 5 15; do
  cell pointmaze $H none model_name=point_maze planner=fk_pointmaze goal_H=$H \
    goal_source=random_state planner.num_particles=$(np $H) planner.num_smooth_samples=1
  [ "$H" != 15 ] && cell pusht $H none model_name=pusht planner=fk_pusht goal_H=$H \
    goal_source=dset planner.num_particles=$(np $H) planner.num_smooth_samples=1
  cell wall $H none model_name=wall_single planner=fk_wall goal_H=$H \
    goal_source=random_state planner.num_particles=$(np $H) planner.num_smooth_samples=1
  cell reacher $H egl model_name=lewm_reacher planner=fk goal_H=$H \
    goal_source=random_state planner.num_particles=8 planner.num_smooth_samples=8
done
echo "=== UNBIASED SWEEP COMPLETE $(date +%H:%M) ==="
