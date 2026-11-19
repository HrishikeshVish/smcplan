#!/bin/bash
# Raw-L2 on the FAST seeds the [5,120] band excluded.
#
# writeup/slow_seeds.json kept only seeds the roadmap solves in 5-120 iterations.
# That dropped 148 of 720 roadmap-solved seeds (21%) for exiting in under 5 --
# and the loss is not uniform: Point maze loses ~half its solved seeds to the
# band while Push-T H10 and Reacher H15 lose none. Since fast seeds are easy
# ones that raw L2 very likely also solves, excluding them inflates L2's measured
# failure rate, and it does so hardest in the cell carrying the headline result.
#
# These runs restore each cell's true fast/slow proportion. Raw-L2 arm only: the
# roadmap side of these seeds is already in sweep_fk/table under an identical
# configuration.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=$PWD/sweep_aft/faststratum
CK=ckpt_base_path=$PWD/checkpoints
seeds () { python3 -c "import json;d=json.load(open('writeup/fast_seeds.json'));print(' '.join(map(str,d.get('$1',[]))))"; }
run () {   # env H gl seed args...
  local env=$1 H=$2 gl=$3 s=$4; shift 4
  local OUT=$ROOT/rawl2/$env/$H/seed_$s
  [ -s "$OUT/result.txt" ] && return
  mkdir -p $OUT
  if [ "$gl" = egl ]; then export MUJOCO_GL=egl; unset DISPLAY; else unset MUJOCO_GL; export DISPLAY=:2; fi
  CUDA_VISIBLE_DEVICES=${GPU:-1} GRASP_RESULTS_DIR=$OUT \
    GRASP_FK_DUMP=$OUT/trace.npz GRASP_TRAJ_DUMP=$OUT/traj.npz \
    timeout 3600 python plan.py n_evals=1 seed=$s save_images=False \
    planner.opt_steps=150 planner.eval_every=1 "$@" \
    ++planner.v_null=objective_raw $CK > $OUT/run.log 2>&1
  if grep -q "Success rate:" $OUT/run.log; then
    local ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
    echo "$ok $(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')" > $OUT/result.txt
    echo "[fast $env $H s$s] success=$ok"
  else
    local why=OOM; grep -qi "out of memory" $OUT/run.log || why=CRASH
    echo "[fast $env $H s$s] $why -- not recorded"
  fi
}
cell () {  # env H gl args...
  local env=$1 H=$2 gl=$3; shift 3
  for s in $(seeds ${env}_${H}); do run $env $H $gl $s "$@"; done
  echo "=== CELL DONE $env $H $(date +%H:%M) ==="
}
PM="model_name=point_maze planner=fk_pointmaze goal_source=random_state planner.num_smooth_samples=1"
PT="model_name=pusht planner=fk_pusht goal_source=dset planner.num_smooth_samples=1"
WA="model_name=wall_single planner=fk_wall goal_source=random_state planner.num_smooth_samples=1"
RE="model_name=lewm_reacher planner=fk goal_source=random_state planner.num_particles=8 planner.num_smooth_samples=8"
cell pointmaze H10 none $PM goal_H=10 planner.num_particles=3
cell pointmaze H5  none $PM goal_H=5  planner.num_particles=3
cell pointmaze H15 none $PM goal_H=15 planner.num_particles=2
cell pusht     H5  none $PT goal_H=5  planner.num_particles=3
cell wall      H10 none $WA goal_H=10 planner.num_particles=3
cell wall      H15 none $WA goal_H=15 planner.num_particles=2
cell reacher   H5  egl  $RE goal_H=5
cell reacher   H10 egl  $RE goal_H=10
echo "=== FAST-STRATUM SWEEP COMPLETE $(date +%H:%M) ==="
