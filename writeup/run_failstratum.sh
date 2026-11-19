#!/bin/bash
# Raw-L2 runs on seeds where the ROADMAP FAILS.
#
# Purpose: the existing sweep (sweep_aft/rawl2) drew seeds from slow_seeds.json,
# which filtered to seeds the roadmap SOLVES, so it holds only the
# roadmap-succeeds stratum. These runs fill the complementary stratum so each
# env x horizon cell can be composed at 25 seeds with the same roadmap
# success/failure split as the main table (writeup/table1_iclr.tex).
#
# Seeds come from writeup/fail_seeds.json -- seeds the main-table sweep records
# as roadmap FAILURES, excluding any seed already present in sweep_aft/rawl2.
# Only the raw-L2 arm runs: the roadmap side of these seeds already exists in
# sweep_fk/table (and sweep_cube_leflow/fk for cube).
#
# CONFIG NOTE: H15 pairs against table runs whose particle count is not recorded
# in their logs. np=2 is used here to match run_rawl2.sh, and the H15 cells are
# run LAST so the H5/H10 cells land first if the H15 pairing is later found to
# mismatch.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=$PWD/sweep_aft/failstratum
CK=ckpt_base_path=$PWD/checkpoints

seeds () { python3 -c "import json;print(' '.join(map(str,json.load(open('writeup/fail_seeds.json'))['$1'][:$2])))"; }

run () {   # env H gl seed args...
  local env=$1 H=$2 gl=$3 s=$4; shift 4
  local OUT=$ROOT/rawl2/$env/$H/seed_$s   # $H already carries the "H" prefix
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
    echo "[rawl2-fail $env H$H s$s] success=$ok"
  else
    local why=OOM; grep -qi "out of memory" $OUT/run.log || why=CRASH
    echo "[rawl2-fail $env H$H s$s] $why -- not recorded"
  fi
}
cell () {  # env H n gl args...
  local env=$1 H=$2 n=$3 gl=$4; shift 4
  for s in $(seeds ${env}_${H} $n); do run $env $H $gl $s "$@"; done
  echo "=== CELL DONE $env $H  ($n seeds)  $(date +%H:%M) ==="
}
PM="model_name=point_maze planner=fk_pointmaze goal_source=random_state planner.num_smooth_samples=1"
PT="model_name=pusht planner=fk_pusht goal_source=dset planner.num_smooth_samples=1"
WA="model_name=wall_single planner=fk_wall goal_source=random_state planner.num_smooth_samples=1"
RE="model_name=lewm_reacher planner=fk goal_source=random_state planner.num_particles=8 planner.num_smooth_samples=8"

cell pointmaze H10  5 none $PM goal_H=10 planner.num_particles=3
cell pusht     H10 18 none $PT goal_H=10 planner.num_particles=3
cell wall      H10  9 none $WA goal_H=10 planner.num_particles=3
cell pointmaze H5   6 none $PM goal_H=5  planner.num_particles=3
cell pusht     H5   8 none $PT goal_H=5  planner.num_particles=3
cell wall      H5  11 none $WA goal_H=5  planner.num_particles=3
cell cube      H5   3 egl  model_name=lewm_cube planner=fk_cube goal_H=5 goal_source=dset planner.beta_max=10.0
cell pointmaze H15 10 none $PM goal_H=15 planner.num_particles=2
cell wall      H15 15 none $WA goal_H=15 planner.num_particles=2
cell pusht     H15 24 none $PT goal_H=15 planner.num_particles=2
cell reacher   H15  3 egl  $RE goal_H=15
echo "=== FAIL-STRATUM SWEEP COMPLETE $(date +%H:%M) ==="
