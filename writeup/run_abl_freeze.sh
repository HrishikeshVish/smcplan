#!/bin/bash
# Tilt-freeze ablation: freeze_tilt=true (shipped) vs false.
#
# Once beta reaches beta_max the target sequence has ended and the planner stops
# resampling: the -beta*dV drift correction is exact only for a pi0-invariant
# kernel, and this kernel is a non-stationary optimizer that is always
# descending L, so post-anneal reweighting fights that descent. The code comment
# records 145-resample churn breaking seed 9 when this was not done.
#
# This sweep tests that claim rather than asserting it. Point maze H=10, the same
# 20-seed stratified subset as the other knob sweeps, so the numbers sit
# alongside them. The prediction is that freeze_tilt=false is WORSE; if it is
# not, the paragraph in the appendix needs rewriting, not the figure.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset MUJOCO_GL; export DISPLAY=:2
SEEDS="1 6 7 14 20 25 33 38 44 45 54 60 65 68 73 79 84 93 99 100"
ROOT=$PWD/sweep_aft/abl_freeze
for v in false true; do
  for s in $SEEDS; do
    OUT=$ROOT/$v/seed_$s
    [ -s "$OUT/result.txt" ] && continue
    mkdir -p $OUT
    CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT GRASP_TRAJ_DUMP=$OUT/traj.npz \
      timeout 3600 python plan.py n_evals=1 seed=$s save_images=False \
      model_name=point_maze planner=fk_pointmaze goal_H=10 goal_source=random_state \
      planner.opt_steps=150 planner.eval_every=1 planner.num_particles=3 \
      planner.num_smooth_samples=1 ++planner.freeze_tilt=$v \
      ckpt_base_path=$PWD/checkpoints > $OUT/run.log 2>&1
    if grep -q "Success rate:" $OUT/run.log; then
      ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
      echo "$ok $(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')" > $OUT/result.txt
      echo "[freeze=$v s$s] success=$ok"
    else
      why=OOM; grep -qi "out of memory" $OUT/run.log || why=CRASH
      echo "[freeze=$v s$s] $why -- not recorded"
    fi
  done
  echo "=== ARM DONE freeze=$v $(date +%H:%M) ==="
done
echo "=== FREEZE SWEEP COMPLETE $(date +%H:%M) ==="
