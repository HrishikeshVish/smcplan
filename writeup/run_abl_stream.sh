#!/bin/bash
# One ablation stream, launched alongside the others on GPU 1 (the card sits at
# ~35% utilisation with ~1GB per job, so several streams fit). Point maze H=10 on
# the 20-seed stratified subset from writeup/ablation_subset_seeds.json, whose
# success rate reproduces the full 100-seed cell to within two points.
#
#   $1  stream name   $2  hydra override applied to every run in the stream
#   remaining args: the arm values to sweep
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset MUJOCO_GL; export DISPLAY=:2
NAME=$1; KNOB=$2; shift 2
SEEDS="1 6 7 14 20 25 33 38 44 45 54 60 65 68 73 79 84 93 99 100"
ROOT=$PWD/sweep_aft/abl_$NAME
for v in "$@"; do
  for s in $SEEDS; do
    OUT=$ROOT/${v}/seed_$s
    [ -s "$OUT/result.txt" ] && continue
    mkdir -p $OUT
    CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT GRASP_TRAJ_DUMP=$OUT/traj.npz \
      timeout 3600 python plan.py n_evals=1 seed=$s save_images=False \
      model_name=point_maze planner=fk_pointmaze goal_H=10 goal_source=random_state \
      planner.opt_steps=150 planner.eval_every=1 planner.num_particles=3 \
      planner.num_smooth_samples=1 ++planner.$KNOB=$v \
      ckpt_base_path=$PWD/checkpoints > $OUT/run.log 2>&1
    # Only record a result if the run actually reported one. Writing result.txt
    # unconditionally made OOM crashes look like recorded failures, and the
    # resume guard above then skipped them on the next pass.
    if grep -q "Success rate:" $OUT/run.log; then
      ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
      echo "$ok $(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')" > $OUT/result.txt
      echo "[$NAME $KNOB=$v s$s] success=$ok"
    else
      why=OOM; grep -qi "out of memory" $OUT/run.log || why=CRASH
      echo "[$NAME $KNOB=$v s$s] $why -- no result recorded"
    fi
  done
done
echo "=== STREAM $NAME DONE ==="
