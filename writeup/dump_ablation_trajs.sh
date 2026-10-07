#!/bin/bash
# Re-run all 38x2 ablation runs with GRASP_TRAJ_DUMP to get the executed (x,y)
# path, for aggregate trajectory GEOMETRY (path length, wandering, efficiency).
#
# Needed because nothing in the existing logs carries the path: seed_success_log
# field 3 is mean_visual_dist (IMAGE distance), not position, and logs.json's
# mean_vis_dist is only the FINAL position. Final position also cannot separate
# solved runs -- the episode early-exits the moment distance < 0.5, so every
# success ends just inside the radius by construction.
#
# save_images=False here: we want the numbers, not the videos, and it is ~5x faster.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grasp
cd /home/hviswan/Documents/grasp
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU DATASET_DIR=/home/hviswan/Documents/grasp/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset MUJOCO_GL; export DISPLAY=:2
ROOT=/home/hviswan/Documents/grasp/sweep_fk/ablate_trajs
SEEDS="4 6 8 10 12 14 19 21 23 24 25 28 29 30 32 33 34 36 40 41 54 56 58 60 63 71 72 77 80 83 84 85 88 89 91 94 95 98"
for mode in none objective_raw; do
  for s in $SEEDS; do
    OUT=$ROOT/${mode}/seed_$s
    [ -s "$OUT/traj.npz" ] && continue
    mkdir -p "$OUT"
    GRASP_RESULTS_DIR="$OUT" GRASP_TRAJ_DUMP="$OUT/traj.npz" timeout 900 python plan.py \
        model_name=point_maze n_evals=1 planner=fk_pointmaze goal_H=10 \
        goal_source=random_state seed=$s save_images=False \
        planner.opt_steps=150 planner.eval_every=1 planner.num_particles=3 \
        planner.num_smooth_samples=1 planner.v_null=$mode \
        ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints > "$OUT/run.log" 2>&1
  done
  echo "=== DONE traj arm $mode $(date +%T) ==="
done
echo "=== TRAJ DUMPS COMPLETE $(date +%T) ==="
