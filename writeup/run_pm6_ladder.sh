#!/bin/bash
# Point maze H10 seed 6 (FK solves at step 4; AFT fails). Stops at the first failure.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd /home/hviswan/Documents/grasp; unset MUJOCO_GL; export DISPLAY=:2
R=$PWD/sweep_aft/pm6_ladder
B="++planner.kernel=aft ++planner.flow_persist=true ++planner.flow_shared=false ++planner.flow_weighted=false ++planner.flow_steps=1 ++planner.flow_lr=0.1"
declare -A RUNG=([a_adam]="" [b_shift_only]="$B ++planner.flow_scale=false ++planner.flow_logdet_weight=0" [c_scale_nologdet]="$B ++planner.flow_scale=true ++planner.flow_logdet_weight=0")
for r in a_adam b_shift_only c_scale_nologdet; do OUT=$R/$r; mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT timeout 3600 python plan.py n_evals=1 seed=6 save_images=False goal_H=10 \
    model_name=point_maze planner=fk_pointmaze goal_source=random_state planner.opt_steps=150 planner.eval_every=1 \
    planner.num_particles=3 planner.num_smooth_samples=1 ${RUNG[$r]} ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints > $OUT/run.log 2>&1
  ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
  echo "[$r] success=$ok exit=$(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')"
  [ $ok = 0 ] && { echo "=== FIRST FAILURE: $r ==="; break; }
done
echo "=== DONE ==="
