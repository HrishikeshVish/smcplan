#!/bin/bash
# NICE-style flow (autoregressive shift only, log|det dT| = 0) on the two Point maze
# seeds whose FK success is known stable. Then resume the seed-stability screen.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd /home/hviswan/Documents/grasp; unset MUJOCO_GL; export DISPLAY=:2
NICE="++planner.kernel=aft ++planner.flow_type=iaf ++planner.iaf_shift_only=true ++planner.iaf_hids=30 ++planner.iaf_layers=3 ++planner.flow_weighted=true ++planner.flow_steps=10 ++planner.flow_lr=1e-3 ++planner.flow_logdet_weight=0"
for H in 5 10; do
  OUT=$PWD/sweep_aft/nice_pm/H${H}_seed6; mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT timeout 3600 python plan.py n_evals=1 seed=6 save_images=False \
    model_name=point_maze planner=fk_pointmaze goal_H=$H goal_source=random_state planner.opt_steps=150 \
    planner.eval_every=1 planner.num_particles=3 planner.num_smooth_samples=1 $NICE \
    ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints > $OUT/run.log 2>&1
  ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
  echo "[nice pointmaze H$H s6] success=$ok exit=$(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')"
done
echo "=== NICE TEST DONE; resuming seed screen ==="
nohup bash writeup/run_seed_screen.sh >> writeup/seed_screen.log 2>&1 &
