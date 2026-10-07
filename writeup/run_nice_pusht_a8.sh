#!/bin/bash
# NICE-style flow (autoregressive shift only, log|det dT| = 0) on Push-T H10 seeds 23, 24.
# NOTE: seed 23 failed the stability screen (same algorithm flips outcome under fp noise).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd /home/hviswan/Documents/grasp; unset MUJOCO_GL; export DISPLAY=:2
NICE="++planner.kernel=aft ++planner.flow_type=iaf ++planner.iaf_shift_only=true ++planner.iaf_shift_scale=8.0 ++planner.iaf_hids=30 ++planner.iaf_layers=3 ++planner.flow_weighted=true ++planner.flow_steps=10 ++planner.flow_lr=1e-3 ++planner.flow_logdet_weight=0"
for s in 24 23; do
  OUT=$PWD/sweep_aft/nice_pusht_a8/seed_$s; mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT timeout 7200 python plan.py n_evals=1 seed=$s save_images=False \
    model_name=pusht planner=fk_pusht goal_H=10 goal_source=dset planner.opt_steps=150 planner.eval_every=1 \
    planner.num_particles=3 planner.num_smooth_samples=1 $NICE \
    ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints > $OUT/run.log 2>&1
  ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
  echo "[nice-a8 pusht H10 s$s] success=$ok exit=$(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')"
done
echo "=== NICE PUSHT DONE ==="
