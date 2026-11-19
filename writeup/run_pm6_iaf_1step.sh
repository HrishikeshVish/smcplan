#!/bin/bash
# AFT with the reference repo's AffineInverseAutoregressiveFlow (one shared map),
# Point maze H10 seed 6 (FK solves at step 4).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; unset MUJOCO_GL; export DISPLAY=:2
OUT=$PWD/sweep_aft/pm6_iaf_1step; mkdir -p $OUT
CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT timeout 7200 python plan.py n_evals=1 seed=6 save_images=False goal_H=10 \
  model_name=point_maze planner=fk_pointmaze goal_source=random_state planner.opt_steps=150 planner.eval_every=1 \
  planner.num_particles=3 planner.num_smooth_samples=1 \
  ++planner.kernel=aft ++planner.flow_type=iaf ++planner.flow_weighted=true ++planner.flow_steps=1 \
  ++planner.flow_lr=1e-3 ++planner.flow_logdet_weight=1e-3 ++planner.iaf_hids=30 ++planner.iaf_layers=3 \
  ckpt_base_path=$PWD/checkpoints > $OUT/run.log 2>&1
ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
echo "[pm6 iaf 1-step] success=$ok exit=$(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')"
echo "=== DONE ==="
