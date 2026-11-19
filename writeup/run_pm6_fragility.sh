#!/bin/bash
# Is FK's seed-6 success robust? Plain adam kernel, tiny non-AFT perturbations.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; unset MUJOCO_GL; export DISPLAY=:2
R=$PWD/sweep_aft/pm6_fragility
for tag in temp0.9x:planner.langevin_temp=0.0009 temp1.1x:planner.langevin_temp=0.0011 lr0.95x:planner.langevin_lr=0.095; do
  name=${tag%%:*}; arg=${tag#*:}; OUT=$R/$name; mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT timeout 3600 python plan.py n_evals=1 seed=6 save_images=False goal_H=10 \
    model_name=point_maze planner=fk_pointmaze goal_source=random_state planner.opt_steps=150 planner.eval_every=1 \
    planner.num_particles=3 planner.num_smooth_samples=1 $arg ckpt_base_path=$PWD/checkpoints > $OUT/run.log 2>&1
  ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
  echo "[adam $name] success=$ok exit=$(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')"
done
echo "=== DONE ==="
