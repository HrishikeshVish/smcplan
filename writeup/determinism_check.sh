#!/bin/bash
# Same config, same seed, twice; compare goals and per-step planner output.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd /home/hviswan/Documents/grasp; export MUJOCO_GL=egl; unset DISPLAY
for t in 1 2; do OUT=$PWD/sweep_aft/determinism/run$t; rm -rf $OUT; mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT python plan.py model_name=lewm_reacher n_evals=1 planner=fk goal_H=10 \
    goal_source=random_state seed=2 save_images=False planner.opt_steps=15 planner.eval_every=1 planner.num_particles=8 \
    planner.num_smooth_samples=8 ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints > $OUT/run.log 2>&1
done
echo DONE
