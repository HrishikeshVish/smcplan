#!/bin/bash
# Is FK's cube seed-24 success robust? Same update, two code paths (fp noise only).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export MUJOCO_GL=egl; unset DISPLAY
export DATASET_DIR=$PWD/data
run () { local OUT=$PWD/sweep_aft/cube24_stab/$1; mkdir -p $OUT; shift
  CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT timeout 3600 python plan.py n_evals=1 seed=24 save_images=False \
    model_name=lewm_cube planner=fk_cube goal_H=5 goal_source=dset planner.opt_steps=150 planner.eval_every=1 \
    planner.beta_max=10.0 "$@" ckpt_base_path=$PWD/checkpoints > $OUT/run.log 2>&1
  ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
  echo "[$(basename $OUT)] success=$ok exit=$(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}') minpos=$(grep -oE "cube_pos_dist': array\(\[[0-9.]+" $OUT/run.log | awk -F'[' '{print $2}' | sort -n | head -1)"; }
run pathA_adam
run pathB_jko_1step ++planner.kernel=jko ++planner.jko_iters=1 ++planner.jko_tau=1e6
echo "=== STAB DONE ==="
