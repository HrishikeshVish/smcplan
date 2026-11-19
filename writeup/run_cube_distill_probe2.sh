#!/bin/bash
# Warm-started distillation on cube: does derr drop well below tgt?
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export MUJOCO_GL=egl; unset DISPLAY
export DATASET_DIR=$PWD/data
B="++planner.kernel=aft ++planner.flow_type=iaf ++planner.iaf_shift_only=true ++planner.iaf_hids=30 ++planner.iaf_layers=3 ++planner.iaf_distill=true ++planner.flow_weighted=true ++planner.flow_logdet_weight=0 ++planner.iaf_bounded_shift=0.2"
probe () {  # tag steps lr persist
  local OUT=$PWD/sweep_aft/cube_distill_probe2/$1; mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT timeout 3600 python plan.py n_evals=1 seed=17 save_images=False \
    model_name=lewm_cube planner=fk_cube goal_H=5 goal_source=dset planner.opt_steps=25 planner.eval_every=1 \
    planner.beta_max=10.0 $B ++planner.flow_steps=$2 ++planner.flow_lr=$3 ++planner.flow_persist=$4 \
    ckpt_base_path=$PWD/checkpoints > $OUT/run.log 2>&1
  echo "[$1 steps=$2 lr=$3 warm=$4] $(grep 'FK-SMC\] step' $OUT/run.log | tail -1 | grep -oE 'derr=[0-9./]+|disp=[0-9.e-]+|L_dyn=[0-9.]+' | paste -sd' ')"
}
probe warm30   30 3e-3 true
probe warm30lr 30 1e-2 true
probe warm60   60 3e-3 true
echo "=== PROBE2 DONE ==="
