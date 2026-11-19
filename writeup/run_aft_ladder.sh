#!/bin/bash
# Adam kernel -> AFT, one change per rung, on reacher H10 seed 2 (FK solves it,
# AFT does not). Stops at the first failing rung after one confirming rerun
# (FK is nondeterministic at a fixed seed).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export MUJOCO_GL=egl; unset DISPLAY
SEED=${SEED:-2}
ROOT=$PWD/sweep_aft/ladder_s$SEED
F="+planner.kernel=aft planner.langevin_temp=0"
FP="+planner.kernel=aft +planner.flow_persist=true"   # Langevin noise + optimizer state kept
L2="$F +planner.flow_shared=false +planner.flow_scale=false +planner.flow_weighted=false +planner.flow_steps=1 +planner.flow_lr=0.1 +planner.flow_logdet_weight=0"
L3="$FP +planner.flow_shared=false +planner.flow_scale=false +planner.flow_weighted=false +planner.flow_steps=1 +planner.flow_lr=0.1 +planner.flow_logdet_weight=0"
L4="$FP +planner.flow_shared=false +planner.flow_scale=false +planner.flow_weighted=true +planner.flow_steps=1 +planner.flow_lr=0.1 +planner.flow_logdet_weight=0"
L5="$FP +planner.flow_shared=false +planner.flow_scale=true +planner.flow_weighted=true +planner.flow_steps=1 +planner.flow_lr=0.1 +planner.flow_logdet_weight=0"
L6="$FP +planner.flow_shared=true +planner.flow_scale=true +planner.flow_weighted=true +planner.flow_steps=1 +planner.flow_lr=0.1 +planner.flow_logdet_weight=0"
L7="$FP +planner.flow_shared=true +planner.flow_scale=true +planner.flow_weighted=true +planner.flow_steps=1 +planner.flow_lr=0.1 +planner.flow_logdet_weight=1e-3"
L7P="$FP +planner.flow_shared=false +planner.flow_scale=true +planner.flow_weighted=true +planner.flow_steps=1 +planner.flow_lr=0.1 +planner.flow_logdet_weight=1e-3"
declare -A RUNG=(
  [L0_adam]=""
  [L1_no_noise]="planner.langevin_temp=0"
  [L2_perparticle_shift_1step]="$L2"
  [L3_fit_10x0.05]="$L3"
  [L4_weighted]="$L4"
  [L5_scale]="$L5"
  [L6_shared]="$L6"
  [L7_logdet]="$L7"
  [L7p_perparticle_logdet]="$L7P"
)
ORDER=${ORDER:-"L0_adam L1_no_noise L2_perparticle_shift_1step L3_fit_10x0.05 L4_weighted L5_scale L6_shared L7_logdet"}

run () {   # name try
  local OUT=$ROOT/$1/try$2; mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=${GPU:-1} GRASP_RESULTS_DIR=$OUT timeout 3600 python plan.py model_name=lewm_reacher \
    n_evals=1 planner=fk goal_H=10 goal_source=random_state seed=$SEED save_images=False \
    planner.opt_steps=150 planner.eval_every=1 planner.num_particles=8 planner.num_smooth_samples=8 \
    ${RUNG[$1]} ckpt_base_path=$PWD/checkpoints > $OUT/run.log 2>&1
  local ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
  grep -q "Success rate:" $OUT/run.log || { echo "[$1 try$2] CRASHED"; tail -3 $OUT/run.log; return 2; }
  echo "[$1 try$2] success=$ok exit=$(tail -1 $OUT/seed_success_log.txt | awk '{print $5}')"
  return $((1 - ok))
}

for r in $ORDER; do
  run $r 1 && continue
  run $r 2 && { echo "[$r] flaky: failed once, solved on rerun -- continuing"; continue; }
  echo "=== FIRST FAILING RUNG: $r ==="; exit 0
done
echo "=== ALL RUNGS SOLVED ==="
