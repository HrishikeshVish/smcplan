#!/bin/bash
# Preconditioned HMC kernel at FK-like scale: check acceptance and that it solves
# the seeds FK solves. pointmaze H10 s6 (FK: step 4), pusht H10 s24 (FK: 103).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; unset MUJOCO_GL; export DISPLAY=:2
H="++planner.kernel=hmc ++planner.hmc_leaps=1 ++planner.hmc_beta1=0.9 ++planner.hmc_warmup=10"
run () { local OUT_TAG=$1; local OUT=$PWD/sweep_aft/hmc_probe2/$1; mkdir -p $OUT; local env=$2 H_=$3 s=$4; shift 4
  local model=point_maze pl=fk_pointmaze gs=random_state
  [ "$env" = pusht ] && { model=pusht; pl=fk_pusht; gs=dset; }
  CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT timeout 7200 python plan.py n_evals=1 seed=$s save_images=False \
    model_name=$model planner=$pl goal_H=$H_ goal_source=$gs planner.opt_steps=150 planner.eval_every=1 \
    planner.num_particles=3 planner.num_smooth_samples=1 $H "$@" \
    ckpt_base_path=$PWD/checkpoints > $OUT/run.log 2>&1
  ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
  echo "[$OUT_TAG] success=$ok exit=$(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}') mean_acc=$(grep -oE "logdet=[0-9.]+" $OUT/run.log | awk -F= '{t+=$2;n++} END{if(n)printf "%.3f", t/n}')"; }
run pm_s6   pointmaze 10 6
run pusht_s24 pusht 10 24
echo "=== HMC PROBE DONE ==="
