#!/bin/bash
# AFT kernel (fk_planner kernel=aft) on reacher H10, seeds 1-10, 8 particles.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd /home/hviswan/Documents/grasp
export MUJOCO_GL=egl; unset DISPLAY
ROOT=${ROOT:-$PWD/sweep_aft/reacher_H10}
for s in ${SEEDS:-$(seq 1 10)}; do
  OUT=$ROOT/seed_$s; [ -s "$OUT/result.txt" ] && continue; mkdir -p $OUT
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=${GPU:-1} GRASP_RESULTS_DIR=$OUT timeout 3600 python plan.py model_name=lewm_reacher n_evals=1 planner=fk \
    goal_H=10 goal_source=random_state seed=$s save_images=False planner.opt_steps=150 planner.eval_every=1 \
    planner.num_particles=8 planner.num_smooth_samples=8 +planner.kernel=aft ${EXTRA:-} \
    ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints > $OUT/run.log 2>&1
  dt=$(( $(date +%s) - t0 ))
  if grep -q "Success rate:" $OUT/run.log; then ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
    echo "$ok $dt" > $OUT/result.txt; echo "[aft s$s] ${dt}s success=$ok exit=$(tail -1 $OUT/seed_success_log.txt | awk '{print $5}')"
  else echo "[aft s$s] FAILED TO RUN"; tail -3 $OUT/run.log; fi
done
echo "=== AFT DONE ==="
