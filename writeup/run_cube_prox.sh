#!/bin/bash
# Implicit (proximal) transport, no learned map, on the cube seeds from the sanity set.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd /home/hviswan/Documents/grasp; export MUJOCO_GL=egl; unset DISPLAY
export DATASET_DIR=/home/hviswan/Documents/grasp/data
PROX="++planner.kernel=aft ++planner.flow_type=prox ++planner.prox_iters=3"
for s in 24 17 26 30 39 8 9 14 23 32; do
  OUT=$PWD/sweep_aft/cube_prox_m/seed_$s; [ -s "$OUT/result.txt" ] && continue; mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT timeout 7200 python plan.py n_evals=1 seed=$s save_images=False \
    model_name=lewm_cube planner=fk_cube goal_H=5 goal_source=dset planner.opt_steps=150 planner.eval_every=1 \
    planner.beta_max=10.0 $PROX ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints > $OUT/run.log 2>&1
  ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1; echo $ok > $OUT/result.txt
  echo "[prox-m cube s$s] success=$ok exit=$(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')"
done
echo "=== PROX CUBE DONE ==="
