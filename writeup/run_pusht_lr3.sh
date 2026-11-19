#!/bin/bash
# Pause the sanity set after pusht H10 s24, test flow lr 3e-3 on the slow Push-T H10
# seeds (23, 24), then resume the sanity set (it skips finished seeds).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; unset MUJOCO_GL; export DISPLAY=:2
rm -rf sweep_aft/sanity_iaf10/pusht/aft_H15   # partial run killed mid-way
R=$PWD/sweep_aft/pusht_lr3e-3
for s in 23 24; do OUT=$R/seed_$s; mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT timeout 3600 python plan.py n_evals=1 seed=$s save_images=False goal_H=10 \
    model_name=pusht planner=fk_pusht goal_source=dset planner.opt_steps=150 planner.eval_every=1 \
    planner.num_particles=3 planner.num_smooth_samples=1 \
    ++planner.kernel=aft ++planner.flow_type=iaf ++planner.flow_weighted=true ++planner.flow_steps=10 \
    ++planner.flow_lr=3e-3 ++planner.flow_logdet_weight=1e-3 ++planner.iaf_hids=30 ++planner.iaf_layers=3 \
    ckpt_base_path=$PWD/checkpoints > $OUT/run.log 2>&1
  ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
  echo "[pusht H10 s$s aft lr3e-3] success=$ok exit=$(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')"
done
echo "=== LR TEST DONE; resuming sanity set ==="
nohup bash writeup/run_aft_sanity_iaf.sh >> writeup/aft_sanity_iaf10.log 2>&1 &
