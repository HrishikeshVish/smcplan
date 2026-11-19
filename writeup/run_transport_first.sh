#!/bin/bash
# Alg-1 order (transport BEFORE reweight/resample) vs our order, bounded-shift NICE,
# on the three Push-T seeds that discriminate: H5 s20 (needs a big early move),
# H10 s24 (bounded solves at 57), H10 s23 (bounded gets close, regresses late).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; unset MUJOCO_GL; export DISPLAY=:2
BASE="++planner.kernel=aft ++planner.flow_type=iaf ++planner.iaf_shift_only=true ++planner.iaf_bounded_shift=0.1 ++planner.iaf_hids=30 ++planner.iaf_layers=3 ++planner.flow_weighted=true ++planner.flow_steps=10 ++planner.flow_lr=1e-3 ++planner.flow_logdet_weight=0 ++planner.transport_first=true"
for spec in "5 20" "10 24" "10 23"; do
  set -- $spec; H=$1; s=$2
  OUT=$PWD/sweep_aft/transport_first/H${H}_seed$s; mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT timeout 7200 python plan.py n_evals=1 seed=$s save_images=False \
    model_name=pusht planner=fk_pusht goal_H=$H goal_source=dset planner.opt_steps=150 planner.eval_every=1 \
    planner.num_particles=3 planner.num_smooth_samples=1 $BASE \
    ckpt_base_path=$PWD/checkpoints > $OUT/run.log 2>&1
  ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
  echo "[transport-first pusht H$H s$s] success=$ok exit=$(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')"
done
echo "=== TRANSPORT-FIRST DONE ==="
