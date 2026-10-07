#!/bin/bash
# Option 2: distil the per-particle Adam step into ONE shared flow, then transport.
# Push-T seeds 20 (FK solves at 29), 24 (FK 103), 23 (FK 77).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd /home/hviswan/Documents/grasp; unset MUJOCO_GL; export DISPLAY=:2
D="++planner.kernel=aft ++planner.flow_type=iaf ++planner.iaf_shift_only=true ++planner.iaf_bounded_shift=0.2 ++planner.iaf_hids=30 ++planner.iaf_layers=3 ++planner.iaf_distill=true ++planner.flow_weighted=true ++planner.flow_steps=30 ++planner.flow_lr=3e-3 ++planner.flow_logdet_weight=0"
for spec in "5 20" "10 24" "10 23"; do
  set -- $spec; H=$1; s=$2
  OUT=$PWD/sweep_aft/distill_bd/H${H}_seed$s; mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT timeout 7200 python plan.py n_evals=1 seed=$s save_images=False \
    model_name=pusht planner=fk_pusht goal_H=$H goal_source=dset planner.opt_steps=150 planner.eval_every=1 \
    planner.num_particles=3 planner.num_smooth_samples=1 $D \
    ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints > $OUT/run.log 2>&1
  ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
  echo "[distill-bd0.2 pusht H$H s$s] success=$ok exit=$(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')"
done
echo "=== DISTILL DONE ==="
