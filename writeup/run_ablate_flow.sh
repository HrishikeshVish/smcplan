#!/bin/bash
# Ablate the two axes that separate the AFT move from FK's per-particle Adam step:
#   B = per-particle flows, position-dependent displacement (interpolation kept)
#   C = ONE shared constant shift, no position dependence (interpolation removed)
# Both use the bounded shift (delta=0.1), weighted fit, 10 Adam steps.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd /home/hviswan/Documents/grasp; unset MUJOCO_GL; export DISPLAY=:2
COMMON="++planner.flow_weighted=true ++planner.flow_steps=10 ++planner.flow_lr=1e-3 ++planner.flow_logdet_weight=0"
B="++planner.kernel=aft ++planner.flow_type=iaf ++planner.iaf_shift_only=true ++planner.iaf_bounded_shift=0.1 ++planner.iaf_hids=30 ++planner.iaf_layers=3 ++planner.iaf_per_particle=true $COMMON"
C="++planner.kernel=aft ++planner.flow_type=diag ++planner.flow_shared=true ++planner.flow_scale=false ++planner.flow_persist=false $COMMON"
for arm in B C; do
  case $arm in B) ARGS="$B"; tag=perparticle_iaf;; C) ARGS="$C"; tag=shared_constshift;; esac
  for spec in "5 20" "10 24" "10 23"; do
    set -- $spec; H=$1; s=$2
    OUT=$PWD/sweep_aft/ablate_flow/${tag}/H${H}_seed$s; [ -s "$OUT/result.txt" ] && continue; mkdir -p $OUT
    CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT timeout 7200 python plan.py n_evals=1 seed=$s save_images=False \
      model_name=pusht planner=fk_pusht goal_H=$H goal_source=dset planner.opt_steps=150 planner.eval_every=1 \
      planner.num_particles=3 planner.num_smooth_samples=1 $ARGS \
      ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints > $OUT/run.log 2>&1
    ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1; echo $ok > $OUT/result.txt
    echo "[$tag pusht H$H s$s] success=$ok exit=$(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')"
  done
done
echo "=== ABLATION DONE ==="
