#!/bin/bash
# Push-T H10 seed 23 (FK solves at step 77; AFT plateaus just outside the goal).
# Walk from the adam kernel to the IAF flow, one change at a time.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; unset MUJOCO_GL; export DISPLAY=:2
R=$PWD/sweep_aft/pusht23_ladder
IAF="++planner.kernel=aft ++planner.flow_type=iaf ++planner.iaf_hids=30 ++planner.iaf_layers=3"
DIAG="++planner.kernel=aft ++planner.flow_type=diag ++planner.flow_shared=false ++planner.flow_scale=false ++planner.flow_persist=true ++planner.flow_steps=1 ++planner.flow_lr=0.1"
declare -A RUNG=(
  [h_noflow_unweighted]="$DIAG ++planner.flow_weighted=false ++planner.flow_logdet_weight=0"
  [g_noflow_weighted_logdet]="$DIAG ++planner.flow_weighted=true ++planner.flow_logdet_weight=1e-3"
  [a_adam]=""
  [b_iaf_base]="$IAF ++planner.flow_weighted=true ++planner.flow_steps=10 ++planner.flow_lr=1e-3 ++planner.flow_logdet_weight=1e-3"
  [c_iaf_nologdet]="$IAF ++planner.flow_weighted=true ++planner.flow_steps=10 ++planner.flow_lr=1e-3 ++planner.flow_logdet_weight=0"
  [d_iaf_unweighted]="$IAF ++planner.flow_weighted=false ++planner.flow_steps=10 ++planner.flow_lr=1e-3 ++planner.flow_logdet_weight=0"
  [f_iaf_shift_only]="$IAF ++planner.iaf_shift_only=true ++planner.flow_weighted=true ++planner.flow_steps=10 ++planner.flow_lr=1e-3 ++planner.flow_logdet_weight=0"
  [e_iaf_40steps]="$IAF ++planner.flow_weighted=false ++planner.flow_steps=40 ++planner.flow_lr=1e-3 ++planner.flow_logdet_weight=0"
)
for r in ${ORDER:-a_adam b_iaf_base c_iaf_nologdet d_iaf_unweighted e_iaf_40steps}; do
  OUT=$R/$r; [ -s "$OUT/result.txt" ] && continue; mkdir -p $OUT
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT timeout 7200 python plan.py n_evals=1 seed=23 save_images=False goal_H=10 \
    model_name=pusht planner=fk_pusht goal_source=dset planner.opt_steps=150 planner.eval_every=1 \
    planner.num_particles=3 planner.num_smooth_samples=1 ${RUNG[$r]} \
    ckpt_base_path=$PWD/checkpoints > $OUT/run.log 2>&1
  ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1; echo "$ok" > $OUT/result.txt
  d=$(grep -oE "'state_dist': array\(\[[0-9.]+" $OUT/run.log | tail -1 | grep -oE "[0-9.]+$")
  echo "[$r] success=$ok exit=$(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}') final_dist=${d:-?} ($(( $(date +%s) - t0 ))s)"
done
echo "=== LADDER DONE ==="
