#!/bin/bash
# Stability screen: run the SAME algorithm twice per seed through two code paths
# that differ only in floating-point noise (adam kernel vs the no-flow shift map).
# A seed is stable only if both solve -- seeds where FK's success flips under 1e-4
# perturbations (e.g. pusht H10 s23) cannot be used to compare configurations.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd /home/hviswan/Documents/grasp; unset MUJOCO_GL; export DISPLAY=:2
R=$PWD/sweep_aft/seed_screen
NOFLOW="++planner.kernel=aft ++planner.flow_type=diag ++planner.flow_shared=false ++planner.flow_scale=false ++planner.flow_persist=true ++planner.flow_steps=1 ++planner.flow_lr=0.1 ++planner.flow_weighted=false ++planner.flow_logdet_weight=0"
np () { [ "$1" = 15 ] && echo 2 || echo 3; }
run () {   # env H seed arm extra...
  local env=$1 H=$2 s=$3 arm=$4; shift 4
  local OUT=$R/$env/H$H/seed_${s}_$arm; [ -s "$OUT/result.txt" ] && return; mkdir -p $OUT
  local model=point_maze planner=fk_pointmaze gsrc=random_state
  [ "$env" = pusht ] && { model=pusht; planner=fk_pusht; gsrc=dset; }
  CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT timeout 3600 python plan.py n_evals=1 seed=$s save_images=False \
    model_name=$model planner=$planner goal_H=$H goal_source=$gsrc planner.opt_steps=150 planner.eval_every=1 \
    planner.num_particles=$(np $H) planner.num_smooth_samples=1 "$@" \
    ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints > $OUT/run.log 2>&1
  local ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1; echo $ok > $OUT/result.txt
  echo "[$env H$H s$s $arm] success=$ok exit=$(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')"
}
for spec in "pointmaze 5 6 17 19 20" "pointmaze 10 6 19 23" "pointmaze 15 19 31 3" \
            "pusht 5 10 20 34 45" "pusht 10 16 23 24" "pusht 15 6 61 94"; do
  set -- $spec; env=$1; H=$2; shift 2
  for s in "$@"; do run $env $H $s pathA; run $env $H $s pathB $NOFLOW; done
done
echo "=== SCREEN DONE ==="
