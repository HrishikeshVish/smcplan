#!/bin/bash
# AFT sanity test (Adam distilled into a shared bounded NICE flow, delta=0.2) on seeds where FK beat the baselines
# (seeds + selection rule in writeup/aft_sanity_seeds.txt). Same planner settings
# as each env's FK sweep; only the move changes. Reacher goals were unseeded in
# the sweep, so reacher runs BOTH fk and aft on the (now fixed) goals.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=$PWD/sweep_aft/sanity_distill
AFT="++planner.kernel=aft ++planner.flow_type=iaf ++planner.iaf_shift_only=true ++planner.iaf_bounded_shift=0.2 ++planner.iaf_hids=30 ++planner.iaf_layers=3 ++planner.iaf_distill=true ++planner.flow_weighted=true ++planner.flow_steps=30 ++planner.flow_lr=3e-3 ++planner.flow_logdet_weight=0"
CK=ckpt_base_path=$PWD/checkpoints

run () {  # env H seed arm gl args...
  local env=$1 H=$2 s=$3 arm=$4 gl=$5; shift 5
  local OUT=$ROOT/$env/${arm}_H$H/seed_$s
  [ -s "$OUT/result.txt" ] && return; mkdir -p $OUT
  if [ "$gl" = egl ]; then export MUJOCO_GL=egl; unset DISPLAY; else unset MUJOCO_GL; export DISPLAY=:2; fi
  for try in 1 2 3; do
    local t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=${GPU:-1} GRASP_RESULTS_DIR=$OUT timeout 3600 python plan.py n_evals=1 seed=$s \
      save_images=False planner.opt_steps=150 planner.eval_every=1 "$@" $CK > $OUT/run.log 2>&1
    if grep -q "Success rate:" $OUT/run.log; then
      local ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
      echo "$ok $(( $(date +%s) - t0 ))" > $OUT/result.txt
      echo "[$env H$H s$s $arm] success=$ok exit=$(tail -1 $OUT/seed_success_log.txt | awk '{print $5}')"
      return
    fi
    grep -qiE "outofmemoryerror|out of memory" $OUT/run.log && { echo "[$env H$H s$s $arm] OOM try $try"; sleep 60; continue; }
    echo "[$env H$H s$s $arm] CRASHED: $(grep -oE '[A-Za-z]*Error[^\"]*' $OUT/run.log | head -1 | cut -c1-100)"; return
  done
}

pm_np () { [ "$1" = 15 ] && echo 2 || echo 3; }
for H in 5 10 15; do case $H in 5) S="10 20 34 45";; 10) S="16 23 24";; 15) S="6 61 94";; esac
  for s in $S; do run pusht $H $s aft none model_name=pusht planner=fk_pusht goal_H=$H goal_source=dset \
      planner.num_particles=$(pm_np $H) planner.num_smooth_samples=1 $AFT; done; done
for s in 17 24 26 30 39 8 9 14 23 32; do
  run cube 5 $s aft egl model_name=lewm_cube planner=fk_cube goal_H=5 goal_source=dset planner.beta_max=10.0 $AFT; done
for H in 5 10 15; do case $H in 5) S="1 2 3 4";; 10) S="1 2 3";; 15) S="1 2 3";; esac
  for s in $S; do for arm in fk aft; do
    A=""; [ $arm = aft ] && A="$AFT"
    run reacher $H $s $arm egl model_name=lewm_reacher planner=fk goal_H=$H goal_source=random_state \
      planner.num_particles=8 planner.num_smooth_samples=8 $A; done; done; done
for H in 5 10 15; do case $H in 5) S="6 17 19 20";; 10) S="6 19 23";; 15) S="19 31 3";; esac
  for s in $S; do run pointmaze $H $s aft none model_name=point_maze planner=fk_pointmaze goal_H=$H goal_source=random_state \
      planner.num_particles=$(pm_np $H) planner.num_smooth_samples=1 $AFT; done; done
echo "=== AFT SANITY DONE ==="
