#!/bin/bash
# Push-T baselines: the 8 unscheduled cells plus the 19 missing gd H5 seeds.
# This is the gap that leaves our Push-T H10/H15 numbers (27 and 5) standing
# without any comparison.
#
# Protocol matches writeup/run_sweep.sh exactly: goal_source=dset (Push-T's
# random start/goal pairs are frequently unconnectable and floor every planner
# near zero), 150 optimiser steps, CEM matched to 150 rollouts (5 x 30).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU DATASET_DIR=$PWD/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset MUJOCO_GL; export DISPLAY=:2

ROOT=$PWD/sweep
NEED_MIB=${NEED_MIB:-7000}
GATE_TIMEOUT=${GATE_TIMEOUT:-86400}
PER_RUN_TIMEOUT=1200
SEEDS=${SEEDS:-$(seq 1 100)}
HORIZONS=${HORIZONS:-"5 10 15"}

gpu_free () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU; }
wait_for_gpu () {
    local waited=0
    while [ "$(gpu_free)" -lt "$NEED_MIB" ]; do
        [ "$waited" -ge "$GATE_TIMEOUT" ] && { echo "=== GATE TIMEOUT ==="; return 1; }
        sleep 60; waited=$((waited + 60))
    done
}
planner_args () {
    case $1 in
        gd)    echo "planner=gd planner.opt_steps=150 planner.eval_every=1 planner.hjb_tau=0.0" ;;
        grasp) echo "planner=grasp planner.opt_steps=150 planner.eval_every=1" ;;
        cem)   echo "planner=cem planner.opt_steps=5 planner.num_samples=30 planner.topk=3 planner.eval_every=1" ;;
    esac
}

for planner in ${PLANNERS:-gd grasp cem}; do
  for H in $HORIZONS; do
    for s in $SEEDS; do
      OUT=$ROOT/pusht/${planner}_H${H}/seed_$s
      [ -s "$OUT/result.txt" ] && continue
      mkdir -p "$OUT"
      wait_for_gpu || { echo "=== ABANDONED pusht $planner H$H s$s ==="; continue; }
      t0=$(date +%s)
      GRASP_RESULTS_DIR="$OUT" timeout $PER_RUN_TIMEOUT python plan.py \
          model_name=pusht n_evals=1 goal_H=$H goal_source=dset seed=$s \
          save_images=False $(planner_args $planner) \
          ckpt_base_path=$PWD/checkpoints \
          > "$OUT/run.log" 2>&1
      dt=$(( $(date +%s) - t0 ))
      if grep -q "Success rate:" "$OUT/run.log" 2>/dev/null; then
          ok=0; grep -q "Success rate:  1.0" "$OUT/run.log" && ok=1
          echo "$ok $dt" > "$OUT/result.txt"
          echo "[pusht $planner H$H s$s] ${dt}s success=$ok"
      else
          echo "[pusht $planner H$H s$s] FAILED ${dt}s: $(grep -oE '[A-Za-z]*Error[^\"]*' $OUT/run.log|head -1|cut -c1-70)"
      fi
    done
    echo "=== DONE pusht $planner H$H $(date +%T) ==="
  done
done
echo "=== PUSHT BASELINES COMPLETE $(date +%T) ==="
