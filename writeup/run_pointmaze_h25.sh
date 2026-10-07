#!/bin/bash
# Point maze long horizon: goal_H=25 (= 125 environment steps at frameskip 5),
# all four planners x 100 seeds.
#
# WHY POINT MAZE AND NOT PUSH-T. Push-T at H=25 is all zeros -- gd 1/100,
# grasp 0, cem 0, ours 0 -- so it bounds where everything stops working but
# says nothing comparative. Point maze does NOT collapse with horizon
# (gd 54/51/50, cem 51/67/69 at H=5/10/15), so a longer horizon there is
# actually informative.
#
# Goals are sampled independently at random valid maze positions
# (point_maze_wrapper.sample_random_init_goal_states), so goal difficulty does
# not grow with H -- the longer horizon buys budget, not distance. 125 env
# steps also fits under the environment's 150-step registration.
#
# Our arm runs 2 particles: measured 2026-09-20, np=3 OOMs at H=25 on this
# DINO-WM env and np=2 runs. That is the smallest population anywhere in the
# paper and must be stated when the cell is reported.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grasp
cd /home/hviswan/Documents/grasp
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU DATASET_DIR=/home/hviswan/Documents/grasp/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset MUJOCO_GL; export DISPLAY=:2

ROOT=/home/hviswan/Documents/grasp/sweep_pointmaze_h25
# H=25 retains 125 env steps of rollout for autograd and needs most of the card:
# a 2-particle run climbed to 8.6 GB and OOM'd while the H<=15 sweeps held only
# 5.3 GB between them. Gate high so this waits for the card rather than dying.
NEED_MIB=${NEED_MIB:-11500}
GATE_TIMEOUT=${GATE_TIMEOUT:-86400}
PER_RUN_TIMEOUT=2400
SEEDS=${SEEDS:-$(seq 1 100)}
H=25

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
        # 2 particles, not 3: the retained rollout grows with horizon on the
        # DINO-WM envs and 3 already OOM at H=15 there, so H=25 gets the H15 setting.
        fk)    echo "planner=fk_pointmaze planner.opt_steps=150 planner.eval_every=1 planner.num_particles=2 planner.num_smooth_samples=1" ;;
    esac
}

for planner in ${PLANNERS:-fk gd cem grasp}; do
  for s in $SEEDS; do
    OUT=$ROOT/${planner}/seed_$s
    [ -s "$OUT/result.txt" ] && continue
    mkdir -p "$OUT"
    for try in 1 2 3; do
      wait_for_gpu || { echo "=== ABANDONED $planner s$s ==="; break; }
      t0=$(date +%s)
      GRASP_RESULTS_DIR="$OUT" timeout $PER_RUN_TIMEOUT python plan.py \
          model_name=point_maze n_evals=1 goal_H=$H goal_source=random_state seed=$s \
          save_images=False $(planner_args $planner) \
          ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints \
          > "$OUT/run.log" 2>&1
      dt=$(( $(date +%s) - t0 ))
      if grep -q "Success rate:" "$OUT/run.log" 2>/dev/null; then
          ok=0; grep -q "Success rate:  1.0" "$OUT/run.log" && ok=1
          echo "$ok $dt" > "$OUT/result.txt"
          echo "[pm-h25 $planner s$s] ${dt}s success=$ok"
          break
      fi
      # BOTH spellings: PyTorch raises "OutOfMemoryError" in one path and
      # "AcceleratorError: CUDA error: out of memory" in another. Matching only
      # the first is why 36 seeds were dropped without a retry.
      if grep -qiE "outofmemoryerror|out of memory" "$OUT/run.log"; then
          echo "[pm-h25 $planner s$s] OOM try $try, retrying"; sleep 30
      else
          echo "[pm-h25 $planner s$s] FAILED ${dt}s: $(grep -oE '[A-Za-z]*Error[^\"]*' $OUT/run.log|head -1|cut -c1-70)"
          break
      fi
    done
  done
  echo "=== DONE pm-h25 $planner $(date +%T) ==="
done
echo "=== PUSHT H25 COMPLETE $(date +%T) ==="
