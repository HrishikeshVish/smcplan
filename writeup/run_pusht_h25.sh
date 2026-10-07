#!/bin/bash
# Push-T long horizon: goal_H=25 (= 125 environment steps at frameskip 5),
# all four planners x 100 seeds, dataset goals.
#
# WHY dset AND NOT random_state. Push-T's sample_random_init_goal_states places
# the T-block and the agent independently, so many start/goal pairs are simply
# unconnectable. Measured over the quarantined runs in
# sweep/_pusht_random_state_invalid: 2 successes in 664 runs across gd/grasp/cem
# and H in {5,10,15}, and it does NOT improve with horizon (gd 0%/1%/1%). A
# longer horizon there buys a longer floor, not a long-horizon result.
#
# H=25 is supported by the data: Push-T episodes are 246 frames and plan.py
# needs frameskip*goal_H+1 = 126.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grasp
cd /home/hviswan/Documents/grasp
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU DATASET_DIR=/home/hviswan/Documents/grasp/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset MUJOCO_GL; export DISPLAY=:2

ROOT=/home/hviswan/Documents/grasp/sweep_pusht_h25
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
        fk)    echo "planner=fk_pusht planner.opt_steps=150 planner.eval_every=1 planner.num_particles=2 planner.num_smooth_samples=1" ;;
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
          model_name=pusht n_evals=1 goal_H=$H goal_source=dset seed=$s \
          save_images=False $(planner_args $planner) \
          ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints \
          > "$OUT/run.log" 2>&1
      dt=$(( $(date +%s) - t0 ))
      if grep -q "Success rate:" "$OUT/run.log" 2>/dev/null; then
          ok=0; grep -q "Success rate:  1.0" "$OUT/run.log" && ok=1
          echo "$ok $dt" > "$OUT/result.txt"
          echo "[pusht-h25 $planner s$s] ${dt}s success=$ok"
          break
      fi
      # BOTH spellings: PyTorch raises "OutOfMemoryError" in one path and
      # "AcceleratorError: CUDA error: out of memory" in another. Matching only
      # the first is why 36 seeds were dropped without a retry.
      if grep -qiE "outofmemoryerror|out of memory" "$OUT/run.log"; then
          echo "[pusht-h25 $planner s$s] OOM try $try, retrying"; sleep 30
      else
          echo "[pusht-h25 $planner s$s] FAILED ${dt}s: $(grep -oE '[A-Za-z]*Error[^\"]*' $OUT/run.log|head -1|cut -c1-70)"
          break
      fi
    done
  done
  echo "=== DONE pusht-h25 $planner $(date +%T) ==="
done
echo "=== PUSHT H25 COMPLETE $(date +%T) ==="
