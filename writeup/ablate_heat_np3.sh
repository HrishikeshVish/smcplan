#!/bin/bash
# HEAT-EQUATION ABLATION, np=3, pointmaze H10, 38 seeds.
#
# ARMS: none (heat cost-to-go V) vs objective_raw (the L2 planning objective,
# substituted AS-IS into the FK tilt). The earlier scale-matched arm is
# DELETED on purpose -- see WHY SCALE IS NOT A CONFOUND below.
#
# WHY np=3 AND NOT np=2. At P=2 the ablation is mathematically degenerate.
# Standardizing any 2-vector gives exactly +/- sqrt(2)/2, so rescaling to V's
# mean/std returns Vbar +/- |Va-Vb|/2 -- i.e. EXACTLY {Va,Vb}. The rescaled arm
# is then an identity whenever the two quantities agree on which of the two
# particles is worse, which is why the np=2 runs produced byte-identical traces
# (same dEV, same cumKL=0.64, same resamples=1) on every seed. More generally,
# with 2 particles "V's ordering" carries at most ONE BIT, so no amount of extra
# seeds can make that comparison informative. P>=3 is the minimum for ordering
# to mean anything.
#
# WHY SCALE IS NOT A CONFOUND. Measured at np=2/H5, final cumKL:
#   none          0.66  0.68  0.04
#   objective_raw 0.18  0.48  0.00
# The raw objective tilts SYSTEMATICALLY WEAKER -- on seed 90 it never acted at
# all. Initially treated as a confound to normalize away; that was wrong. beta
# is ESS-adaptive and capped at beta_max, so the realized selection pressure is
# a function of how far apart the tilt quantity spreads the population. If the
# heat solve spreads particles more than the objective does, that spread IS the
# mechanism, and normalizing it away deletes the effect being measured. So the
# raw substitution is the honest ablation, and cumKL/ESS/resamples are PRIMARY
# outcomes here, not diagnostics.
#
# SEEDS: the 38 pointmaze H10 seeds where GD FAILS and real FK SUCCEEDS. GD is
# exactly the L2 objective, so the objective alone provably does not solve any
# of them, and real FK at np=3 solves all 38 -- full headroom to fall.
#
# METRICS: success is nearly useless here (none is 38/38 by construction). The
# question is whether ours is MORE OPTIMAL, so exit_step (iterations to first
# success, as in the paper's tab:main) and final state_dist are the primary
# comparisons, on the seeds where both arms succeed.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grasp
cd /home/hviswan/Documents/grasp
GPU=${GPU:-1}
export CUDA_VISIBLE_DEVICES=$GPU DATASET_DIR=/home/hviswan/Documents/grasp/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset MUJOCO_GL; export DISPLAY=:2

ROOT=${ROOT:-/home/hviswan/Documents/grasp/sweep_fk/ablate_np3}
H=${H:-10}
NP=${NP:-3}
STEPS=150
PER_RUN_TIMEOUT=1200
MAX_TRIES=3
# np=3 at goal_H=10 measured OK at ~8.5 GB free, OOM at ~6.3 GB. Gate high and
# WAIT: a skipped seed silently shrinks the paired set.
NEED_MIB=${NEED_MIB:-8000}
GATE_TIMEOUT=${GATE_TIMEOUT:-86400}
SEEDS=${SEEDS:-"4 6 8 10 12 14 19 21 23 24 25 28 29 30 32 33 34 36 40 41 54 56 58 60 63 71 72 77 80 83 84 85 88 89 91 94 95 98"}

gpu_free () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU; }
wait_for_gpu () {
    local waited=0
    while [ "$(gpu_free)" -lt "$NEED_MIB" ]; do
        [ "$waited" -ge "$GATE_TIMEOUT" ] && { echo "=== GATE TIMEOUT (free=$(gpu_free)) ==="; return 1; }
        sleep 60; waited=$((waited + 60))
    done
    return 0
}

echo "=== ABLATION np=$NP H=$H START $(date +%T) | $(echo $SEEDS|wc -w) seeds x 2 arms ==="
for mode in none objective_raw; do
  for s in $SEEDS; do
    OUT=$ROOT/${mode}_H${H}/seed_$s
    [ -s "$OUT/result.txt" ] && continue
    mkdir -p "$OUT"
    for try in $(seq 1 $MAX_TRIES); do
      wait_for_gpu || { echo "[np3 $mode s$s] ABANDONED (gate)"; break; }
      t0=$(date +%s)
      GRASP_RESULTS_DIR="$OUT" timeout $PER_RUN_TIMEOUT python plan.py \
          model_name=point_maze n_evals=1 planner=fk_pointmaze \
          goal_H=$H goal_source=random_state seed=$s save_images=False \
          planner.opt_steps=$STEPS planner.eval_every=1 \
          planner.num_particles=$NP planner.num_smooth_samples=1 \
          planner.v_null=$mode \
          ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints \
          > "$OUT/run.log" 2>&1
      dt=$(( $(date +%s) - t0 ))
      if grep -q "Success rate:" "$OUT/run.log" 2>/dev/null; then
          ok=0; grep -q "Success rate:  1.0" "$OUT/run.log" && ok=1
          echo "$ok $dt" > "$OUT/result.txt"
          echo "[np3 $mode s$s] ${dt}s success=$ok"
          break
      fi
      if grep -q "OutOfMemoryError" "$OUT/run.log"; then
          echo "[np3 $mode s$s] OOM try $try"; sleep 30
      else
          echo "[np3 $mode s$s] FAILED: $(grep -oE '[A-Za-z]*Error[^\"]*' $OUT/run.log|head -1|cut -c1-90)"; break
      fi
    done
  done
  echo "=== DONE arm $mode $(date +%T) ==="
done
echo "=== ABLATION np3 COMPLETE $(date +%T) ==="
