#!/bin/bash
# HEAT-EQUATION ABLATION on pointmaze, seeds 1-5 x H in {5,10,15}.
#
# Replaces the heat cost-to-go V in the FK tilt with the L2 planning objective
# the Langevin kernel already descends (planner.v_null, fk_planner.py). If the
# roadmap solve carries no information beyond the objective, the ablated runs
# should reproduce the real ones.
#
# Two variants, because a raw swap confounds information with scale:
#   objective_raw -- literal substitution. beta is capped at beta_max, so a
#                    narrower-spread tilt quantity (L_dyn ~1-2.6 vs V on [0,20])
#                    gives a weaker exponent; a null here may be scale, not
#                    information.
#   objective     -- rescaled to V's per-step mean/std, so selection pressure is
#                    held fixed and ONLY the ordering changes. This is the one
#                    that actually isolates the heat solve's content.
#
# BUDGET: np=2 at EVERY horizon, and the real method (v_null=none) is re-run
# here rather than read from sweep_fk/table. Reason: GPU1 has a ~9.3 GB
# co-tenant, and np=3 at goal_H=10 needs ~8 GB (measured 2026-09-11: our process
# reached 6.11 GiB and died with 68 MiB left). Rather than wait, all three arms
# run at np=2 so they are paired WITH EACH OTHER at one budget.
#
# CONSEQUENCE: these numbers are NOT comparable to the main table (pointmaze
# H5/H10 there are np=3), and the absolute rates will be lower. The ablation
# contrast is still valid because every arm shares the budget -- but the caption
# must say np=2, and must not quote the table's 75/81/61 alongside it.
# Re-running "none" also controls for FK's run-to-run nondeterminism, which is
# ~3-4 points at n=100 and much larger at n=5.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
GPU=${GPU:-1}
export CUDA_VISIBLE_DEVICES=$GPU DATASET_DIR=$PWD/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset MUJOCO_GL; export DISPLAY=:2

ROOT=${ROOT:-$PWD/sweep_fk/ablate_heat/pointmaze}

# Seeds are CHOSEN, not 1-5. Each must satisfy three things or the ablation
# cannot say anything:
#   1. real FK SUCCEEDS   -> there is headroom to lose. On seeds 1-5 the real
#                            method scores 14/15, so any ablation result is
#                            indistinguishable from "these seeds are easy".
#   2. GD FAILS           -> the L2 objective ALONE provably does not solve it
#                            (GD is exactly that objective). So if swapping V
#                            for the objective still succeeds, the heat solve
#                            is decorative; if it fails, V carries the work.
#   3. cumKL is LARGE     -> the tilt actually fired in the real run. You cannot
#                            ablate a mechanism that never acted; real seeds 3
#                            (H15) and 1 (H5) sit at cumKL 0.01-0.07.
# Picked 2026-09-11 by ranking (1 AND 2) on cumKL from sweep_fk/table/pointmaze.
# NOTE H15 runs np=2, where per-resample selection KL is capped at log 2 = 0.69
# nats, so every candidate ties there and the ranking is arbitrary by design.
seeds_for () {
  case $1 in
    5)  echo "20 92 90 6 54" ;;   # cumKL 1.65 1.38 0.98 0.90 0.87
    10) echo "85 89 56 12 25" ;;  # cumKL 2.10 1.67 1.66 1.51 1.28
    15) echo "98 91 87 79 68" ;;  # all at the np=2 ceiling of 0.69
  esac
}
STEPS=150
PER_RUN_TIMEOUT=1200
MAX_TRIES=3

particles_for () { echo 2; }   # every horizon, see BUDGET note above

# GPU gate, as in run_fk_sweep.sh. Without it these runs OOM whenever the shared
# card has a co-tenant, and the script marches on leaving MISSING cells -- which
# in a 5-seed ablation reads as a result rather than as a gap.
# np=3 at goal_H=10 measured OK at ~8.5 GB free, OOM at ~6.3 GB.
NEED_MIB=${NEED_MIB:-5000}
GATE_TIMEOUT=${GATE_TIMEOUT:-7200}
gpu_free () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU; }
wait_for_gpu () {
    local waited=0
    while [ "$(gpu_free)" -lt "$NEED_MIB" ]; do
        [ "$waited" -ge "$GATE_TIMEOUT" ] && { echo "=== GATE TIMEOUT (free=$(gpu_free)) ==="; return 1; }
        sleep 30; waited=$((waited + 30))
    done
    return 0
}

for mode in none objective objective_raw; do
  for H in ${HORIZONS:-5 10 15}; do
    np=$(particles_for $H)
    for s in $(seeds_for $H); do
      OUT=$ROOT/${mode}_H${H}/seed_$s
      [ -s "$OUT/result.txt" ] && continue
      mkdir -p "$OUT"
      for try in $(seq 1 $MAX_TRIES); do
        wait_for_gpu || { echo "[ablate $mode H$H s$s] ABANDONED (gate)"; break; }
        t0=$(date +%s)
        GRASP_RESULTS_DIR="$OUT" timeout $PER_RUN_TIMEOUT python plan.py \
            model_name=point_maze n_evals=1 planner=fk_pointmaze \
            goal_H=$H goal_source=random_state seed=$s save_images=False \
            planner.opt_steps=$STEPS planner.eval_every=1 \
            planner.num_particles=$np planner.num_smooth_samples=1 \
            planner.v_null=$mode \
            ckpt_base_path=$PWD/checkpoints \
            > "$OUT/run.log" 2>&1
        dt=$(( $(date +%s) - t0 ))
        if grep -q "Success rate:" "$OUT/run.log" 2>/dev/null; then
            ok=0; grep -q "Success rate:  1.0" "$OUT/run.log" && ok=1
            echo "$ok $dt" > "$OUT/result.txt"
            echo "[ablate $mode H$H s$s np=$np] ${dt}s success=$ok"
            break
        fi
        if grep -q "OutOfMemoryError" "$OUT/run.log"; then
            echo "[ablate $mode H$H s$s] OOM try $try"; sleep 20
        else
            echo "[ablate $mode H$H s$s] FAILED: $(grep -oE '[A-Za-z]*Error[^\"]*' $OUT/run.log|head -1|cut -c1-90)"; break
        fi
      done
    done
    echo "=== DONE ablate $mode H$H $(date +%T) ==="
  done
done
echo "=== ABLATION COMPLETE $(date +%T) ==="
