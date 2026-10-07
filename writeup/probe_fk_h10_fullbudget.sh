#!/bin/bash
# H10 UPPER-BOUND PROBE: FK at fk.yaml's full defaults, to test whether the
# H10 shortfall is an optimization problem at all.
#
# WHY. The budget-matched arm (4 particles x 2 smoothing x 50 steps) scores
# 13/39 at H10 vs 93/100 at H5. Diagnostics over those 39 runs:
#   - ESS = 1.0-3.3 of 4 particles, beta pinned at 1.0 by step ~11, typically
#     ONE resample in 50 steps -> the SMC population is inert, effectively
#     single-chain GD with a saturated tempering schedule.
#   - Failures split by final V_sel: 12 never converged (V_sel 3.5-17.3),
#     14 converged to V_sel < 1 -- the same range as the 12 successes -- and
#     still failed in the environment.
#   - dmin (distance to nearest buffer node; fk_planner.py:126) sits at
#     1.03-1.35 for successes and failures alike, so plans stay on-manifold
#     and dmin does not discriminate. It is not a goal distance.
# Only the 12 non-converged failures are reachable by more particles/steps,
# so the tuning ceiling here is ~23/38 = 0.61. This probe measures how much
# of that is actually recoverable.
#
# BUDGET. Deliberately UNFAIR vs the gd/grasp/cem rows (38,400 rollouts vs
# 150). This is a diagnostic upper bound, NOT a table row -- it goes next to
# the matched arm only as an ablation, never next to the baselines.
# Wall-clock is nearly free: particles batch on GPU (matched 137 s/seed vs
# archived 32p 158 s/seed), but H10 failures will run the full step cap,
# unlike the archived H5 arm which always exited by step 110.
#
# opt_steps=150 not fk.yaml's 300: caps a failing seed at ~13 min.
# Seeds 1-20 overlap the matched arm's H10 seeds for a PAIRED comparison
# (matched scores 6/20 on exactly these seeds).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grasp
cd /home/hviswan/Documents/grasp
export MUJOCO_GL=egl
unset DISPLAY
GPU=${GPU:-1}
export CUDA_VISIBLE_DEVICES=$GPU DATASET_DIR=/home/hviswan/Documents/grasp/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=${ROOT:-/home/hviswan/Documents/grasp/sweep_fk/reacher_probe_H10_fullbudget}
NEED_MIB=${NEED_MIB:-7000}
MAX_TRIES=3
GATE_TIMEOUT=${GATE_TIMEOUT:-1800}
PER_RUN_TIMEOUT=${PER_RUN_TIMEOUT:-1800}
SEEDS=${SEEDS:-$(seq 1 20)}
PARTICLES=${PARTICLES:-32}
SMOOTH=${SMOOTH:-8}
STEPS=${STEPS:-150}
H=${H:-10}

gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU; }
wait_for_gpu() {
    local waited=0
    while [ "$(gpu_free)" -lt "$NEED_MIB" ]; do
        [ "$waited" -ge "$GATE_TIMEOUT" ] && { echo "=== GATE TIMEOUT (free=$(gpu_free)) ==="; return 1; }
        sleep 30; waited=$((waited + 30))
    done
    return 0
}

for s in $SEEDS; do
    OUT=$ROOT/fk_H${H}/seed_$s
    [ -s "$OUT/seed_success_log.txt" ] && continue
    mkdir -p "$OUT"
    for try in $(seq 1 $MAX_TRIES); do
        wait_for_gpu || { echo "=== [probe H$H] seed $s ABANDONED ==="; break; }
        t0=$(date +%s)
        GRASP_RESULTS_DIR="$OUT" timeout $PER_RUN_TIMEOUT python plan.py \
            model_name=lewm_reacher n_evals=1 planner=fk \
            goal_H=$H goal_source=random_state seed=$s \
            save_images=False planner.opt_steps=$STEPS planner.eval_every=1 \
            planner.num_particles=$PARTICLES planner.num_smooth_samples=$SMOOTH \
            ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints \
            > "$OUT/run.log" 2>&1
        dt=$(( $(date +%s) - t0 ))
        if [ -s "$OUT/seed_success_log.txt" ]; then
            echo "[probe fk H$H s$s] ${dt}s $(cat $OUT/seed_success_log.txt)"
            break
        fi
        if grep -q "OutOfMemoryError" "$OUT/run.log"; then
            echo "[probe fk H$H s$s] OOM try $try, retrying"; sleep 20
        else
            echo "[probe fk H$H s$s] FAILED (non-OOM) try $try ${dt}s: $(grep -oE '[A-Za-z]*Error[^"]*' $OUT/run.log | head -1 | cut -c1-90)"
            tail -3 "$OUT/run.log"; break
        fi
    done
done
echo "=== PROBE COMPLETE $(date +%T) ==="
