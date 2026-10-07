#!/bin/bash
# FK reacher under the BASELINE TABLE protocol: goal_source=random_state,
# H in {5,10,15}, 100 seeds -- so FK is directly comparable to the gd/grasp/cem
# rows in sweep/reacher/.
#
# This deliberately does NOT reuse fk_sweep_v1/ (40 seeds, 38/40=0.95): that
# ran goal_source=dset, which is a different and easier protocol and cannot be
# put next to the random_state baselines. dset is also capped at goal_H<=8 on
# reacher (dsets/dmcontrol_dset pre-subsamples by frameskip, then plan.py
# applies frameskip again in traj_len), so it cannot produce H=10 or H=15 at all.
#
# BUDGET. run_sweep.sh equalises WORLD-MODEL ROLLOUTS, not opt steps: gd and
# grasp use 1 trajectory x 150 steps = 150 rollouts, and cem was deliberately
# detuned (30 samples x 5 steps) to match. fk.yaml's defaults would spend
# 32 particles x 8 smoothing x 150 steps = 38,400 rollouts = 256x the baseline,
# which is not a comparison anyone can defend -- an earlier arm at those
# defaults scored 50/50 at H=5 and is archived under
# sweep_fk/_reacher_budget256x/ for reference.
# We therefore override to 4 particles x 2 smoothing x 50 steps = 400 rollouts
# (2.7x baseline). 50 steps rather than 40 because observed exit steps reached
# 83; truncating at 40 would discard successes the method genuinely finds.
#
# Resumable: a seed with a non-empty seed_success_log.txt is skipped.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grasp
cd /home/hviswan/Documents/grasp
export MUJOCO_GL=egl                 # reacher: required
unset DISPLAY
GPU=${GPU:-1}
export CUDA_VISIBLE_DEVICES=$GPU DATASET_DIR=/home/hviswan/Documents/grasp/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=${ROOT:-/home/hviswan/Documents/grasp/sweep_fk/reacher}
NEED_MIB=${NEED_MIB:-7000}
MAX_TRIES=3
GATE_TIMEOUT=${GATE_TIMEOUT:-1800}
PER_RUN_TIMEOUT=${PER_RUN_TIMEOUT:-1800}
SEEDS=${SEEDS:-$(seq 1 100)}
PARTICLES=${PARTICLES:-4}
SMOOTH=${SMOOTH:-2}
STEPS=${STEPS:-50}
HORIZONS=${HORIZONS:-"5 10 15"}

gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU; }
wait_for_gpu() {
    local waited=0
    while [ "$(gpu_free)" -lt "$NEED_MIB" ]; do
        [ "$waited" -ge "$GATE_TIMEOUT" ] && { echo "=== GATE TIMEOUT (free=$(gpu_free)) ==="; return 1; }
        sleep 30; waited=$((waited + 30))
    done
    return 0
}

for H in $HORIZONS; do
    for s in $SEEDS; do
        OUT=$ROOT/fk_H${H}/seed_$s
        [ -s "$OUT/seed_success_log.txt" ] && continue
        mkdir -p "$OUT"
        for try in $(seq 1 $MAX_TRIES); do
            wait_for_gpu || { echo "=== [fk H$H] seed $s ABANDONED ==="; break; }
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
                echo "[reacher fk H$H s$s] ${dt}s $(cat $OUT/seed_success_log.txt)"
                break
            fi
            if grep -q "OutOfMemoryError" "$OUT/run.log"; then
                echo "[reacher fk H$H s$s] OOM try $try, retrying"; sleep 20
            else
                echo "[reacher fk H$H s$s] FAILED (non-OOM) try $try ${dt}s: $(grep -oE '[A-Za-z]*Error[^\"]*' $OUT/run.log | head -1 | cut -c1-90)"
                tail -3 "$OUT/run.log"; break
            fi
        done
    done
    echo "=== DONE reacher fk H$H $(date +%T) ==="
done
echo "=== FK REACHER COMPLETE $(date +%T) ==="
