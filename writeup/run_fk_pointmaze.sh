#!/bin/bash
# FK pointmaze: extend the gh10 tilt/no-tilt A/B from 60 -> 100 seeds.
#
# Provenance (pm_fk_gh10/queue_fk_after_gd.sh + run_control.sh), reproduced
# exactly so seeds 61-100 are commensurable with the existing 1-60:
#   planner=fk_pointmaze goal_H=10 goal_source=random_state
#   planner.opt_steps=150 planner.eval_every=1
#   planner.num_particles=3 planner.num_smooth_samples=1
#   +planner.sample_type=randn planner.action_reg=0
# The no-tilt arm adds planner.beta_max=0. That override is inferred, not
# found in a script: pm_fk_notilt_flip and pm_fk_vanillacfg are identical at
# step 0 on the same seed (L_dyn=1.7408, V_sel=3.6568, pop 3.94, dmin=0.5200)
# and differ only in beta (0.000 vs 1.000).
#
# Resumable: unlike run_control.sh, a seed with a non-empty seed_success_log.txt
# is SKIPPED, not re-run.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grasp
cd /home/hviswan/Documents/grasp
unset MUJOCO_GL                      # pointmaze: MUJOCO_GL must be unset
GPU=${GPU:-1}
export CUDA_VISIBLE_DEVICES=$GPU DATASET_DIR=/home/hviswan/Documents/grasp/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NEED_MIB=${NEED_MIB:-6500}   # 3 particles; lower than the 9000 baseline gate
MAX_TRIES=3
GATE_TIMEOUT=${GATE_TIMEOUT:-1800}
SEEDS=${SEEDS:-$(seq 61 100)}

gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU; }
wait_for_gpu() {
    local waited=0
    while [ "$(gpu_free)" -lt "$NEED_MIB" ]; do
        [ "$waited" -ge "$GATE_TIMEOUT" ] && { echo "=== GATE TIMEOUT (free=$(gpu_free)) ==="; return 1; }
        sleep 30; waited=$((waited + 30))
    done
    return 0
}

run_arm () {                          # outdir  extra-overrides
    local BASE=$(readlink -m "$1") EXTRA="$2"
    mkdir -p "$BASE"
    echo "=== ARM START $(date +%T) : $(basename $BASE) : [$EXTRA] ==="
    for s in $SEEDS; do
        local OUT=$BASE/seed_$s
        [ -s "$OUT/seed_success_log.txt" ] && continue
        mkdir -p "$OUT"
        for try in $(seq 1 $MAX_TRIES); do
            wait_for_gpu || { echo "=== seed $s ABANDONED ==="; break; }
            GRASP_RESULTS_DIR="$OUT" python plan.py model_name=point_maze n_evals=1 \
                planner=fk_pointmaze goal_H=10 goal_source=random_state seed=$s \
                save_images=False planner.opt_steps=150 planner.eval_every=1 \
                planner.num_particles=3 planner.num_smooth_samples=1 \
                +planner.sample_type=randn planner.action_reg=0 $EXTRA \
                ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints \
                > "$OUT/run.log" 2>&1
            if [ -s "$OUT/seed_success_log.txt" ]; then
                echo "=== [$(basename $BASE)] seed $s done $(date +%T): $(cat $OUT/seed_success_log.txt)"
                break
            fi
            if grep -q "OutOfMemoryError" "$OUT/run.log"; then
                echo "=== [$(basename $BASE)] seed $s OOM try $try, retrying ==="; sleep 20
            else
                echo "=== [$(basename $BASE)] seed $s FAILED (non-OOM) try $try ==="
                tail -3 "$OUT/run.log"; break
            fi
        done
    done
    echo "=== ARM COMPLETE $(date +%T) : $(basename $BASE) ==="
}

run_arm pm_fk_vanillacfg   ''
run_arm pm_fk_notilt_flip  'planner.beta_max=0'
echo "=== FK POINTMAZE COMPLETE $(date +%T) ==="
