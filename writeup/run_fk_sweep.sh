#!/bin/bash
# FK (ours) arm of the planner-comparison table: H in {5,10,15} x 100 seeds,
# to sit alongside the {gd, grasp, cem} rows produced by run_sweep.sh.
#
# PARTICLE COUNT IS NOT UNIFORM ACROSS ENVS, and the table must say so.
#   reacher   -> 8 particles x 8 smoothing. LeWM encodes each frame as a
#                SINGLE 192-D token, so concurrent rollouts are cheap; 8p is
#                the config that scored 19/20 on the H10 probe (32p scored
#                12/12 but at 38,400 rollouts, deliberately off-table).
#   pointmaze -> 3 particles x 1 smoothing (NO Gaussian smoothing: S=1 takes
#   pusht        the S=1 branch in fk_planner). These are the DINO-WM path,
#                where encode_obs yields (1, 3, 196, 384) -- 196 ViT patch
#                tokens per frame -- and the rollout is retained for autograd,
#                so cost is quadratic in patch tokens. Measured ceiling on a
#                16 GB 5080 at goal_H=10, S=1: np=8 OOM, np=4 OOM, np=3 OK
#                (2026-09-09, GPU1 sharing with a 6.7 GB co-tenant; np=8 also
#                OOM'd on a near-empty card 2026-08-30). This is a hardware
#                limit, not a tuning choice -- label these rows reduced-budget.
#
# Goal source is per-env, copied from run_sweep.sh so the FK rows are
# protocol-matched to the baselines: pusht -> dset (its random_state pairs are
# routinely unconnectable and floored every planner at ~0/100), reacher and
# pointmaze -> random_state (the LeWM loaders double-apply frameskip, capping
# dset at goal_H <= 8, so H=10/15 are unreachable there).
#
# Envs run in the order given on the command line and each env completes
# before the next starts. Resumable: a seed whose result.txt is non-empty is
# skipped, so re-running after an interrupt picks up where it stopped.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
GPU=${GPU:-1}
export CUDA_VISIBLE_DEVICES=$GPU DATASET_DIR=$PWD/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=${ROOT:-$PWD/sweep_fk/table}
SEEDS=${SEEDS:-$(seq 1 100)}
HORIZONS=${HORIZONS:-"5 10 15"}
STEPS=${STEPS:-150}
NEED_MIB=${NEED_MIB:-7000}
GATE_TIMEOUT=${GATE_TIMEOUT:-3600}
PER_RUN_TIMEOUT=${PER_RUN_TIMEOUT:-2400}
MAX_TRIES=${MAX_TRIES:-3}

gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU; }
wait_for_gpu() {
    local waited=0
    while [ "$(gpu_free)" -lt "$NEED_MIB" ]; do
        [ "$waited" -ge "$GATE_TIMEOUT" ] && { echo "=== GATE TIMEOUT (free=$(gpu_free)) ==="; return 1; }
        sleep 30; waited=$((waited + 30))
    done
    return 0
}

# PARTICLES/SMOOTH env vars override the per-env defaults below. Needed because
# the ceiling is horizon-dependent, not just env-dependent: the retained rollout
# grows with H, so goal_H=15 OOMs at the np=3 that goal_H=10 tolerates. Measured
# 2026-09-10 on GPU1 (~9.3 GB free): pointmaze and pusht both OOM at np=3 / H15
# and run at np=2. Invoke that arm as
#   HORIZONS=15 PARTICLES=2 bash writeup/run_fk_sweep.sh <env>
run_env () {                  # name model planner gl goal_source particles smooth
    local ename=$1 model=$2 pl=$3 gl=$4 gsrc=$5 np=${PARTICLES:-$6} sm=${SMOOTH:-$7}
    if [ "$gl" = "egl" ]; then export MUJOCO_GL=egl; unset DISPLAY
    else unset MUJOCO_GL; export DISPLAY=:2; fi
    for H in $HORIZONS; do
        for s in $SEEDS; do
            local OUT=$ROOT/$ename/fk_H${H}/seed_$s
            [ -s "$OUT/result.txt" ] && continue
            mkdir -p "$OUT"
            for try in $(seq 1 $MAX_TRIES); do
                wait_for_gpu || { echo "=== ABANDONED $ename H$H seed $s (gate) ==="; break; }
                local t0=$(date +%s)
                GRASP_RESULTS_DIR="$OUT" timeout $PER_RUN_TIMEOUT python plan.py \
                    model_name=$model n_evals=1 planner=$pl \
                    goal_H=$H goal_source=$gsrc seed=$s save_images=False \
                    planner.opt_steps=$STEPS planner.eval_every=1 \
                    planner.num_particles=$np planner.num_smooth_samples=$sm \
                    ckpt_base_path=$PWD/checkpoints \
                    > "$OUT/run.log" 2>&1
                local dt=$(( $(date +%s) - t0 ))
                # Harvest from run.log, as run_sweep.sh does: uniform across
                # planners and independent of who writes seed_success_log.txt.
                if grep -q "Success rate:" "$OUT/run.log" 2>/dev/null; then
                    local ok=0
                    grep -q "Success rate:  1.0" "$OUT/run.log" && ok=1
                    echo "$ok $dt" > "$OUT/result.txt"
                    echo "[$ename fk H$H s$s] ${dt}s success=$ok"
                    break
                fi
                if grep -q "OutOfMemoryError" "$OUT/run.log"; then
                    echo "[$ename fk H$H s$s] OOM try $try (np=$np sm=$sm), retrying"; sleep 20
                else
                    echo "[$ename fk H$H s$s] FAILED ${dt}s: $(grep -oE '[A-Za-z]*Error[^\"]*' $OUT/run.log | head -1 | cut -c1-90)"
                    break
                fi
            done
        done
        echo "=== DONE $ename fk H$H $(date +%T) ==="
    done
    echo "=== ENV COMPLETE $ename $(date +%T) ==="
}

for spec in "$@"; do
    case $spec in
        reacher)   run_env reacher   lewm_reacher fk           egl  random_state 8 8 ;;
        pointmaze) run_env pointmaze point_maze   fk_pointmaze none random_state 3 1 ;;
        pusht)     run_env pusht     pusht        fk_pusht     none dset         3 1 ;;
        # Wall: DINO-WM like pointmaze/pusht, so the same 3-particle ceiling.
        # Buffer verified on-manifold (dmin 0.43) before this was added.
        wall)      run_env wall      wall_single  fk_wall      none random_state 3 1 ;;
        *) echo "unknown env: $spec" ;;
    esac
done
echo "=== FK SWEEP COMPLETE $(date +%T) ==="
