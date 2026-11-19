#!/bin/bash
# Planner comparison sweep: {gd, grasp, cem} x H in {5,10,15} x 100 seeds.
#
# Budget: all planners get 150 optimisation steps. CEM is additionally tuned to
# 150 world-model rollouts (num_samples * opt_steps = 30 * 5) so it matches GD's
# rollout count, since stock cem.yaml (300 x 30 = 9000) would give it 60x the
# compute. Per-run wall clock is logged so the wall-clock view can be reported
# alongside the rollout-matched one.
#
# Goal source: random_state, not dset. The LeWM loaders (dsets/dmcontrol_dset)
# already subsample episodes by frameskip -- reacher's 201-frame episodes become
# 41-frame items -- while plan.py then computes traj_len = frameskip*goal_H+1,
# applying frameskip a second time. That caps dset at goal_H <= 8 on reacher and
# cube, so H=10 and H=15 are unreachable there. random_state needs only a
# 2-frame segment and works at every horizon on every env, giving one uniform
# protocol across the whole table.
#
# Resumable: a seed whose result.txt already exists is skipped.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU DATASET_DIR=$PWD/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=$PWD/sweep
NEED_MIB=9000
GATE_TIMEOUT=3600
SEEDS=${SEEDS:-$(seq 1 100)}
HORIZONS=${HORIZONS:-"5 10 15"}
ROOT=${ROOT_OVERRIDE:-$ROOT}
PER_RUN_TIMEOUT=1200

gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU; }
wait_for_gpu() {
    local waited=0
    while [ "$(gpu_free)" -lt "$NEED_MIB" ]; do
        [ "$waited" -ge "$GATE_TIMEOUT" ] && { echo "=== GATE TIMEOUT (free=$(gpu_free)) ==="; return 1; }
        sleep 30; waited=$((waited + 30))
    done
    return 0
}

# per-planner overrides at a matched 150-step / 150-rollout budget
planner_args () {
    case $1 in
        gd)    echo "planner=gd planner.opt_steps=150 planner.eval_every=1 planner.hjb_tau=0.0" ;;
        grasp) echo "planner=grasp planner.opt_steps=150 planner.eval_every=1" ;;
        cem)   echo "planner=cem planner.opt_steps=5 planner.num_samples=30 planner.topk=3 planner.eval_every=1" ;;
    esac
}

run_env () {                       # name model mujoco_gl
    local ename=$1 model=$2 gl=$3
    if [ "$gl" = "egl" ]; then export MUJOCO_GL=egl; unset DISPLAY
    else unset MUJOCO_GL; export DISPLAY=:2; fi
    for planner in gd grasp cem; do
        for H in $HORIZONS; do
            for s in $SEEDS; do
                OUT=$ROOT/$ename/${planner}_H${H}/seed_$s
                # result.txt is written by this script, not the planner:
                # planning/grasp.py never writes seed_success_log.txt, so keying
                # resume on that file would re-run every GRASP seed forever.
                [ -s "$OUT/result.txt" ] && continue
                mkdir -p "$OUT"
                wait_for_gpu || { echo "=== ABANDONED $ename $planner H$H seed $s ==="; continue; }
                local t0=$(date +%s)
                GRASP_RESULTS_DIR="$OUT" timeout $PER_RUN_TIMEOUT python plan.py \
                    model_name=$model n_evals=1 goal_H=$H goal_source=random_state seed=$s \
                    save_images=False $(planner_args $planner) \
                    ckpt_base_path=$PWD/checkpoints \
                    > "$OUT/run.log" 2>&1
                local dt=$(( $(date +%s) - t0 ))
                # Harvest uniformly from run.log: success iff the evaluator ever
                # reported 1.0, which matches gd.py's early-exit-on-success
                # semantics and works for every planner including GRASP.
                if grep -q "Success rate:" "$OUT/run.log" 2>/dev/null; then
                    local ok=0
                    grep -q "Success rate:  1.0" "$OUT/run.log" && ok=1
                    echo "$ok $dt" > "$OUT/result.txt"
                    echo "[$ename $planner H$H s$s] ${dt}s success=$ok"
                else
                    echo "[$ename $planner H$H s$s] FAILED ${dt}s: $(grep -oE '[A-Za-z]*Error[^\"]*' $OUT/run.log | head -1 | cut -c1-90)"
                fi
            done
            echo "=== DONE $ename $planner H$H $(date +%T) ==="
        done
    done
    echo "=== ENV COMPLETE $ename $(date +%T) ==="
}

for spec in "$@"; do
    case $spec in
        reacher)   run_env reacher   lewm_reacher egl ;;
        pointmaze) run_env pointmaze point_maze   none ;;
        cube)      run_env cube      lewm_cube    egl ;;
        pusht)     run_env pusht     pusht        none ;;
        wall)      run_env wall      wall_single  none ;;
    esac
done
echo "=== SWEEP COMPLETE $(date +%T) ==="
