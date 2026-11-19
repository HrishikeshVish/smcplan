#!/bin/bash
# CUBE under LeFlow's protocol (github.com/hsiangwei0903/LeFlow, config/eval/cube.yaml).
#
# PROTOCOL, matched to the reference:
#   goal_H=5          -- LeFlow plan_config.horizon=5 x action_block=5 = 25 env
#                        steps, which equals its eval.goal_offset_steps=25.
#                        Cube is therefore NOT run on the shared H in {5,10,15}
#                        grid: H=15 would be 75 env steps against a 50-step
#                        episode cap.
#   goal_source=dset  -- goal is the state 25 env steps later in an expert
#                        episode. LeFlow samples start rows UNIFORMLY over all
#                        valid ones (eval.py:108-129) with NO filter on whether
#                        the block moves.
#   50 seeds          -- LeFlow eval.num_eval=50.
#   mpc_cem           -- LeFlow solver: receding-horizon CEM. Its
#                        receding_horizon=5 is our n_taken_actions=5, and its
#                        CEM block (num_samples 300 / n_steps 30 / topk 30)
#                        matches conf/planner/mpc_cem.yaml exactly.
#
# KNOWN CAVEAT, to be stated in the caption: 38.3% of windows under this
# protocol are TRIVIAL -- the block moves <= the 0.04 m success radius, so the
# goal is already satisfied at t=0 and any planner scores them. Measured over
# 2651 sampled windows; median block displacement 0.0943 m. This is a property
# of the reference benchmark, not of our sampling. Report the trivial fraction
# alongside the success rate, or these numbers will read as ~38% free.
#
# Success criterion is now the reference one: ||qpos[14:17] - goal[14:17]|| <= 0.04
# (env/dmcontrol/dmcontrol_env_wrapper.py, matching cube_env.py:1318-1344).
# qpos[14:17] was verified identical to the dataset's privileged_block_0_pos.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU DATASET_DIR=$PWD/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl; unset DISPLAY

ROOT=${ROOT:-$PWD/sweep_cube_leflow}
SEEDS=${SEEDS:-$(seq 1 50)}
PER_RUN_TIMEOUT=2400

planner_args () {
    case $1 in
        mpc_cem) echo "planner=mpc_cem" ;;                      # LeFlow's own solver
        gd)      echo "planner=gd_cube planner.opt_steps=150 planner.eval_every=1" ;;
        grasp)   echo "planner=grasp planner.opt_steps=150 planner.eval_every=1" ;;
        # FK (ours). beta_max=10 rather than the default 1: the 2026-09-15 tuning
        # grid showed the tilt is otherwise near-inert on cube (cumKL 0.14,
        # ESS 28.6/32). beta_max=10 raises it to cumKL 0.71 / ESS 19.2. NOTE it
        # did NOT change the outcome (4/5 non-trivial in every arm), so do not
        # claim the roadmap V is doing the work on cube -- see the near-zero
        # V-vs-distance Spearman (0.003) in the same runs.
        fk)      echo "planner=fk_cube planner.opt_steps=150 planner.eval_every=1 planner.beta_max=10.0" ;;
    esac
}

for p in ${PLANNERS:-mpc_cem gd grasp fk}; do
  for s in $SEEDS; do
    OUT=$ROOT/$p/seed_$s
    [ -s "$OUT/result.txt" ] && continue
    mkdir -p "$OUT"
    t0=$(date +%s)
    GRASP_TRAJ_DUMP="$OUT/traj.npz" GRASP_RESULTS_DIR="$OUT" timeout $PER_RUN_TIMEOUT python plan.py \
        model_name=lewm_cube n_evals=1 goal_H=5 goal_source=dset seed=$s \
        save_images=False $(planner_args $p) \
        ckpt_base_path=$PWD/checkpoints \
        > "$OUT/run.log" 2>&1
    dt=$(( $(date +%s) - t0 ))
    if grep -q "Success rate:" "$OUT/run.log" 2>/dev/null; then
        ok=0; grep -q "Success rate:  1.0" "$OUT/run.log" && ok=1
        echo "$ok $dt" > "$OUT/result.txt"
        echo "[cube $p s$s] ${dt}s success=$ok"
    else
        echo "[cube $p s$s] FAILED ${dt}s: $(grep -oE '[A-Za-z]*Error[^\"]*' $OUT/run.log|head -1|cut -c1-80)"
    fi
  done
  echo "=== DONE cube $p $(date +%T) ==="
done
echo "=== CUBE LEFLOW SWEEP COMPLETE $(date +%T) ==="
