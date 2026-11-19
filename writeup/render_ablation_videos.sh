#!/bin/bash
# Re-run the seeds where the two ablation arms DIVERGE, with the official
# DINO-WM video writer enabled (save_images=True -> planning/evaluator.py:239,
# decoded through the pointmaze VQVAE) and with GRASP_TRAJ_DUMP set so we also
# get the true executed (x,y) path for a 2D maze trace.
#
# Seeds: 23 (ours solves in 5 steps, L2 objective FAILS at the cap),
#        19 (3 vs 82 steps), 6 (4 vs 11 steps).
# Everything else matches the ablation exactly: np=3, H=10, 150 steps.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU DATASET_DIR=$PWD/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset MUJOCO_GL; export DISPLAY=:2

ROOT=${ROOT:-$PWD/sweep_fk/renders}
SEEDS=${SEEDS:-"23 19 6"}
for mode in none objective_raw; do
  for s in $SEEDS; do
    OUT=$ROOT/${mode}/seed_$s
    [ -s "$OUT/traj.npz" ] && { echo "[render $mode s$s] cached"; continue; }
    mkdir -p "$OUT"
    t0=$(date +%s)
    GRASP_RESULTS_DIR="$OUT" GRASP_TRAJ_DUMP="$OUT/traj.npz" \
    timeout 1800 python plan.py \
        model_name=point_maze n_evals=1 planner=fk_pointmaze \
        goal_H=10 goal_source=random_state seed=$s save_images=True \
        planner.opt_steps=150 planner.eval_every=1 \
        planner.num_particles=3 planner.num_smooth_samples=1 \
        planner.v_null=$mode \
        ckpt_base_path=$PWD/checkpoints \
        > "$OUT/run.log" 2>&1
    # The official writer uses a bare relative filename, so the mp4 lands in the
    # Hydra run dir, not GRASP_RESULTS_DIR. Recover it from the logged path.
    HYD=$(grep -oE "Planning result saved dir: .*" "$OUT/run.log" | head -1 | sed 's/.*saved dir: //')
    if [ -n "$HYD" ]; then
        for f in "$HYD"/output_final_*.mp4; do [ -e "$f" ] && cp "$f" "$OUT/rollout.mp4"; done
        [ -e "$HYD/output_final.png" ] && cp "$HYD/output_final.png" "$OUT/rollout.png"
    fi
    echo "[render $mode s$s] $(( $(date +%s) - t0 ))s  traj=$([ -s $OUT/traj.npz ] && echo yes || echo NO)  mp4=$([ -s $OUT/rollout.mp4 ] && echo yes || echo NO)"
  done
done
echo "=== RENDERS COMPLETE $(date +%T) ==="
