#!/bin/bash
# FK-on-cube tuning grid, 6 seeds x 4 configs.
#
# WHAT WE ARE FIXING. The first fk_cube smoke run planned fine but its tilt was
# effectively INERT: final ESS 29.4/32 (weights ~uniform), one resample,
# cumKL 0.12 nats, and V-vs-latent-distance Spearman 0.0031. Queries also sat
# further off-manifold than on working envs: dmin/sigma_perturb = 1.74 here vs
# 0.92 (reacher) and 0.61 (pusht). So V carries little signal and FK degenerates
# to best-of-N GD -- it may still SOLVE seeds, but not for the claimed reason.
#
# THE FOUR ARMS, each targeting a different hypothesis:
#   base    the current config, as a control.
#   beta    beta_max 1 -> 10. beta is ESS-adaptive but CAPPED at beta_max, so if
#           V's spread across particles is small the exponent saturates and the
#           tilt stays weak. Raising the cap is the direct remedy for a weak
#           tilt that is NOT caused by bad V ordering.
#   dense   224k-node buffer (5600 episodes) with graph_n_nodes 150k. Cube
#           explores a 41-D state space against reacher's 4-D, so 56k nodes are
#           a far sparser cover -- the likely cause of the high dmin.
#   rw      solver_mode rw_diffusion instead of heat. On reacher this scored
#           Spearman 0.60 vs heat's 0.29, so it is the better solver where the
#           heat resolvent is flat.
#
# SEEDS: 4,5,7,8 are known NON-TRIVIAL (block displacement 0.242/0.133/0.041/
# 0.045 m); 9 and 10 are added to reach six and their displacement is recorded
# from the traj dump so trivial ones can be excluded at analysis time. Scoring
# on trivial goals is meaningless -- they are solved at t=0 by any planner.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export CUDA_VISIBLE_DEVICES=${GPU:-0} DATASET_DIR=$PWD/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MUJOCO_GL=egl; unset DISPLAY

ROOT=$PWD/sweep_fk/cube_tune
SEEDS=${SEEDS:-"4 5 7 8 9 10"}
STEPS=${STEPS:-150}

args_for () {
  case $1 in
    base)  echo "" ;;
    beta)  echo "planner.beta_max=10.0" ;;
    dense) echo "planner.replay_buffer_path=$PWD/latent_buffers/lewm_cube_dense.pt planner.graph_n_nodes=150000" ;;
    rw)    echo "planner.solver_mode=rw_diffusion" ;;
  esac
}

for cfg in ${CFGS:-base beta dense rw}; do
  for s in $SEEDS; do
    OUT=$ROOT/$cfg/seed_$s
    [ -s "$OUT/result.txt" ] && continue
    mkdir -p "$OUT"
    t0=$(date +%s)
    GRASP_TRAJ_DUMP="$OUT/traj.npz" GRASP_RESULTS_DIR="$OUT" timeout 2400 python plan.py \
        model_name=lewm_cube n_evals=1 planner=fk_cube goal_H=5 goal_source=dset \
        seed=$s save_images=False planner.opt_steps=$STEPS planner.eval_every=1 \
        $(args_for $cfg) \
        ckpt_base_path=$PWD/checkpoints > "$OUT/run.log" 2>&1
    dt=$(( $(date +%s) - t0 ))
    if grep -q "Success rate:" "$OUT/run.log" 2>/dev/null; then
        ok=0; grep -q "Success rate:  1.0" "$OUT/run.log" && ok=1
        echo "$ok $dt" > "$OUT/result.txt"
        echo "[tune $cfg s$s] ${dt}s ok=$ok $(grep -oE 'cumKL=[0-9.]+ nats' $OUT/run.log|tail -1) $(grep -oE 'final ESS=[0-9./]+' $OUT/run.log|tail -1) $(grep -oE 'dmin=[0-9.]+' $OUT/run.log|tail -1)"
    else
        echo "[tune $cfg s$s] FAILED ${dt}s: $(grep -oE '[A-Za-z]*Error[^\"]*' $OUT/run.log|head -1|cut -c1-70)"
    fi
  done
  echo "=== DONE tune $cfg $(date +%T) ==="
done
echo "=== FK CUBE TUNE COMPLETE $(date +%T) ==="
