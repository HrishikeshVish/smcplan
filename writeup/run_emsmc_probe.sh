#!/bin/bash
# First real test of planning/em_smc.py: the EM planner (SMC E-step with a
# pi-invariant kernel + Adam M-step driven by the SMC weights) against the
# recorded FKSMCPlanner baseline in sweep_fk/table/pointmaze/fk_H5, same seeds,
# same budget, same population size.
#
# Two arms, to separate the two M-step gradients:
#   both   pathwise (world-model gradient, GRASP) + score (MPPI's weighted mean)
#   score  score only -- MPPI with MCMC rejuvenation, no world-model gradient
# ARM and GPU come from the environment.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset MUJOCO_GL; export DISPLAY=:0
ARM=${ARM:-both}
GPU=${GPU:-1}
# Per-arm overrides. Default: the arm name is the m_step mode, smoothing off
# (matching fk_pointmaze's num_smooth_samples=1 in the baseline sweep).
ARGS=${ARGS:-"planner.m_step=$ARM planner.m_smooth_samples=1"}
ROOT=$PWD/sweep_emsmc/pointmaze_H5/$ARM
CK=ckpt_base_path=$PWD/checkpoints
SEEDS=${SEEDS:-"1 10 100 11 12 13 14 15 16 17 18 19 2 20 21 22 23 24 25 26"}
for s in $SEEDS; do
  OUT=$ROOT/seed_$s
  [ -s "$OUT/result.txt" ] && continue
  mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=$GPU GRASP_RESULTS_DIR=$OUT \
    timeout 3600 python plan.py model_name=point_maze n_evals=1 \
    planner=emsmc_pointmaze goal_H=5 goal_source=random_state seed=$s \
    save_images=False planner.opt_steps=150 planner.eval_every=1 \
    planner.num_particles=3 $ARGS $CK > $OUT/run.log 2>&1
  # Success from seed_success_log.txt field 1 (the planner's own selected-action
  # evaluation), never from a run.log grep -- the oracle probe's per-particle
  # eval_actions calls each print "Success rate:" and would be counted as success.
  if [ -s "$OUT/seed_success_log.txt" ]; then
    ok=$(tail -1 $OUT/seed_success_log.txt | awk '{print ($1>=1.0)?1:0}')
    echo "$ok $(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')" > $OUT/result.txt
    echo "[emsmc/$ARM s$s] success=$ok $(cat $OUT/result.txt)"
  else
    why=OOM; grep -qi "out of memory" $OUT/run.log || why=CRASH
    echo "[emsmc/$ARM s$s] $why -- not recorded"
  fi
done
echo "=== EMSMC PROBE DONE arm=$ARM $(date +%H:%M) ==="
