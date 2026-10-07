#!/bin/bash
# Finish specific (kernel, seed) pairs for the coverage sweep.
#
# A SEPARATE FILE on purpose: bash re-reads a running script by byte offset, so
# editing run_emsmc_coverage.sh mid-flight killed one runner with a syntax error
# (2026-09-26). Never edit a shell script that is executing -- add a new one.
#
# Success is read from seed_success_log.txt field 1 (the planner's own
# selected-action eval), never from a run.log grep: the oracle probe calls
# eval_actions per particle and each prints "Success rate:".
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd /home/hviswan/Documents/grasp
export DATASET_DIR=/home/hviswan/Documents/grasp/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset MUJOCO_GL; export DISPLAY=:0
GPU=${GPU:-0}; K=${K:?set K}; SEEDS=${SEEDS:?set SEEDS}
CK=ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints
BASE="planner.m_step=score planner.m_smooth_samples=1 planner.beta_schedule=adaptive planner.beta_max=1.0 planner.eval_chunk=8 planner.num_particles=32"
EXTRA=""; [ "$K" = ula ] && EXTRA="planner.kernel_step_size=0.18"
for s in $SEEDS; do
  OUT=$PWD/sweep_emsmc/coverage/$K/seed_$s
  [ -s "$OUT/result.txt" ] && continue
  mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=$GPU GRASP_RESULTS_DIR=$OUT \
    GRASP_EM_ORACLE=$OUT/oracle.npz GRASP_EM_ORACLE_EVERY=30 \
    timeout 5400 python plan.py model_name=point_maze n_evals=1 \
    planner=emsmc_pointmaze goal_H=5 goal_source=random_state seed=$s \
    save_images=False planner.opt_steps=150 planner.eval_every=1 \
    $BASE ++planner.kernel=$K $EXTRA $CK > $OUT/run.log 2>&1
  if [ -s "$OUT/seed_success_log.txt" ]; then
    ok=$(tail -1 $OUT/seed_success_log.txt | awk '{print ($1>=1.0)?1:0}')
    echo "$ok $(tail -1 $OUT/seed_success_log.txt | awk '{print $5}')" > $OUT/result.txt
    echo "[$K s$s] success=$ok"
  else
    echo "[$K s$s] CRASH/OOM -- no planner write"
  fi
done
echo "=== FINISH DONE $K $(date +%H:%M) ==="
