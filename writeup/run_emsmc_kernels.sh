#!/bin/bash
# Do the other MCMC kernels rescue the seeds MALA fails?
#
# Runs on the 20 seeds EM/p32-MALA failed in the 100-seed confirmation (7 of
# which FK solves). KERNELS is a space-separated list run in sequence on one
# GPU. `mala` is included as a RE-RUN CONTROL: these are by construction the
# seeds MALA failed, so any seed a different kernel solves has to be compared
# against how many MALA itself solves on a second pass, or kernel effect and
# run-to-run variance are indistinguishable.
#
# ULA has no accept step, so dual averaging is inactive for it and its step size
# is fixed to MALA's median converged value (0.18) -- the step size MALA's own
# acceptance criterion selected on this target.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset MUJOCO_GL; export DISPLAY=:0
GPU=${GPU:-0}; KERNELS=${KERNELS:?set KERNELS}
SEEDS=$(cat "${SEEDFILE:?set SEEDFILE}")
CK=ckpt_base_path=$PWD/checkpoints
BASE="planner.m_step=score planner.m_smooth_samples=1 planner.beta_schedule=adaptive planner.beta_max=1.0 planner.eval_chunk=8 planner.num_particles=32"
for K in $KERNELS; do
  EXTRA=""
  [ "$K" = ula ] && EXTRA="planner.kernel_step_size=0.18"
  ROOT=$PWD/sweep_emsmc/failed_kernels/$K
  for s in $SEEDS; do
    OUT=$ROOT/seed_$s
    [ -s "$OUT/result.txt" ] && continue
    mkdir -p $OUT
    CUDA_VISIBLE_DEVICES=$GPU GRASP_RESULTS_DIR=$OUT \
      timeout 3600 python plan.py model_name=point_maze n_evals=1 \
      planner=emsmc_pointmaze goal_H=5 goal_source=random_state seed=$s \
      save_images=False planner.opt_steps=150 planner.eval_every=1 \
      $BASE ++planner.kernel=$K $EXTRA $CK > $OUT/run.log 2>&1
    # Read success from seed_success_log.txt field 1, which the planner writes
    # from its OWN selected-action evaluation. Do NOT grep run.log: with the
    # oracle probe enabled, eval_actions is called once per particle and each
    # call prints "Success rate:", so a grep records particle-level coverage as
    # planner success (caught 2026-09-26 on mala/seed_59).
    if [ -s "$OUT/seed_success_log.txt" ]; then
      ok=$(tail -1 $OUT/seed_success_log.txt | awk '{print ($1>=1.0)?1:0}')
      echo "$ok $(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')" > $OUT/result.txt
      echo "[$K s$s] success=$ok"
    else
      why=OOM; grep -qi "out of memory" $OUT/run.log || why=CRASH
      echo "[$K s$s] $why"
    fi
  done
  echo "=== KERNEL $K DONE $(date +%H:%M) ==="
done
