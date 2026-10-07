#!/bin/bash
# Coverage + transmission measurement across kernels.
#
# Answers the two-part question directly:
#  Q1 which kernel produces the best CANDIDATES?  -> softminC / Cmin (path-free
#     latent-space coverage; the accumulated log Zhat is invalid here because the
#     M-step fits mu to the particles) plus the oracle's real-env coverage
#     max_i success(a_i).
#  Q2 do better candidates help DOWNSTREAM?       -> does a kernel's coverage
#     advantage show up in success, on a MIXED seed set where success is not
#     floored. The earlier 20-seed set was all MALA failures, so success was
#     pinned at 0 and transmission was untestable there.
#
# The oracle probe evaluates every particle in the real env at checkpoints, which
# separates coverage failure (population holds no winner -- kernel's job) from
# selection failure (it does and argmin task loss misses it -- the proxy's job),
# and reports Spearman(latent objective, true distance) as the z->s fidelity.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd /home/hviswan/Documents/grasp
export DATASET_DIR=/home/hviswan/Documents/grasp/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset MUJOCO_GL; export DISPLAY=:0
GPU=${GPU:-0}; KERNELS=${KERNELS:?set KERNELS}
SEEDS=$(cat "${SEEDFILE:?set SEEDFILE}")
CK=ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints
BASE="planner.m_step=score planner.m_smooth_samples=1 planner.beta_schedule=adaptive planner.beta_max=1.0 planner.eval_chunk=8 planner.num_particles=32"
for K in $KERNELS; do
  EXTRA=""
  [ "$K" = ula ] && EXTRA="planner.kernel_step_size=0.18"
  ROOT=$PWD/sweep_emsmc/coverage/$K
  for s in $SEEDS; do
    OUT=$ROOT/seed_$s
    [ -s "$OUT/result.txt" ] && continue
    mkdir -p $OUT
    CUDA_VISIBLE_DEVICES=$GPU GRASP_RESULTS_DIR=$OUT \
      GRASP_EM_ORACLE=$OUT/oracle.npz GRASP_EM_ORACLE_EVERY=30 \
      timeout 5400 python plan.py model_name=point_maze n_evals=1 \
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
  echo "=== COVERAGE KERNEL $K DONE $(date +%H:%M) ==="
done
