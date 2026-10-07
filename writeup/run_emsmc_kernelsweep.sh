#!/bin/bash
# Kernel sweep for the EM/SMC planner on the remaining environments.
# GPU1 ONLY, by request -- GPU0 is left free.
#
# Config is the one validated on pointmaze over 40 paired seeds (JALA 33/40 vs
# FK 32/40): P=12, eps=0.18, adaptive beta capped at 1.0, m_step=score, 150 steps.
# Only the KERNEL varies:
#   ula   JALA (ULA + Jarzynski weight correction), IDENTITY metric. The metric is
#         off deliberately: for an unadjusted kernel the Jarzynski increment is
#         computed in the preconditioned coordinates, so an anisotropic metric
#         widens the between-particle weight spread and causes resampling thrash
#         (pointmaze P=12: 80-93 resamples, 5-6 of 12 unique, vs 0-18 and 12/12).
#   hmcP  HMC L=5 jittered + pSGLD metric. For an MH kernel the increment is 0 so
#         the metric is free, and it is NECESSARY -- a 2x2 on pointmaze seed 6
#         showed neither momentum nor metric alone suffices. Costs 5 rollouts per
#         kernel step.
#   mala  identity metric, single step: the baseline control.
#   rwm   gradient-free control.
#
# Per-env invocations copied from the recorded fk sweeps so the rows stay
# comparable: wall/reacher use goal_source=random_state, cube uses dset with
# beta_max=10.0; reacher needs MUJOCO_GL=egl while the DINO-WM envs need an X
# display.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd /home/hviswan/Documents/grasp
export DATASET_DIR=/home/hviswan/Documents/grasp/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV=${ENV:?set ENV to wall|reacher|cube}
KERNELS=${KERNELS:-"ula hmcP mala rwm"}
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8"}
CK=ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints
case $ENV in
  wall)    MODEL="model_name=wall_single planner=emsmc_wall goal_H=10 goal_source=random_state"; GL=none;;
  reacher) MODEL="model_name=lewm_reacher planner=emsmc_reacher goal_H=10 goal_source=random_state"; GL=egl;;
  cube)    MODEL="model_name=lewm_cube planner=emsmc_cube goal_H=5 goal_source=dset planner.beta_max=10.0"; GL=none;;
  *) echo "bad ENV"; exit 1;;
esac
if [ "$GL" = egl ]; then export MUJOCO_GL=egl; unset DISPLAY; else unset MUJOCO_GL; export DISPLAY=:0; fi
for K in $KERNELS; do
  case $K in
    ula)  X="++planner.kernel=ula planner.precond=none planner.kernel_step_size=0.18";;
    hmcP) X="++planner.kernel=hmc planner.precond=grad planner.precond_power=0.5 planner.hmc_leaps=5";;
    mala) X="++planner.kernel=mala planner.precond=none";;
    rwm)  X="++planner.kernel=rwm planner.precond=none";;
    *) echo "bad kernel $K"; continue;;
  esac
  for s in $SEEDS; do
    OUT=$PWD/sweep_emsmc/kernelsweep/$ENV/$K/seed_$s
    [ -s "$OUT/result.txt" ] && continue
    mkdir -p $OUT
    CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT timeout 5400 python plan.py \
      n_evals=1 seed=$s save_images=False planner.eval_every=1 \
      $MODEL $X $CK > $OUT/run.log 2>&1
    # success from the planner's own write, never a run.log grep
    if [ -s "$OUT/seed_success_log.txt" ]; then
      ok=$(tail -1 $OUT/seed_success_log.txt | awk '{print ($1>=1.0)?1:0}')
      echo "$ok $(tail -1 $OUT/seed_success_log.txt | awk '{print $5}')" > $OUT/result.txt
      echo "[$ENV/$K s$s] success=$ok"
    else
      why=OOM; grep -qi "out of memory" $OUT/run.log || why=CRASH
      echo "[$ENV/$K s$s] $why"
    fi
  done
  echo "=== $ENV / $K DONE $(date +%H:%M) ==="
done
echo "=== KERNEL SWEEP $ENV COMPLETE ==="
