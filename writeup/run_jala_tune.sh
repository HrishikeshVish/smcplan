#!/bin/bash
# Tune JALA's step size on seed 6.
#
# HMC's tuning was (metric, multi-leapfrog). JALA already has the metric
# (ulaP, failed) and has no momentum, so its untuned knob is EPS -- which was
# hardcoded to 0.18, borrowed from MALA's median. That is the wrong constraint:
# MALA's eps is capped by the Metropolis accept rate, which pins drift/noise at
# ~0.5 whatever the temperature. JALA has NO accept step, so eps can be pushed
# until the drift dominates:  SNR = sqrt(eps/2)*||g||/sqrt(d), so with ||g||~7
# and d=10, SNR=1 at eps~0.41 and SNR~2 at eps~1.6.
# The Jarzynski increment corrects the resulting non-invariance exactly, paying
# in WEIGHT VARIANCE instead of rejection -- so the binding constraint should be
# ESS, not acceptance. This sweep finds where that bites.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset MUJOCO_GL; export DISPLAY=:0
GPU=${GPU:-0}; EPSL=${EPSL:?set EPSL}; SEED=${SEED:-6}
CK=ckpt_base_path=$PWD/checkpoints
for E in $EPSL; do
  OUT=$PWD/sweep_emsmc/jala_tune/eps_$E
  [ -s "$OUT/result.txt" ] && continue
  mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=$GPU GRASP_RESULTS_DIR=$OUT timeout 5400 python plan.py \
    model_name=point_maze n_evals=1 planner=emsmc_pointmaze goal_H=5 \
    goal_source=random_state seed=$SEED save_images=False planner.opt_steps=150 \
    planner.eval_every=1 planner.num_particles=32 planner.m_step=score \
    planner.m_smooth_samples=1 planner.beta_schedule=adaptive planner.beta_max=1.0 \
    planner.eval_chunk=8 ++planner.kernel=ula planner.precond=grad \
    planner.precond_power=0.5 planner.kernel_step_size=$E $CK > $OUT/run.log 2>&1
  if [ -s "$OUT/seed_success_log.txt" ]; then
    ok=$(tail -1 $OUT/seed_success_log.txt | awk '{print ($1>=1.0)?1:0}')
    echo "$ok $(tail -1 $OUT/seed_success_log.txt | awk '{print $5}')" > $OUT/result.txt
    echo "[jala eps=$E] success=$ok exit=$(cut -d' ' -f2 $OUT/result.txt)"
  else echo "[jala eps=$E] CRASH"; fi
done
echo "=== JALA TUNE DONE ($EPSL) ==="
