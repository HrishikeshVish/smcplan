#!/bin/bash
# Particle-count sweep: N in {1,2,4,8,16,32}.
#
# Run on REACHER, not Point maze. The other knob sweeps (tau, budget, tilt) use
# Point maze H=10, but DINO-WM encodes each frame as 196 ViT patch tokens and the
# environment is capped at 3 particles by memory -- N=16 and N=32 are not
# expressible there. Reacher is the LeWM path: a single 192-d token per frame, so
# particles are cheap, and it is also the environment the kernel block was tuned
# on. N=8 is its shipped setting.
#
# N=1 is the degenerate case of Proposition 2: a single particle cannot be
# resampled, so the run is one kernel chain and any gap to N>1 is attributable
# to the population rather than to the tilt.
#
# GPU 1 is shared with other jobs (~4.6 GB resident), so this runs one at a time.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl; unset DISPLAY
SEEDS="1 6 7 14 20 25 33 38 44 45 54 60 65 68 73 79 84 93 99 100"
ROOT=$PWD/sweep_aft/abl_particles
for N in 8 1 2 4 16 32; do          # default first, so the reference lands early
  for s in $SEEDS; do
    OUT=$ROOT/$N/seed_$s
    [ -s "$OUT/result.txt" ] && continue
    mkdir -p $OUT
    CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT GRASP_TRAJ_DUMP=$OUT/traj.npz \
      timeout 3600 python plan.py n_evals=1 seed=$s save_images=False \
      model_name=lewm_reacher planner=fk goal_H=10 goal_source=random_state \
      planner.opt_steps=150 planner.eval_every=1 planner.num_particles=$N \
      planner.num_smooth_samples=8 \
      ckpt_base_path=$PWD/checkpoints > $OUT/run.log 2>&1
    if grep -q "Success rate:" $OUT/run.log; then
      ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
      echo "$ok $(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')" > $OUT/result.txt
      echo "[particles N=$N s$s] success=$ok"
    else
      why=OOM; grep -qi "out of memory" $OUT/run.log || why=CRASH
      echo "[particles N=$N s$s] $why -- not recorded"
    fi
  done
  echo "=== ARM DONE N=$N $(date +%H:%M) ==="
done
echo "=== PARTICLE SWEEP COMPLETE $(date +%H:%M) ==="
