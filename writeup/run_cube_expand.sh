#!/bin/bash
# Out-of-sample test of the tail-suppression claim.
#
# The >1.5x large-deviation result (29/12 on the four non-reacher envs) was found
# post hoc, after several stratifications, so it needs confirmation on data that
# did not contribute to finding it. Cube is the right vehicle: it is in the group
# where the effect is predicted, and only 6 of the 43 in-band seeds were paired,
# leaving 38 untouched. The prediction registered BEFORE running is:
#   (a) roadmap >= L2 on success, and
#   (b) roadmap ahead on the >1.5x final-distance split,
#   (c) NO advantage once restricted to pairs where both arms succeed.
# Cube is a LeWM env and fits alongside the point-maze budget sweep on the
# other card, so this runs on GPU 0 while the queue keeps GPU 1.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl; unset DISPLAY
ROOT=$PWD/sweep_aft/rawl2
CK=ckpt_base_path=$PWD/checkpoints

for s in $(python3 -c "import json;print(' '.join(map(str,json.load(open('writeup/slow_seeds.json'))['cube_H5_expand'])))"); do
  for arm in roadmap rawl2; do
    OUT=$ROOT/$arm/cube/H5/seed_$s
    [ -s "$OUT/result.txt" ] && continue
    mkdir -p $OUT
    A=""; [ "$arm" = rawl2 ] && A="++planner.v_null=objective_raw"
    CUDA_VISIBLE_DEVICES=${GPU:-0} GRASP_RESULTS_DIR=$OUT \
      GRASP_FK_DUMP=$OUT/trace.npz GRASP_TRAJ_DUMP=$OUT/traj.npz \
      timeout 3600 python plan.py n_evals=1 seed=$s save_images=False \
      planner.opt_steps=150 planner.eval_every=1 \
      model_name=lewm_cube planner=fk_cube goal_H=5 goal_source=dset \
      planner.beta_max=10.0 $A $CK > $OUT/run.log 2>&1
    if grep -q "Success rate:" $OUT/run.log; then
      ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
      echo "$ok $(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')" > $OUT/result.txt
      echo "[$arm cube H5 s$s] success=$ok"
    else
      why=OOM; grep -qi "out of memory" $OUT/run.log || why=CRASH
      echo "[$arm cube H5 s$s] $why -- not recorded"
    fi
  done
done
echo "=== CUBE EXPANSION DONE $(date +%H:%M) ==="
