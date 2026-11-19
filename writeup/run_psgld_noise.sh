#!/bin/bash
# pSGLD noise-level probe on cube seed 17 (FK solves it; pSGLD at scale 1.0 does
# not, best cube_pos_dist 0.061 vs the 0.04 threshold). sqrt(P) is not
# dimensionless, so the matched noise can be much larger than FK's isotropic
# sqrt(2*lr*T); this walks the multiplier down and includes plain FK as the
# control on the same code path.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl; unset DISPLAY
ROOT=$PWD/sweep_aft/psgld_noise/cube_s17
CK=ckpt_base_path=$PWD/checkpoints

run () {  # tag args...
  local tag=$1; shift
  local OUT=$ROOT/$tag
  [ -s "$OUT/result.txt" ] && return; mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=${GPU:-1} GRASP_RESULTS_DIR=$OUT timeout 3600 python plan.py n_evals=1 seed=17 \
    save_images=False planner.opt_steps=150 planner.eval_every=1 \
    model_name=lewm_cube planner=fk_cube goal_H=5 goal_source=dset planner.beta_max=10.0 \
    "$@" $CK > $OUT/run.log 2>&1
  grep -q "Success rate:" $OUT/run.log || { echo "[$tag] CRASHED: $(grep -oE '[A-Za-z]*Error[^\"]*' $OUT/run.log | head -1 | cut -c1-90)"; return; }
  local ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
  echo "$ok" > $OUT/result.txt
  local best=$(grep -oE "array\(\[[0-9.]+\]\)\}" $OUT/run.log | grep -oE "[0-9.]+" | sort -g | head -1)
  echo "[$tag] success=$ok exit=$(tail -1 $OUT/seed_success_log.txt | awk '{print $5}') best_dist=$best sqrtP=$(grep -oE 'mean sqrt\(P\) at freeze: [0-9.eE+-]+' $OUT/run.log | head -1 | awk '{print $NF}')"
}

run fk_control
for sc in 0.3 0.1 0.03 0.0; do
  run psgld_$sc ++planner.precond_noise=true ++planner.precond_freeze=10 ++planner.precond_noise_scale=$sc
done
echo "=== PSGLD NOISE PROBE DONE ==="
