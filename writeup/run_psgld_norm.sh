#!/bin/bash
# Follow-up to run_psgld_noise.sh: the normalised pSGLD, P <- P/mean(P), so the
# injected noise has exactly FK's sqrt(2*lr*T) variance budget and the
# preconditioner only reshapes it. This separates the two things full pSGLD
# changes at once (anisotropy vs. a hotter chain). Run on cube seeds 17 and 24
# -- 24 is the seed that passes the stability screen, so it discriminates.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd /home/hviswan/Documents/grasp
export DATASET_DIR=/home/hviswan/Documents/grasp/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl; unset DISPLAY
CK=ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints

# wait for the unnormalised probe to release the GPU
while pgrep -f "run_psgld_noise.sh" > /dev/null; do sleep 20; done

run () {  # seed tag args...
  local s=$1 tag=$2; shift 2
  local OUT=$PWD/sweep_aft/psgld_noise/cube_s$s/$tag
  [ -s "$OUT/result.txt" ] && return; mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=${GPU:-1} GRASP_RESULTS_DIR=$OUT timeout 3600 python plan.py n_evals=1 seed=$s \
    save_images=False planner.opt_steps=150 planner.eval_every=1 \
    model_name=lewm_cube planner=fk_cube goal_H=5 goal_source=dset planner.beta_max=10.0 \
    "$@" $CK > $OUT/run.log 2>&1
  grep -q "Success rate:" $OUT/run.log || { echo "[s$s $tag] CRASHED: $(grep -oE '[A-Za-z]*Error[^\"]*' $OUT/run.log | head -1 | cut -c1-90)"; return; }
  local ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
  echo "$ok" > $OUT/result.txt
  local best=$(grep -oE "array\(\[[0-9.]+\]\)\}" $OUT/run.log | grep -oE "[0-9.]+" | sort -g | head -1)
  echo "[s$s $tag] success=$ok exit=$(tail -1 $OUT/seed_success_log.txt | awk '{print $5}') best_dist=$best"
}

N="++planner.precond_noise=true ++planner.precond_freeze=10 ++planner.precond_normalize=true"
for s in 24 17; do
  run $s fk_control
  run $s psgld_norm_1.0 $N ++planner.precond_noise_scale=1.0
  run $s psgld_norm_0.5 $N ++planner.precond_noise_scale=0.5
done
echo "=== PSGLD NORM PROBE DONE ==="
