#!/bin/bash
# SVGD / mapping-particle-filter mutation (KL steepest descent in an RKHS) on the
# Push-T sanity seeds. Everything else is FK: V-tilt weights, resampling, Adam.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; unset MUJOCO_GL; export DISPLAY=:2
SV="++planner.kernel=svgd"
np () { [ "$1" = 15 ] && echo 2 || echo 3; }
for spec in "5 10" "5 20" "5 34" "5 45" "10 16" "10 24" "10 23" "15 6" "15 61" "15 94"; do
  set -- $spec; H=$1; s=$2
  OUT=$PWD/sweep_aft/svgd_pusht/H${H}_seed$s; [ -s "$OUT/result.txt" ] && continue; mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT timeout 7200 python plan.py n_evals=1 seed=$s save_images=False \
    model_name=pusht planner=fk_pusht goal_H=$H goal_source=dset planner.opt_steps=150 planner.eval_every=1 \
    planner.num_particles=$(np $H) planner.num_smooth_samples=1 $SV \
    ckpt_base_path=$PWD/checkpoints > $OUT/run.log 2>&1
  ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1; echo $ok > $OUT/result.txt
  echo "[svgd pusht H$H s$s] success=$ok exit=$(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')"
done
echo "=== SVGD PUSHT DONE ==="
