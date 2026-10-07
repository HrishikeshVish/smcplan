#!/bin/bash
# ULA/SGLD kernel (kernel=sgld): FK's update with Adam's preconditioner+momentum
# removed. a <- a - lr*grad L + sqrt(2*lr*T)*xi, i.e. ULA on pi ∝ exp(-L/T) with
# eps = lr*T. Every other hyperparameter identical to FK.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd /home/hviswan/Documents/grasp; unset MUJOCO_GL; export DISPLAY=:2
np () { [ "$1" = 15 ] && echo 2 || echo 3; }
for spec in "5 10" "5 20" "5 34" "5 45" "10 16" "10 24" "10 23" "15 6" "15 61" "15 94"; do
  set -- $spec; H=$1; s=$2
  OUT=$PWD/sweep_aft/ula_pusht/H${H}_seed$s; [ -s "$OUT/result.txt" ] && continue; mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT timeout 7200 python plan.py n_evals=1 seed=$s save_images=False \
    model_name=pusht planner=fk_pusht goal_H=$H goal_source=dset planner.opt_steps=150 planner.eval_every=1 \
    planner.num_particles=$(np $H) planner.num_smooth_samples=1 +planner.kernel=sgld \
    ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints > $OUT/run.log 2>&1
  ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1; echo $ok > $OUT/result.txt
  echo "[ula pusht H$H s$s] success=$ok exit=$(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')"
done
echo "=== ULA PUSHT DONE ==="
