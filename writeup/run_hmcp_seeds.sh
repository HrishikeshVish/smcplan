#!/bin/bash
# hmcP (HMC L=5 jittered + pSGLD metric) on a seed list, to test whether the
# seed-6 win generalises or was a lucky run. Deliberately mixes seeds the OLD
# config solved (regression check -- does the new tuning break them?) with seeds
# it failed (improvement check).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset MUJOCO_GL; export DISPLAY=:0
GPU=${GPU:-0}; SEEDS=${SEEDS:?set SEEDS}
CK=ckpt_base_path=$PWD/checkpoints
for s in $SEEDS; do
  OUT=$PWD/sweep_emsmc/hmcp_seeds/seed_$s
  [ -s "$OUT/result.txt" ] && continue
  mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=$GPU GRASP_RESULTS_DIR=$OUT timeout 5400 python plan.py \
    model_name=point_maze n_evals=1 planner=emsmc_pointmaze goal_H=5 \
    goal_source=random_state seed=$s save_images=False planner.opt_steps=150 \
    planner.eval_every=1 planner.num_particles=32 planner.m_step=score \
    planner.m_smooth_samples=1 planner.beta_schedule=adaptive planner.beta_max=1.0 \
    planner.eval_chunk=8 ++planner.kernel=hmc planner.precond=grad \
    planner.precond_power=0.5 planner.hmc_leaps=5 $CK > $OUT/run.log 2>&1
  if [ -s "$OUT/seed_success_log.txt" ]; then
    ok=$(tail -1 $OUT/seed_success_log.txt | awk '{print ($1>=1.0)?1:0}')
    echo "$ok $(tail -1 $OUT/seed_success_log.txt | awk '{print $5}')" > $OUT/result.txt
    echo "[hmcP s$s] success=$ok"
  else echo "[hmcP s$s] CRASH"; fi
done
echo "=== HMCP SEEDS DONE ($SEEDS) ==="
