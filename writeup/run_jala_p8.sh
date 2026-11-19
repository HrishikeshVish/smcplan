#!/bin/bash
# Does JALA's result hold at P=8 instead of 32?
#
# Motivation is budget: P=32 costs 33 rollout sequences/iteration against FK's 3,
# and the earlier finding was that the EM planner needed P=32 to match FK at all
# (~10x the rollouts). ULA costs 1 rollout per kernel step, so at P=8 the whole
# iteration is 9 sequences -- 3.7x cheaper than P=32 and within 3x of FK.
#
# Two arms, because JALA's 9/12 sweep result predates preconditioning:
#   ident = the exact sweep config (identity metric, eps=0.18) at lower P
#   pscond = the metric-fixed version
# Seeds: 19, 25, 52 are the three JALA uniquely solved at P=32; 1, 2 are
# regression checks.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset MUJOCO_GL; export DISPLAY=:0
GPU=${GPU:-0}; ARM=${ARM:?set ARM}; SEEDS=${SEEDS:-"1 2 19 25 52"}; P=${P:-8}
CK=ckpt_base_path=$PWD/checkpoints
case $ARM in
  ident)  X="planner.precond=none";;
  pscond) X="planner.precond=grad planner.precond_power=0.5";;
  *) echo "bad ARM"; exit 1;;
esac
for s in $SEEDS; do
  OUT=$PWD/sweep_emsmc/jala_p$P/$ARM/seed_$s
  [ -s "$OUT/result.txt" ] && continue
  mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=$GPU GRASP_RESULTS_DIR=$OUT timeout 5400 python plan.py \
    model_name=point_maze n_evals=1 planner=emsmc_pointmaze goal_H=5 \
    goal_source=random_state seed=$s save_images=False planner.opt_steps=150 \
    planner.eval_every=1 planner.num_particles=$P planner.m_step=score \
    planner.m_smooth_samples=1 planner.beta_schedule=adaptive planner.beta_max=1.0 \
    planner.eval_chunk=8 ++planner.kernel=ula planner.kernel_step_size=0.18 \
    $X $CK > $OUT/run.log 2>&1
  if [ -s "$OUT/seed_success_log.txt" ]; then
    ok=$(tail -1 $OUT/seed_success_log.txt | awk '{print ($1>=1.0)?1:0}')
    echo "$ok $(tail -1 $OUT/seed_success_log.txt | awk '{print $5}')" > $OUT/result.txt
    echo "[P$P/$ARM s$s] success=$ok"
  else echo "[P$P/$ARM s$s] CRASH"; fi
done
echo "=== JALA P=$P $ARM DONE ==="
