#!/bin/bash
# 1-seed sanity check before committing to a sweep: does the correctly-tuned
# kernel solve seed 6? FK solves seed 6; all four kernels in the previous sweep
# failed it with an identity metric and HMC at L=1.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset MUJOCO_GL; export DISPLAY=:0
GPU=${GPU:-0}; ARMS=${ARMS:?set ARMS}; SEED=${SEED:-6}
CK=ckpt_base_path=$PWD/checkpoints
BASE="planner.m_step=score planner.m_smooth_samples=1 planner.beta_schedule=adaptive planner.beta_max=1.0 planner.eval_chunk=8 planner.num_particles=32"
for A in $ARMS; do
  case $A in
    base)   X="++planner.kernel=mala planner.precond=none";;
    malaP)  X="++planner.kernel=mala planner.precond=grad planner.precond_power=0.5";;
    hmcP)   X="++planner.kernel=hmc planner.precond=grad planner.precond_power=0.5 planner.hmc_leaps=5";;
    hmcW)   X="++planner.kernel=hmc planner.precond=grad planner.precond_power=1.0 planner.hmc_leaps=5";;
    ulaP)   X="++planner.kernel=ula planner.precond=grad planner.precond_power=0.5 planner.kernel_step_size=0.18";;
    *) echo "unknown arm $A"; continue;;
  esac
  OUT=$PWD/sweep_emsmc/seed6check/$A
  mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=$GPU GRASP_RESULTS_DIR=$OUT timeout 5400 python plan.py \
    model_name=point_maze n_evals=1 planner=emsmc_pointmaze goal_H=5 \
    goal_source=random_state seed=$SEED save_images=False planner.opt_steps=150 \
    planner.eval_every=1 $BASE $X $CK > $OUT/run.log 2>&1
  ok=$(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print ($1>=1.0)?1:0}')
  echo "[$A] success=${ok:-CRASH} exit=$(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')"
done
echo "=== SEED6 CHECK DONE ($ARMS) ==="
