#!/bin/bash
# 2026-10-12 (user: "pick a setting where we solve and GD/baselines don't",
# GPU0 only): Push-T seeds where ours solves and GD-12, GRASP-12, CEM-12 and
# MPPI-12 all fail in the Table 1 sweeps, and whose GOAL block pose lies on the
# env's fixed green target (256, 256, pi/4) (CPU screen, scratchpad pusht_goals.py):
#   H=10 seed 51 (7.6 px), H=10 seed 29 (8.8 px), H=15 seed 50 (11.3 px), H=15 seed 19 (14.2 px)
# Same configs and run() as render_frames_eq12.sh.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset MUJOCO_GL; export DISPLAY=:0
CK=ckpt_base_path=$PWD/checkpoints
OURS="++planner.kernel=ula planner.precond=none planner.kernel_step_size=0.18 \
planner.num_particles=12 planner.beta_schedule=adaptive planner.beta_max=1.0 \
planner.m_step=pathwise planner.m_smooth_samples=1 ++planner.lambda_v=auto ++planner.v_traj=all ++planner.v_base=2.0 ++planner.heat_t_rule=degree ++planner.surrogate_steps=3000 planner.opt_steps=150 \
planner.eval_every=1 planner.stop_on_success=true"
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0; }
run() {  # name env H seed arm
  local name=$1 e=$2 H=$3 s=$4 a=$5 OUT=$PWD/figs/renders/$1
  [ -s "$OUT/traj.npz" ] && return
  mkdir -p $OUT
  case $e in
    pointmaze) M="model_name=point_maze goal_source=random_state"; PDE="planner.graph_k_neighbors=4 ++planner.heat_c=1.0"; P=emsmc_pointmaze;;
    pusht)     M="model_name=pusht goal_source=dset"; PDE="planner.graph_k_neighbors=64 ++planner.heat_c=10.0"; P=emsmc_pusht;;
  esac
  case $e:$H in pointmaze:10|pusht:10) CH=4;; *) CH=8;; esac
  if [ $a = ours ]; then ARGS="$M planner=$P $OURS $PDE planner.eval_chunk=$CH"
  else ARGS="$M planner=gd planner.opt_steps=150 planner.eval_every=1 planner.hjb_tau=0.0 planner.num_samples=12 planner.var_actions=1.0 ++planner.num_candidates=1 ++planner.cand_chunk=$CH"; fi
  while [ "$(gpu_free)" -lt 10000 ]; do sleep 20; done
  CUDA_VISIBLE_DEVICES=0 GRASP_RESULTS_DIR=$OUT GRASP_TRAJ_DUMP=$OUT/traj.npz GRASP_TRAJ_DUMP_FRAMES=1 \
    timeout 3600 python plan.py n_evals=1 seed=$s goal_H=$H save_images=False $ARGS $CK > $OUT/run.log 2>&1
  echo "[$name] $(tail -1 $OUT/seed_success_log.txt 2>/dev/null) dump=$( [ -s $OUT/traj.npz ] && echo ok || echo MISSING)"
}
run pusht_ours12_H10_s51 pusht 10 51 ours
run pusht_ours12_H15_s50 pusht 15 50 ours
run pusht_ours12_H10_s29 pusht 10 29 ours
run pusht_ours12_H15_s19 pusht 15 19 ours
run pusht_gd12_H10_s51 pusht 10 51 gd12
run pusht_gd12_H15_s50 pusht 15 50 gd12
run pusht_gd12_H10_s29 pusht 10 29 gd12
run pusht_gd12_H15_s19 pusht 15 19 gd12
# The planner returns its run-argmin plan, which can differ from the plan
# whose evaluation succeeded (seed 51's returned plan fails on the agent's
# position). Re-render ours with the plan that succeeded (plan.py hook).
# H=15 runs above OOM'd beside another job on GPU0 and are not retried.
export GRASP_TRAJ_DUMP_SUCCESS=1
run pusht_ours12_H10_s51_succ pusht 10 51 ours
run pusht_ours12_H10_s29_succ pusht 10 29 ours
echo "=== RENDERS DONE $(date +%T) ==="
