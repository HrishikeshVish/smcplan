#!/bin/bash
# 2026-10-07 (user-approved GPU use, GPU0 only): executed-rollout dumps for the
# frame-strip figures under the equal-particle protocol of Table 1.
#   main text : ours (JALA-EM, pdefix config) vs GD with 12 particles (gds12)
#               Point maze H=10 seed 88, Push-T H=5 seed 10 (seeds where ours'
#               FINAL executed plan succeeds -- the strip shows the final plan --
#               and GD-12 fails; the first picks, 10 and 18, solved mid-run but
#               returned a final plan that misses the goal)
#   appendix  : the seeds of the earlier strips (Point maze H=10 seed 24,
#               Push-T H=5 seed 20): ours and GD-12 here, GD-1 from the existing
#               figs/renders/{pm_gd,pusht_gd}
# Configs copied from writeup/run_emsmc_final.sh (ours) and
# writeup/run_baselines_ms.sh (gds12). One run at a time; waits for 10 GB free
# so a co-located job keeps its headroom.
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
run pm_ours12_s88    pointmaze 10 88 ours
run pm_gd12_s88      pointmaze 10 88 gd12
run pusht_ours12_s10 pusht 5 10 ours
run pusht_gd12_s10   pusht 5 10 gd12
run pm_ours12_s24    pointmaze 10 24 ours
run pm_gd12_s24      pointmaze 10 24 gd12
run pusht_ours12_s20 pusht 5 20 ours
run pusht_gd12_s20   pusht 5 20 gd12
echo "=== RENDERS DONE $(date +%T) ==="
