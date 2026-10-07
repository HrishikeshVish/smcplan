#!/bin/bash
# JALA-EM on CUBE under LeFlow's protocol (see writeup/run_cube_leflow.sh for the
# protocol notes): lewm_cube, goal_H=5, goal_source=dset, seeds 1..50, reference
# 0.04 m block-position success test. Writes sweep_cube_leflow/<ARM>/seed_s so
# writeup/make_cube_table.py reads it next to the gd/grasp/mpc_cem/fk arms.
# Planner config pinned exactly as in writeup/run_emsmc_table.sh (L2 cost,
# lambda_v=0). ARM=pde switches on the calibrated roadmap cost (k=16, c=1;
# sweep_emsmc/calib_*.log) with the trajectory-weighted Vhat and auto lambda_V.
# Claim queue: any number of workers (GPU=<g>) can share it.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd /home/hviswan/Documents/grasp
export DATASET_DIR=/home/hviswan/Documents/grasp/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl; unset DISPLAY
GPU=${GPU:?set GPU}
ARM=${ARM:-em}
ROOT=/home/hviswan/Documents/grasp/sweep_cube_leflow
SEEDS=${SEEDS:-$(seq 1 50)}
NEED_MIB=${NEED_MIB:-6000}
KERNEL=${KERNEL:-ula}
case $KERNEL in   # E-step kernel; ula = JALA (main). Others: settings of writeup/run_emsmc_kernelsweep.sh
  ula)  KX="++planner.kernel=ula planner.precond=none planner.kernel_step_size=0.18";;
  hmc)  KX="++planner.kernel=hmc planner.precond=grad planner.precond_power=0.5 planner.hmc_leaps=5 planner.hmc_jitter=true";;
  mala) KX="++planner.kernel=mala planner.precond=none";;
  rwm)  KX="++planner.kernel=rwm planner.precond=none";;
  *) echo "bad KERNEL $KERNEL"; exit 1;;
esac
PIN="$KX \
planner.num_particles=12 planner.beta_schedule=adaptive planner.beta_max=1.0 \
planner.m_step=score planner.opt_steps=150 planner.eval_every=1 \
planner.stop_on_success=true planner.eval_chunk=8"
if [ "$ARM" = pdefix ] || [[ "$ARM" == kfix_* ]]; then
  # fixed JALA-EM: pathwise M-step (no extra rollouts) + calibrated cube PDE
  PIN="${PIN/planner.m_step=score/planner.m_step=pathwise planner.m_smooth_samples=1}"
  COST="++planner.lambda_v=auto ++planner.v_traj=all ++planner.v_base=2.0 \
++planner.heat_t_rule=degree ++planner.heat_c=1.0 planner.graph_k_neighbors=16 \
++planner.surrogate_steps=3000"
elif [ "$ARM" = l2fix ]; then
  # fixed JALA-EM with the L2 cost (lambda_v=0): the paired arm of pdefix for
  # the PDE-vs-L2 comparison; identical except for the cost.
  PIN="${PIN/planner.m_step=score/planner.m_step=pathwise planner.m_smooth_samples=1}"
  COST="planner.lambda_v=0.0"
elif [ "$ARM" = pde ]; then
  COST="++planner.lambda_v=auto ++planner.v_traj=all ++planner.v_base=2.0 \
++planner.heat_t_rule=degree ++planner.heat_c=1.0 planner.graph_k_neighbors=16 \
++planner.surrogate_steps=3000"
else
  COST="planner.lambda_v=0.0"
fi
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU; }
for s in $SEEDS; do
  OUT=$ROOT/$ARM/seed_$s
  [ -s "$OUT/result.txt" ] && continue
  mkdir -p $ROOT/$ARM
  mkdir "$OUT.claim" 2>/dev/null || continue
  while [ "$(gpu_free)" -lt "$NEED_MIB" ]; do sleep 20; done
  mkdir -p "$OUT"; t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU GRASP_TRAJ_DUMP="$OUT/traj.npz" GRASP_RESULTS_DIR=$OUT \
    timeout 5400 python plan.py model_name=lewm_cube planner=emsmc_cube n_evals=1 \
    goal_H=5 goal_source=dset seed=$s save_images=False $PIN $COST \
    ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints > $OUT/run.log 2>&1
  dt=$(( $(date +%s) - t0 ))
  if [ -s "$OUT/seed_success_log.txt" ]; then
    read ok it < <(tail -1 $OUT/seed_success_log.txt | awk '{print ($1>=1.0)?1:0, $5}')
    echo "$ok $dt $it" > $OUT/result.txt          # field 1 = success, as the table reads
    echo "[gpu$GPU cube $ARM s$s] success=$ok iter=$it ${dt}s"
    rmdir "$OUT.claim"
  else
    why=CRASH; grep -qi "out of memory" $OUT/run.log && why=OOM
    echo "[gpu$GPU cube $ARM s$s] $why ${dt}s (claim left in place)"
  fi
done
echo "=== gpu$GPU cube $ARM WORKER DONE $(date +%T) ==="
