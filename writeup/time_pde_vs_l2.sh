#!/bin/bash
# Wall clock of the PDE cost against the L2 cost, our planner only (appendix).
# One run per environment and cost (seed 1, 30 iterations, no early stop); the
# per-trial time at K=150 is extrapolated as setup + 150 * per-iteration time.
# Same settings as writeup/run_emsmc_final.sh (pdefix) and run_emsmc_cube.sh;
# the L2 arm sets lambda_v=0 and hjb_tau=0, so no roadmap is loaded or built.
# GPU0 only, one run at a time (GPU1 is reserved for another project).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd /home/hviswan/Documents/grasp
export DATASET_DIR=/home/hviswan/Documents/grasp/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRASP_TIMING=1
ROOT=${ROOT:-$PWD/sweep_timing}
ITERS=${ITERS:-30}
CK=ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints
PIN="++planner.kernel=ula planner.precond=none planner.kernel_step_size=0.18 \
planner.num_particles=12 planner.beta_schedule=adaptive planner.beta_max=1.0 \
planner.m_step=pathwise planner.m_smooth_samples=1 planner.opt_steps=$ITERS \
planner.eval_every=$ITERS planner.stop_on_success=false"
PDECOST="++planner.lambda_v=auto ++planner.v_traj=all ++planner.v_base=2.0 \
++planner.heat_t_rule=degree ++planner.surrogate_steps=3000"
L2COST="planner.lambda_v=0.0 ++planner.hjb_tau=0.0"
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0; }
for e in pointmaze wall pusht reacher cube; do
  case $e in
    reacher)   M="model_name=lewm_reacher planner=emsmc_reacher goal_source=random_state"; H=10; GL=egl
               K="planner.graph_k_neighbors=16 ++planner.heat_c=3.0"; CH=8;;
    pointmaze) M="model_name=point_maze planner=emsmc_pointmaze goal_source=random_state"; H=10; GL=x
               K="planner.graph_k_neighbors=4 ++planner.heat_c=1.0"; CH=4;;
    wall)      M="model_name=wall_single planner=emsmc_wall goal_source=random_state"; H=10; GL=x
               K="planner.graph_k_neighbors=32 ++planner.heat_c=30.0"; CH=8;;
    pusht)     M="model_name=pusht planner=emsmc_pusht goal_source=dset"; H=10; GL=x
               K="planner.graph_k_neighbors=64 ++planner.heat_c=10.0"; CH=4;;
    cube)      M="model_name=lewm_cube planner=emsmc_cube goal_source=dset"; H=5; GL=egl
               K="planner.graph_k_neighbors=16 ++planner.heat_c=1.0"; CH=8;;
  esac
  if [ "$GL" = egl ]; then export MUJOCO_GL=egl; unset DISPLAY; else unset MUJOCO_GL; export DISPLAY=:0; fi
  for arm in pde l2; do
    OUT=$ROOT/$e/${arm}_H$H
    [ -s "$OUT/result.txt" ] && continue
    mkdir -p $OUT
    if [ $arm = pde ]; then COST="$PDECOST $K"; else COST="$L2COST"; fi
    while [ "$(gpu_free)" -lt 6000 ]; do sleep 20; done
    t0=$(date +%s.%N)
    CUDA_VISIBLE_DEVICES=0 GRASP_RESULTS_DIR=$OUT timeout 3600 python plan.py n_evals=1 seed=1 \
      goal_H=$H save_images=False $M $PIN $COST planner.eval_chunk=$CH $CK > $OUT/run.log 2>&1
    dt=$(echo "$(date +%s.%N) - $t0" | bc)
    echo "process_wall=${dt}s" > $OUT/result.txt
    echo "[$e $arm H$H] process ${dt}s | $(grep -h '\[Timing\]' $OUT/run.log | tr '\n' ' ')"
  done
done
echo "=== TIMING DONE $(date +%T) ==="
