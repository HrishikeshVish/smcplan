#!/bin/bash
# Per-iteration particle traces for the sampler-diagnostics figures.
#
# TWO arms per seed, which is the comparison the figure is for:
#   pde  : tilt on the roadmap cost-to-go C            (ours)
#   l2   : tilt on the same L2 objective the kernel descends (v_null=objective),
#          rescaled to C's per-step mean/std so ONLY the ordering differs --
#          otherwise a smaller spread would weaken the tilt and read as "no
#          effect" for the wrong reason.
# This mirrors Kim et al. (2503.15704, Figs. 26-28), whose axis of comparison is
# adaptive vs fixed tuning; ours is PDE vs L2 inside the same SMC sampler.
#
# 8 seeds per environment, stratified so the subset's success rate matches that
# cell's full 100-seed rate (writeup/diag_seeds.json) -- the traces are therefore
# a fair sample of runs, not the ones that happen to look best. Following Kim et
# al. (2503.15704, Figs. 26-28), which show the realised schedules from a subset
# of 8 runs rather than one representative trace.
#
# GRASP_FK_DUMP records beta, the normalised weights w, V, L and resample events
# at every planner iteration; ESS is recomputed from w when plotting. One file
# per planning call, so trace_call000.npz is the cold start.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=$PWD/sweep_aft/fk_dumps
CK=ckpt_base_path=$PWD/checkpoints

run () {  # env seed gl args...
  local env=$1 s=$2 gl=$3; shift 3
  for arm in pde l2; do
    local A=""; [ $arm = l2 ] && A="++planner.v_null=objective"
    _one $env $s $arm $gl "$@" $A
  done
}

_one () {  # env seed arm gl args...
  local env=$1 s=$2 arm=$3 gl=$4; shift 4
  local OUT=$ROOT/$arm/$env/seed_$s
  ls $OUT/trace_call000.npz >/dev/null 2>&1 && return
  mkdir -p $OUT
  if [ "$gl" = egl ]; then export MUJOCO_GL=egl; unset DISPLAY; else unset MUJOCO_GL; export DISPLAY=:2; fi
  CUDA_VISIBLE_DEVICES=${GPU:-1} GRASP_RESULTS_DIR=$OUT GRASP_FK_DUMP=$OUT/trace.npz \
    timeout 3600 python plan.py n_evals=1 seed=$s save_images=False \
    planner.opt_steps=150 planner.eval_every=1 "$@" $CK > $OUT/run.log 2>&1
  echo "[$arm $env s$s] calls=$(ls $OUT/trace_call*.npz 2>/dev/null | wc -l) ok=$(grep -c 'Success rate:  1.0' $OUT/run.log)"
}

for s in 1 5 9 16 25 33 42 50; do
  run cube $s egl model_name=lewm_cube planner=fk_cube goal_H=5 goal_source=dset planner.beta_max=10.0
done
for s in 1 2 19 34 50 65 81 100; do
  run reacher $s egl model_name=lewm_reacher planner=fk goal_H=10 goal_source=random_state \
      planner.num_particles=8 planner.num_smooth_samples=8
done
for s in 1 7 20 38 60 79 99 100; do
  run pointmaze $s none model_name=point_maze planner=fk_pointmaze goal_H=10 goal_source=random_state \
      planner.num_particles=3 planner.num_smooth_samples=1
done
for s in 1 2 30 43 54 78 98 100; do
  run pusht $s none model_name=pusht planner=fk_pusht goal_H=5 goal_source=dset \
      planner.num_particles=3 planner.num_smooth_samples=1
done
echo "=== FK DUMPS DONE ==="
