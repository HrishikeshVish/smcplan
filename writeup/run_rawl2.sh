#!/bin/bash
# Roadmap cost vs RAW L2 tilt -- the comparison the paper's claim actually makes.
#
# The claim is that a PDE-derived cost structure is better suited NUMERICALLY for
# tilting: its spread over the population is what makes the exponent do useful
# work. An earlier run used v_null=objective, which standardises L2 to V's mean
# and standard deviation -- that hands L2 the property under test and reduces the
# experiment to a comparison of orderings. v_null=objective_raw substitutes L2 as
# it actually is, which is the point.
#
# Seeds: 5 per environment-horizon from writeup/slow_seeds.json -- seeds the
# roadmap SOLVES but needs more than 2 iterations for, so there is convergence
# behaviour to compare. Trivial seeds (exit at 0-2) would show nothing.
#
# Both dumps are on: GRASP_FK_DUMP for the particle trace (beta, weights, ESS)
# and GRASP_TRAJ_DUMP for the executed state trajectory, so path length and
# straightness are recoverable -- distance to the goal is not trajectory quality.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=$PWD/sweep_aft/rawl2
CK=ckpt_base_path=$PWD/checkpoints
SEEDS_JSON=writeup/slow_seeds.json

run () {  # env H arm gl seed args...
  local env=$1 H=$2 arm=$3 gl=$4 s=$5; shift 5
  local OUT=$ROOT/$arm/$env/H$H/seed_$s
  [ -s "$OUT/result.txt" ] && return
  mkdir -p $OUT
  if [ "$gl" = egl ]; then export MUJOCO_GL=egl; unset DISPLAY; else unset MUJOCO_GL; export DISPLAY=:2; fi
  local A=""; [ "$arm" = rawl2 ] && A="++planner.v_null=objective_raw"
  CUDA_VISIBLE_DEVICES=${GPU:-1} GRASP_RESULTS_DIR=$OUT \
    GRASP_FK_DUMP=$OUT/trace.npz GRASP_TRAJ_DUMP=$OUT/traj.npz \
    timeout 3600 python plan.py n_evals=1 seed=$s save_images=False \
    planner.opt_steps=150 planner.eval_every=1 "$@" $A $CK > $OUT/run.log 2>&1
  if grep -q "Success rate:" $OUT/run.log; then
    local ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
    echo "$ok $(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')" > $OUT/result.txt
    echo "[$arm $env H$H s$s] success=$ok"
  else
    why=OOM; grep -qi "out of memory" $OUT/run.log || why=CRASH
    echo "[$arm $env H$H s$s] $why -- not recorded"
  fi
}

np () { [ "$1" = 15 ] && echo 2 || echo 3; }
# SEED_LIMIT: run only the first N seeds of each cell, so the comparison can be
# piloted on one seed per cell before committing the rest.
LIMIT=${SEED_LIMIT:-99}
seeds () { python3 -c "import json;print(' '.join(map(str,json.load(open('$SEEDS_JSON'))['$1'][:$LIMIT])))"; }

# Arms are interleaved PER CELL, not looped outside the cells: each cell runs
# roadmap then rawl2 back to back, so a paired comparison is readable as soon as
# the first cells land rather than only after all 13 roadmap runs finish.
cell () {  # env H gl args...
  local env=$1 H=$2 gl=$3; shift 3
  for s in $(seeds ${env}_H${H}); do
    for arm in roadmap rawl2; do
      run $env $H $arm $gl $s "$@"
    done
  done
}

for H in 5 10 15; do
  cell pointmaze $H none model_name=point_maze planner=fk_pointmaze goal_H=$H \
    goal_source=random_state planner.num_particles=$(np $H) planner.num_smooth_samples=1
  cell pusht $H none model_name=pusht planner=fk_pusht goal_H=$H goal_source=dset \
    planner.num_particles=$(np $H) planner.num_smooth_samples=1
  cell wall $H none model_name=wall_single planner=fk_wall goal_H=$H \
    goal_source=random_state planner.num_particles=$(np $H) planner.num_smooth_samples=1
  cell reacher $H egl model_name=lewm_reacher planner=fk goal_H=$H \
    goal_source=random_state planner.num_particles=8 planner.num_smooth_samples=8
done
cell cube 5 egl model_name=lewm_cube planner=fk_cube goal_H=5 goal_source=dset \
  planner.beta_max=10.0
echo "=== RAWL2 DONE ==="
