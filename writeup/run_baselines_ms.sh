#!/bin/bash
# 12-candidate baselines: GD-12 (12 independent starts, best-of by current
# objective; planning/gd.py _plan_multistart) and GRASP-12 (GRASP's own
# num_parallel_trajectories=12, best-of by its loss). Same protocol as the
# 1-candidate rows (writeup/run_sweep.sh): 150 optimisation steps, eval_every=1,
# per-env model and goal source, seeds 1..100, H in {5,10,15}; cube under
# LeFlow's protocol (goal_H=5, dset, seeds 1..50, planner=gd_cube as the tuned
# GD row). This matches JALA-EM's P=12 samples and 150 steps.
#
# Claim queue shared by any number of workers (GPU=<g>). result.txt = "ok dt"
# (field 1 success), read by make_main_table.py / make_cube_table.py.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=${GPU:?set GPU}
NEED_MIB=${NEED_MIB:-12000}
PLANNERS=${PLANNERS:-"gd12 grasp12"}
ENVS=${ENVS:-"pointmaze wall reacher pusht cube"}
HORIZONS=${HORIZONS:-"5 10 15"}
CK=ckpt_base_path=$PWD/checkpoints
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU; }

run_one () {   # OUT args...
  local OUT=$1; shift
  [ -s "$OUT/result.txt" ] && return
  mkdir -p "$(dirname $OUT)"
  mkdir "$OUT.claim" 2>/dev/null || return
  while [ "$(gpu_free)" -lt "$NEED_MIB" ]; do sleep 20; done
  mkdir -p "$OUT"; local t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU GRASP_RESULTS_DIR=$OUT timeout 5400 python plan.py \
    n_evals=1 save_images=False "$@" $CK > $OUT/run.log 2>&1
  local dt=$(( $(date +%s) - t0 ))
  if grep -q "Success rate:" $OUT/run.log; then
    local ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1    # any evaluation, as run_sweep.sh
    echo "$ok $dt" > $OUT/result.txt
    echo "[gpu$GPU $OUT] success=$ok ${dt}s"
    rmdir "$OUT.claim"
  else
    local why=CRASH; grep -qi "out of memory" $OUT/run.log && why=OOM
    echo "[gpu$GPU $OUT] $why ${dt}s (claim left in place)"
  fi
}

for e in $ENVS; do
  case $e in
    reacher)   M="model_name=lewm_reacher goal_source=random_state"; GL=egl;;
    pointmaze) M="model_name=point_maze goal_source=random_state"; GL=x;;
    wall)      M="model_name=wall_single goal_source=random_state"; GL=x;;
    pusht)     M="model_name=pusht goal_source=dset"; GL=x;;
    cube)      M="model_name=lewm_cube goal_source=dset"; GL=egl;;
  esac
  if [ "$GL" = egl ]; then export MUJOCO_GL=egl; unset DISPLAY; else unset MUJOCO_GL; export DISPLAY=:0; fi
  HS=$HORIZONS; [ $e = cube ] && HS=5
  for H in $HS; do
    # GD-12 backward chunk: exact, memory only (same sizes as the EM table runs)
    case $e:$H in
      pointmaze:5|pusht:5) CH=8;; pointmaze:10|pusht:10) CH=4;; pointmaze:15|pusht:15) CH=2;; *) CH=8;;
    esac
    SEEDS=$(seq 1 100); [ $e = cube ] && SEEDS=$(seq 1 50)
    for s in $SEEDS; do
      for p in $PLANNERS; do
        if [ $e = cube ]; then OUT=$PWD/sweep_cube_leflow/$p/seed_$s
        else OUT=$PWD/sweep/$e/${p}_H$H/seed_$s; fi
        case $p in
          gd12)
            GDP=gd; [ $e = cube ] && GDP=gd_cube
            run_one $OUT $M goal_H=$H seed=$s planner=$GDP planner.opt_steps=150 planner.eval_every=1 \
              planner.hjb_tau=0.0 ++planner.num_candidates=12 ++planner.cand_chunk=$CH ;;
          gds12)
            # Smoothed GD: ONE trajectory, 12 Gaussian-perturbed copies per step,
            # gradient averaged over them (noise 1.0 annealed to 0, as gd_cube).
            # 12 rollouts/step like JALA-EM; no population, no selection.
            GDP=gd; [ $e = cube ] && GDP=gd_cube
            run_one $OUT $M goal_H=$H seed=$s planner=$GDP planner.opt_steps=150 planner.eval_every=1 \
              planner.hjb_tau=0.0 planner.num_samples=12 planner.var_actions=1.0 \
              ++planner.num_candidates=1 ++planner.cand_chunk=$CH ;;
          grasp12)
            run_one $OUT $M goal_H=$H seed=$s planner=grasp planner.opt_steps=150 planner.eval_every=1 \
              ++planner.num_parallel_trajectories=12 ;;
        esac
      done
    done
    echo "=== gpu$GPU $e H$H done $(date +%T) ==="
  done
done
echo "=== gpu$GPU BASELINES WORKER DONE $(date +%T) ==="
