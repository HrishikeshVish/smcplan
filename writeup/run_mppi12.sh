#!/bin/bash
# Equal-particle MPPI (official stable_worldmodel solver, planning/mppi.py): 12
# samples per iteration x 150 iterations, solver defaults otherwise (temperature
# 0.5, var_scale 1, all samples weighted since topk=30 >= 12).
# Protocol as the other baseline rows (writeup/run_sweep.sh): per-env model and
# goal source, eval_every=1, seeds 1..100, H in {5,10,15}; cube open-loop CEM
# under LeFlow's goal protocol (goal_H=5, dset, seeds 1..50).
# Claim queue shared by any number of workers (GPU=<g>); result.txt = "ok dt".
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=${GPU:?set GPU}
NEED_MIB=${NEED_MIB:-8000}
ENVS=${ENVS:-"pointmaze wall reacher pusht cube"}
HORIZONS=${HORIZONS:-"5 10 15"}
CK=ckpt_base_path=$PWD/checkpoints
PLAN="planner=mppi planner.opt_steps=150 planner.num_samples=12 planner.eval_every=1"
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU; }
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
    SEEDS=$(seq 1 100); [ $e = cube ] && SEEDS=$(seq 1 50)
    for s in $SEEDS; do
      if [ $e = cube ]; then OUT=$PWD/sweep_cube_leflow/mppi12/seed_$s
      else OUT=$PWD/sweep/$e/mppi12_H$H/seed_$s; fi
      [ -s "$OUT/result.txt" ] && continue
      mkdir -p "$(dirname $OUT)"
      mkdir "$OUT.claim" 2>/dev/null || continue
      while [ "$(gpu_free)" -lt "$NEED_MIB" ]; do sleep 20; done
      mkdir -p "$OUT"; t0=$(date +%s)
      CUDA_VISIBLE_DEVICES=$GPU GRASP_RESULTS_DIR=$OUT timeout 5400 python plan.py \
        n_evals=1 save_images=False $M goal_H=$H seed=$s $PLAN $CK > $OUT/run.log 2>&1
      dt=$(( $(date +%s) - t0 ))
      if grep -q "Success rate:" $OUT/run.log; then
        ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
        echo "$ok $dt" > $OUT/result.txt; echo "[gpu$GPU $OUT] success=$ok ${dt}s"; rmdir "$OUT.claim"
      else
        why=CRASH; grep -qi "out of memory" $OUT/run.log && why=OOM
        echo "[gpu$GPU $OUT] $why ${dt}s (claim left in place)"
      fi
    done
    echo "=== gpu$GPU mppi12 $e H$H done $(date +%T) ==="
  done
done
echo "=== gpu$GPU MPPI12 WORKER DONE $(date +%T) ==="
