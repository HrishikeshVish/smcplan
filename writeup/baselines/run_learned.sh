#!/bin/bash
# Learned baselines under our protocol, same seeds and output layout as the
# other rows (sweep/<env>/<arm>_H<H>/seed_<s>, Cube: sweep_cube_leflow/<arm>):
#   leflow12  LeFlow with 12 sampled paths (the particle count of every planner)
#   leflow64  LeFlow with its default 64 paths
#   gcidm_ol  GC-IDM chained through the frozen LeWM, open-loop (our adaptation)
#   gcidm_cl  GC-IDM acting in closed loop at every env step (released usage; reference)
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(dirname "$0")/../.."
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MUJOCO_GL=egl; unset DISPLAY
GPU=${GPU:-0}; W=$PWD/third_party/work; CK=ckpt_base_path=$PWD/checkpoints
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU; }
for e in reacher cube; do
  case $e in reacher) M="model_name=lewm_reacher goal_source=random_state"; HS="5 10 15"; SEEDS=$(seq 1 100);;
             cube)    M="model_name=lewm_cube goal_source=dset"; HS=5; SEEDS=$(seq 1 50);; esac
  for H in $HS; do for arm in ${ARMS:-leflow12 leflow64 gcidm_ol gcidm_cl}; do
    case $arm in
      leflow12) P="planner=leflow planner.checkpoint_root=$W/leflow/$e planner.num_samples=12";;
      leflow64) P="planner=leflow planner.checkpoint_root=$W/leflow/$e planner.num_samples=64";;
      gcidm_ol) P="planner=gcidm planner.checkpoint_root=$W/gcidm/$e planner.mode=open_loop";;
      gcidm_cl) P="planner=gcidm planner.checkpoint_root=$W/gcidm/$e planner.mode=closed_loop";;
    esac
    for s in $SEEDS; do
      if [ $e = cube ]; then OUT=$PWD/sweep_cube_leflow/$arm/seed_$s; else OUT=$PWD/sweep/$e/${arm}_H$H/seed_$s; fi
      [ -s "$OUT/result.txt" ] && continue
      mkdir -p $OUT
      while [ "$(gpu_free)" -lt 4000 ]; do sleep 20; done
      t0=$(date +%s)
      CUDA_VISIBLE_DEVICES=$GPU GRASP_RESULTS_DIR=$OUT timeout 1800 python plan.py n_evals=1 save_images=False \
        $M goal_H=$H seed=$s $P $CK > $OUT/run.log 2>&1
      dt=$(( $(date +%s) - t0 ))
      if grep -q "Success rate:" $OUT/run.log; then
        ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1; echo "$ok $dt" > $OUT/result.txt
      else echo "[$e $arm H$H s$s] CRASH ${dt}s"; fi
    done
    echo "=== $e $arm H$H done $(date +%T) ==="
  done; done
done
