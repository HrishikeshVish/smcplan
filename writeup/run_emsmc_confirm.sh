#!/bin/bash
# Confirmation run: the selected EM config on ALL 100 seeds of the recorded
# FK baseline (sweep_fk/table/pointmaze/fk_H5), so the success-rate comparison
# is paired on the same seeds at an n where it can actually be tested.
# The 20-seed probes are config selection only -- with 3-4 discordant pairs
# McNemar has essentially no power, so nothing there can establish
# non-degradation either way.
#
# ARM (label), GPU, ARGS (planner overrides), HALF (1|2, to split across GPUs).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd /home/hviswan/Documents/grasp
export DATASET_DIR=/home/hviswan/Documents/grasp/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset MUJOCO_GL; export DISPLAY=:0
ARM=${ARM:?set ARM}; GPU=${GPU:-0}; ARGS=${ARGS:?set ARGS}
ROOT=$PWD/sweep_emsmc/pointmaze_H5_100/$ARM
CK=ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints
ALL=$(ls -d sweep_fk/table/pointmaze/fk_H5/seed_* | sed 's/.*seed_//' | sort -n)
N=$(echo "$ALL" | wc -l)
case "${HALF:-0}" in
  1) SEEDS=$(echo "$ALL" | head -n $((N/2)));;
  2) SEEDS=$(echo "$ALL" | tail -n +$((N/2+1)));;
  *) SEEDS=$ALL;;
esac
for s in $SEEDS; do
  OUT=$ROOT/seed_$s
  [ -s "$OUT/result.txt" ] && continue
  mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=$GPU GRASP_RESULTS_DIR=$OUT \
    timeout 3600 python plan.py model_name=point_maze n_evals=1 \
    planner=emsmc_pointmaze goal_H=5 goal_source=random_state seed=$s \
    save_images=False planner.opt_steps=150 planner.eval_every=1 \
    planner.num_particles=3 $ARGS $CK > $OUT/run.log 2>&1
  # Success from seed_success_log.txt field 1 (the planner's own selected-action
  # evaluation), never from a run.log grep -- the oracle probe's per-particle
  # eval_actions calls each print "Success rate:" and would be counted as success.
  if [ -s "$OUT/seed_success_log.txt" ]; then
    ok=$(tail -1 $OUT/seed_success_log.txt | awk '{print ($1>=1.0)?1:0}')
    echo "$ok $(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')" > $OUT/result.txt
    echo "[$ARM s$s] success=$ok"
  else
    why=OOM; grep -qi "out of memory" $OUT/run.log || why=CRASH
    echo "[$ARM s$s] $why -- not recorded"
  fi
done
echo "=== CONFIRM DONE arm=$ARM half=${HALF:-0} $(date +%H:%M) ==="
