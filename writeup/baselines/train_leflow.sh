#!/bin/bash
# LeFlow, official model/losses/config, trained on the cached embeddings
# (train_leflow_cached.py): Reacher H=5,10,15 and Cube H=5.
cd "$(dirname "$0")/../.."
PY=~/.venvs/leflow/bin/python; W=$PWD/third_party/work; mkdir -p $W/leflow
export CUDA_VISIBLE_DEVICES=0
for spec in "cube 5" "reacher 5" "reacher 10" "reacher 15"; do
  set -- $spec; e=$1; H=$2; OUT=$W/leflow/${e}_h$H
  [ -s $OUT/latent_planner.pt ] && grep -q "^done" $OUT/train.log 2>/dev/null && continue
  case $e in reacher) H5=data/dmc/reacher.h5;; cube) H5=data/ogbench/cube_single_expert.h5;; esac
  mkdir -p $OUT
  $PY writeup/baselines/train_leflow_cached.py --embeddings $W/emb/${e}_train.npz --h5 $H5 \
     --lewm $PWD/checkpoints/outputs/lewm_$e/weights.pt --horizon $H --out $OUT > $OUT/train.log 2>&1 || exit 1
done
