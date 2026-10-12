#!/bin/bash
# GC-IDM, official trainer and README hyperparameters (50 epochs, lr 1e-3,
# batch 1024), on the train-split embeddings. frameskip 1 is the official
# closed-loop policy; frameskip 5 (one action block per world-model step,
# goals up to 75 raw steps for H=15) is used for our open-loop adaptation.
cd /home/hviswan/Documents/grasp
PY=~/.venvs/leflow/bin/python; W=$PWD/third_party/work; mkdir -p $W/gcidm
export CUDA_VISIBLE_DEVICES=0
for spec in "reacher 2 1 50" "reacher 2 5 75" "cube 5 1 50" "cube 5 5 50"; do
  set -- $spec; e=$1; ad=$2; fs=$3; mh=$4; OUT=$W/gcidm/${e}_fs${fs}.pt
  [ -s $OUT ] && continue
  (cd third_party/GCIDM && $PY train_idm.py train --embeddings $W/emb/${e}_train.npz --output $OUT \
     --action-dim $ad --frameskip $fs --max-goal-horizon $mh --epochs 50 --lr 1e-3 --batch-size 1024 \
     --device cuda:0) || exit 1
done
