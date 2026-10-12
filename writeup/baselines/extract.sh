#!/bin/bash
# LeWM embeddings of every frame (GC-IDM's official extractor), then the
# train-split episodes only (first 90%, as our roadmaps and splits use).
cd /home/hviswan/Documents/grasp
PY=~/.venvs/leflow/bin/python; W=$PWD/third_party/work/emb; mkdir -p $W
export CUDA_VISIBLE_DEVICES=0
for e in reacher cube; do
  case $e in reacher) H5=data/dmc/reacher.h5;; cube) H5=data/ogbench/cube_single_expert.h5;; esac
  [ -s $W/${e}_train.npz ] && continue
  (cd third_party/GCIDM && $PY train_idm.py extract --checkpoint $OLDPWD/checkpoints/outputs/lewm_$e \
     --h5 $OLDPWD/$H5 --output $W/${e}_all.npz --batch-size 512 --device cuda:0) || exit 1
  $PY writeup/baselines/split_embeddings.py $W/${e}_all.npz $W/${e}_train.npz 0.9 || exit 1
done
