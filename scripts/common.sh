#!/bin/bash
# Shared settings of the scripts. Override any of these from the environment.
#
#   PYTHON            interpreter                                  (default: python)
#   DATASET_DIR       datasets                                     (default: <repo>/data)
#   GRASP_CKPT_BASE   world models, loaded from $GRASP_CKPT_BASE/outputs  (default: <repo>/checkpoints)
#   GRASP_BUFFER_DIR  latent roadmap buffers                       (default: <repo>/latent_buffers)
#
# conf/plan*.yaml and conf/planner/*.yaml read the last two through ${oc.env:...},
# so setting them here is enough; nothing holds an absolute path.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${PYTHON:-python}
export DATASET_DIR=${DATASET_DIR:-$REPO_ROOT/data}
export GRASP_CKPT_BASE=${GRASP_CKPT_BASE:-$REPO_ROOT/checkpoints}
export GRASP_BUFFER_DIR=${GRASP_BUFFER_DIR:-$REPO_ROOT/latent_buffers}
