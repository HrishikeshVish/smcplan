#!/bin/bash
# Fetch the datasets and the pretrained world models.
#
#   bash scripts/download_data.sh <base>          # e.g. /scratch/$USER/grasp
#
# Runs install.sh, which pulls the DINO-WM datasets and checkpoints from the OSF
# bucket published with DINO-WM (https://osf.io/bmw48/) into
#   <base>/datasets   Point maze, Push-T, Wall, and the LeWM dmc/ogbench data
#   <base>/outputs    point_maze, pusht, wall_single, lewm_reacher, lewm_cube
# and then links them where the configs look by default, unless <repo>/data or
# <repo>/checkpoints already exists.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_ROOT"
[ $# -ge 1 ] || { sed -n 2,12p "$0"; exit 1; }
BASE=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")

./install.sh --dataset-dir "$BASE" --download-datasets

link() {  # link <target> <name>, never over an existing directory
  if [ -e "$2" ]; then echo "[grasp] $2 exists, leaving it alone"; else ln -s "$1" "$2"; echo "[grasp] $2 -> $1"; fi
}
link "$BASE/datasets" data
link "$BASE" checkpoints

cat <<MSG

Done. The defaults now resolve; to point elsewhere instead, export
  DATASET_DIR=$BASE/datasets
  GRASP_CKPT_BASE=$BASE
Next: bash scripts/prepare_buffers.sh all
MSG
