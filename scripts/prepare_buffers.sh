#!/bin/bash
# Build the latent roadmap buffers that the PDE cost-to-go is solved on. One per
# environment, written to $GRASP_BUFFER_DIR under the names the planner configs
# expect. Needs the datasets and checkpoints (scripts/download_data.sh) and a GPU.
#
#   bash scripts/prepare_buffers.sh all
#   bash scripts/prepare_buffers.sh pointmaze|wall|pusht|reacher|cube
#
# Each one encodes dataset states with the frozen world model and stores the
# observed transitions as z_curr/z_next; planning/roadmap_cost.py reads nothing
# else. The expected shape is printed after each build, so a buffer that came out
# a different size is visible immediately. The size matters: emsmc_pointmaze and
# emsmc_wall set graph_n_nodes to the whole buffer (3800 and 12600), so a
# different row count is a different roadmap.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_ROOT"
[ $# -ge 1 ] || { sed -n 2,8p "$0"; exit 1; }
mkdir -p "$GRASP_BUFFER_DIR"

build() {  # build <name> <expected shape> <args...>
  local name=$1 want=$2; shift 2
  echo "[grasp] $name -> $GRASP_BUFFER_DIR/$name  (expect $want)"
  "$PYTHON" build_latent_buffer.py "$@" --output-path "$GRASP_BUFFER_DIR/$name"
  "$PYTHON" - "$GRASP_BUFFER_DIR/$name" <<'PY'
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
print("[grasp] got", {k: tuple(v.shape) for k, v in d.items() if hasattr(v, "shape")})
PY
}

one() {
  case $1 in
    # Point maze encodes only the episodes whose frames are on disk; the builder
    # warns which, and the paper's buffer is the 200 that were extracted.
    pointmaze) build pointmaze_buffer.pt            "z_curr (3800, 10)"    --env pointmaze    --latent-key proprio ;;
    wall)      build wall_buffer.pt                 "z_curr (12600, 10)"   --env wall_single  --latent-key proprio --max-episodes 1400 ;;
    # The shipped Push-T buffer also carries y_curr/y_next (raw proprio) from an
    # earlier script. The roadmap never reads them.
    pusht)     build pusht_offline_replay_buffer.pt "z_curr (994573, 10)"  --env pusht        --latent-key proprio ;;
    reacher)   build lewm_reacher_fixed.pt          "z_curr (360000, 192)" --env lewm_reacher --latent-key visual ;;
    cube)      build lewm_cube_buffer.pt            "z_curr (56000, 192)"  --env lewm_cube    --latent-key visual --max-episodes 1400 ;;
    *) echo "unknown environment $1"; exit 1 ;;
  esac
}

if [ "$1" = all ]; then for e in pointmaze wall pusht reacher cube; do one "$e"; done; else one "$1"; fi
