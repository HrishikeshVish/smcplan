#!/bin/bash
# Check out the third-party repositories, which are pinned submodules rather than
# vendored copies. None of them is needed to plan with this repo or to rebuild any
# table; see third_party/README.md for what each one is for.
#
#   bash scripts/baselines.sh            # all of them
#   bash scripts/baselines.sh LeFlow     # one, by directory name
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_ROOT"
if [ $# -ge 1 ]; then
  git submodule update --init --recursive "third_party/$1"
else
  git submodule update --init --recursive third_party
fi
git submodule status third_party
