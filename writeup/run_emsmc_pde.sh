#!/bin/bash
# PDE-cost arm (lambda_V auto, trajectory-weighted Vhat, per-env calibrated roadmap)
# of the JALA-EM table; otherwise identical to run_emsmc_table.sh. Calibration:
# writeup/calib_pde.py + calib_surrogate.py (offline episodes only, no planning seeds).
# Original header follows.
# JALA-EM ("Ours") arm of the main planner-comparison table:
# {reacher, pointmaze, wall, pusht} x H in {5,10,15} x seeds 1..100, protocol-
# matched to the recorded baselines and to the FK table it replaces
# (writeup/run_fk_sweep_fixed.sh): same model names, same per-env goal source,
# 150 iterations, eval_every=1, one episode per seed.
#
# Planner config is the validated JALA-EM setting, pinned on the command line so
# no YAML default can leak in: ULA + Jarzynski weights, identity metric,
# eps=0.18 fixed, P=12, adaptive beta capped at 1, score M-step, lambda_v=0
# (the L2 cost the baselines use; the roadmap cost-to-go is a separate arm).
#
# Work queue: every worker walks the same ordered list and claims a seed with an
# atomic mkdir, so any number of workers on any GPUs can share it, and a re-run
# after an interrupt skips finished seeds (non-empty result.txt). A claim whose
# worker died leaves a stale .claim dir; remove it to requeue the seed.
#
# Usage: GPU=0 bash writeup/run_emsmc_table.sh   (start one worker per slot)
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=${GPU:?set GPU}
ROOT=${ROOT:-$PWD/sweep_emsmc/table}
ENVS=${ENVS:-"pointmaze wall reacher pusht"}
HORIZONS=${HORIZONS:-"5 10 15"}
SEEDS=${SEEDS:-$(seq 1 100)}
LAMBDA_V=${LAMBDA_V:-auto}
ARM=${ARM:-pde}
EXTRA=${EXTRA:-}
NEED_MIB=${NEED_MIB:-6000}
CK=ckpt_base_path=$PWD/checkpoints
PIN="++planner.kernel=ula planner.precond=none planner.kernel_step_size=0.18 \
planner.num_particles=12 planner.beta_schedule=adaptive planner.beta_max=1.0 \
planner.m_step=score ++planner.lambda_v=$LAMBDA_V ++planner.v_traj=all ++planner.v_base=2.0 ++planner.heat_t_rule=degree ++planner.surrogate_steps=3000 planner.opt_steps=150 \
planner.eval_every=1 planner.stop_on_success=true"

gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU; }

for e in $ENVS; do
  case $e in
    reacher)   MODEL="model_name=lewm_reacher planner=emsmc_reacher goal_source=random_state"; GL=egl;;
    pointmaze) MODEL="model_name=point_maze planner=emsmc_pointmaze goal_source=random_state"; GL=x;;
    wall)      MODEL="model_name=wall_single planner=emsmc_wall goal_source=random_state"; GL=x;;
    pusht)     MODEL="model_name=pusht planner=emsmc_pusht goal_source=dset"; GL=x;;
    *) echo "bad env $e"; continue;;
  esac
  case $e in   # calibrated roadmap per env (k, c): see sweep_emsmc/calib_*.log
    pointmaze) PDE="planner.graph_k_neighbors=4 ++planner.heat_c=1.0";;
    wall)      PDE="planner.graph_k_neighbors=32 ++planner.heat_c=30.0";;
    pusht)     PDE="planner.graph_k_neighbors=64 ++planner.heat_c=10.0";;
    reacher)   PDE="planner.graph_k_neighbors=16 ++planner.heat_c=3.0";;
  esac
  if [ "$GL" = egl ]; then export MUJOCO_GL=egl; unset DISPLAY; else unset MUJOCO_GL; export DISPLAY=:0; fi
  for H in $HORIZONS; do
    # Rollout chunk: exact (rows are independent), it only caps peak memory. The
    # DINO-WM ViT rollout retained for autograd grows superlinearly with H:
    # pointmaze/pusht OOM a 16 GB card at H=15 with chunk 8 AND chunk 4 (pilot
    # 2026-09-28), as FK did at 3 particles. Wall (H15, chunk 8) and the
    # single-token LeWM reacher fit. Override with CHUNK=<n>.
    case $e:$H in
      pointmaze:5|pusht:5) CH=8;; pointmaze:10|pusht:10) CH=4;; pointmaze:15|pusht:15) CH=2;;
      *) CH=8;;
    esac
    CH=${CHUNK:-$CH}
    for s in $SEEDS; do
      OUT=$ROOT/$e/${ARM}_H$H/seed_$s
      [ -s "$OUT/result.txt" ] && continue
      mkdir -p "$(dirname $OUT)"
      mkdir "$OUT.claim" 2>/dev/null || continue          # another worker has it
      while [ "$(gpu_free)" -lt "$NEED_MIB" ]; do sleep 20; done
      mkdir -p "$OUT"; t0=$(date +%s)
      CUDA_VISIBLE_DEVICES=$GPU GRASP_RESULTS_DIR=$OUT timeout 5400 python plan.py \
        n_evals=1 seed=$s goal_H=$H save_images=False $MODEL $PIN $PDE planner.eval_chunk=$CH $EXTRA $CK \
        > $OUT/run.log 2>&1
      dt=$(( $(date +%s) - t0 ))
      # success from the planner's own write, never a run.log grep
      if [ -s "$OUT/seed_success_log.txt" ]; then
        read ok it < <(tail -1 $OUT/seed_success_log.txt | awk '{print ($1>=1.0)?1:0, $5}')
        echo "$ok $it $dt" > $OUT/result.txt
        echo "[gpu$GPU $e H$H s$s] success=$ok iter=$it ${dt}s"
        rmdir "$OUT.claim"
      else
        why=CRASH; grep -qi "out of memory" $OUT/run.log && why=OOM
        echo "[gpu$GPU $e H$H s$s] $why ${dt}s (claim left in place)"
      fi
    done
    echo "=== gpu$GPU $e H$H pass done $(date +%T) ==="
  done
done
echo "=== gpu$GPU WORKER DONE $(date +%T) ==="
