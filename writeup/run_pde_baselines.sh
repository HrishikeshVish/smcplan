#!/usr/bin/env bash
# Reviewer WT5H Q2: does the PDE roadmap cost also improve the baselines?
# Runs the Table-1 12-candidate arms of GD, CEM and MPPI with the PDE cost
# added to their objective (planning/roadmap_cost.py), on three cells:
# Point maze H=10, Push-T H=5, Reacher H=15.  Results land in
# sweep/<env>/<arm>_pde_H<H>/seed_<s>/, the same layout the L2 arms use, so
# writeup/equal_particle_stats.py can read both.
#
# GRASP is excluded: its objective never calls objective_fn (goal_loss is
# identically zero, grasp.py:962), so the wrapper cannot reach it.
#
# Usage: GPU=0 CELLS="pointmaze:10 pusht:5" bash writeup/run_pde_baselines.sh
set -u
cd /home/hviswan/Documents/grasp
source ~/miniconda3/etc/profile.d/conda.sh && conda activate grasp
GPU=${GPU:-0}
CELLS=${CELLS:-"pointmaze:10 pusht:5 reacher:15"}
PLANNERS=${PLANNERS:-"gds12 cem12 mppi12"}
SEEDS=${SEEDS:-$(seq 1 100)}
PER_RUN_TIMEOUT=${PER_RUN_TIMEOUT:-1800}
export CUDA_VISIBLE_DEVICES=$GPU
CKPT=/home/hviswan/Documents/grasp/checkpoints
BUF=/home/hviswan/Documents/grasp/latent_buffers

pde_args () {   # per-env roadmap: the (k, c) the Table-1 PDE runs use
  case $1 in                                    # writeup/run_emsmc_pde.sh:52-55
    pointmaze) echo "+pde_cost.replay_buffer_path=$BUF/pointmaze_buffer.pt +pde_cost.latent_key=proprio +pde_cost.graph_n_nodes=3800 +pde_cost.graph_k_neighbors=4 +pde_cost.heat_c=1.0" ;;
    pusht)     echo "+pde_cost.replay_buffer_path=$BUF/pusht_offline_replay_buffer.pt +pde_cost.latent_key=proprio +pde_cost.graph_n_nodes=55000 +pde_cost.graph_k_neighbors=64 +pde_cost.heat_c=10.0" ;;
    reacher)   echo "+pde_cost.replay_buffer_path=$BUF/lewm_reacher_fixed.pt +pde_cost.latent_key=visual +pde_cost.graph_n_nodes=55000 +pde_cost.graph_k_neighbors=16 +pde_cost.heat_c=3.0" ;;
  esac
  echo "+pde_cost.use_full_dim=true +pde_cost.solver_mode=heat +pde_cost.heat_t_rule=degree +pde_cost.surrogate_steps=3000 +pde_cost.v_traj=all +pde_cost.v_base=2.0"
}

for cell in $CELLS; do
  env=${cell%%:*}; H=${cell##*:}
  case $env in                                  # writeup/run_baselines_ms.sh:46-49
    pointmaze) M="model_name=point_maze goal_source=random_state";   unset MUJOCO_GL; export DISPLAY=:0 ;;
    pusht)     M="model_name=pusht goal_source=dset";                unset MUJOCO_GL; export DISPLAY=:0 ;;
    reacher)   M="model_name=lewm_reacher goal_source=random_state"; export MUJOCO_GL=egl; unset DISPLAY ;;
  esac
  case $env:$H in
    pointmaze:10) CH=4 ;; pusht:5) CH=8 ;; *) CH=8 ;;
  esac
  for p in $PLANNERS; do
    case $p in
      gds12)  PARGS="planner=gd planner.opt_steps=150 planner.eval_every=1 planner.hjb_tau=0.0 planner.num_samples=12 planner.var_actions=1.0 ++planner.num_candidates=1 ++planner.cand_chunk=$CH" ;;
      cem12)  PARGS="planner=cem planner.opt_steps=150 planner.num_samples=12 planner.topk=3 planner.eval_every=1" ;;
      mppi12) PARGS="planner=mppi planner.opt_steps=150 planner.num_samples=12 planner.eval_every=1" ;;
    esac
    for s in $SEEDS; do
      OUT=$PWD/sweep/$env/${p}_pde_H$H/seed_$s
      [ -s "$OUT/result.txt" ] && continue
      mkdir -p "$OUT"
      t0=$(date +%s)
      GRASP_RESULTS_DIR="$OUT" timeout $PER_RUN_TIMEOUT python plan.py \
        $M n_evals=1 goal_H=$H seed=$s \
        save_images=False $PARGS ckpt_base_path=$CKPT $(pde_args $env) \
        > "$OUT/run.log" 2>&1
      dt=$(( $(date +%s) - t0 ))
      if grep -q "Success rate:" "$OUT/run.log" 2>/dev/null; then
        ok=0; grep -q "Success rate:  1.0" "$OUT/run.log" && ok=1
        echo "$ok $dt" > "$OUT/result.txt"
        echo "[gpu$GPU $env $p H$H s$s] ${dt}s success=$ok"
      else
        echo "[gpu$GPU $env $p H$H s$s] FAILED ${dt}s: $(grep -oE '[A-Za-z]*Error[^"]*' "$OUT/run.log" | head -1 | cut -c1-90)"
      fi
    done
    echo "=== gpu$GPU $env $p H$H done $(date +%T) ==="
  done
done
echo "=== gpu$GPU PDE-BASELINE WORKER DONE $(date +%T) ==="
