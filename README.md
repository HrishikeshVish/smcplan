<div align="center">

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>Latent Space Planning with Sequential Monte Carlo</h1>
      <p><a href="https://hrishikeshvish.github.io/">Hrishikesh Viswanath</a><sup>1</sup> &emsp; Anindita Ghosh<sup>2</sup> &emsp; Uttaran Bhattacharya<sup>3</sup> &emsp; <a href="https://ideas.cs.purdue.edu">Aniket Bera</a><sup>1</sup></p>
      <p><sup>1</sup>Department of Computer Science, Purdue University &emsp; <sup>2</sup>Saarland University &emsp; <sup>3</sup>Adobe Systems</p>
      <i>Preprint, 2026</i>
    </summary>
  </ul>
</div>

<a href="https://hrishikeshvish.github.io/smcplan/"><img src="https://img.shields.io/badge/Project-Page-0F7B8A?style=for-the-badge&labelColor=F0F0F0"/></a> &emsp;
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10-598BE7?style=for-the-badge&logo=python&logoColor=598BE7&labelColor=F0F0F0"/></a> &emsp;
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=EE4C2C&labelColor=F0F0F0"/></a> &emsp;
<a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia&logoColor=76B900&labelColor=F0F0F0"/></a> &emsp;
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-4CAF50?style=for-the-badge&labelColor=F0F0F0"/></a>

<br><br>
<img src="assets/teaser.png" width="850px" alt="Planning as sequential Monte Carlo over action sequences"/>
<br>
<em>Each particle is a whole action sequence. Every iteration moves the population along the
cost-defined target, reweights it, resamples it when the effective sample size drops, and then
updates the plan from the weighted population.</em>
<br><br>

<div id="toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h2><a href="#overview">Overview</a> &emsp; <a href="#method">Method</a> &emsp; <a href="#results">Results</a> &emsp; <a href="#installation">Installation</a> &emsp; <a href="#quick-start">Quick Start</a> &emsp; <a href="#reproducing-the-paper">Reproducing</a> &emsp; <a href="#citation">Citation</a></h2>
    </summary>
  </ul>
</div>

</div>

---

## Overview

Planning in a learned world model usually means gradient descent on one action sequence. A single
trajectory is cheap, but it commits early, and the places where it gets stuck are exactly the
places where the world model is least trustworthy.

We plan with a **population of action sequences** instead. A plan is a mean sequence $\bm\mu$ with
a reference distribution $q(\cdot\mid\bm\mu)$ around it, and tilting that reference by the planning
cost $C$ gives a target

```math
\gamma_{\bm{\mu}}(\mathbf a) = q(\mathbf a \mid \bm{\mu}) \exp\!\big(-C(\mathbf a)/\alpha\big).
```

The plan moves by one optimizer step per iteration, so the targets of consecutive iterations form a
sequence, and sequential Monte Carlo is what carries a population across it. Each particle takes an
unadjusted Langevin step on the current target,

```math
\mathbf a_k^{(i)} = \mathbf a_{k-1}^{(i)} + \eta\, \nabla_{\mathbf a} \log \gamma_k\big(\mathbf a_{k-1}^{(i)}\big) + \sqrt{2\eta}\, \varepsilon_k^{(i)},
```

the population is reweighted by Jarzynski's equality,
$W_k^{(i)} = W_{k-1}^{(i)} G_k\big(\mathbf a_{k-1}^{(i)}, \mathbf a_k^{(i)}\big)$, and resampled when
its effective sample size falls. The plan is then updated along the weighted population gradient
$\mathbf g_k = \sum_i w_k^{(i)} \nabla_{\bm\mu} U_{\bm\mu_k}\big(\mathbf a_k^{(i)}\big)$, which is
the maximization step of the EM view of MPPI. The sampler and the optimizer are two halves of one
iteration: the E-step supplies the weights, the M-step moves the plan.

### Highlights

- **A population, not a trajectory.** Twelve particles explore in parallel, and resampling moves
  effort onto the ones that are working. The whole study runs at twelve candidates and 150
  iterations, which is also what every baseline gets.
- **Four propagation kernels, one planner.** ULA with Jarzynski weights, plus MALA, random-walk
  Metropolis and HMC, ported from [blackjax](https://github.com/blackjax-devs/blackjax) and checked
  against it in `planning/test_mcmc_kernels.py`.
- **A cost-to-go from a PDE on the latent roadmap.** Latent distance to the goal is a poor guide
  when the straight line in latent space crosses a wall. We build a roadmap over buffer latents,
  solve a heat or random-walk equation on it for a cost-to-go $V$, and add $\lambda_V V$ to the
  terminal distance. `planning/roadmap_cost.py`, `build_latent_buffer.py`.
- **Two world-model families.** DINO-WM (Point maze, Push-T, Wall) and LeWM/JEPA (Reacher, Cube),
  behind one planner interface.
- **Five planners in one codebase**, each at its authors' objective: GD, GRASP, CEM, MPPI and ours.
- **Every config behind a reported number ships as-is**, including the per-environment rollout
  chunking that keeps a 16 GB card from running out of memory at $H=15$.

---

## Method

One iteration is move, reweight, resample, update. `planning/em_smc.py` is the planner of the
paper's Algorithm 1; `planning/fk_planner.py` is an earlier Feynman-Kac variant in which the
cost-to-go enters only through the tempered weights, kept because the ablations compare against it.

The cost $C$ is a terminal latent distance plus, optionally, the roadmap cost-to-go. Setting
`planner.lambda_v=0` recovers the plain $L_2$ objective, which is the row the paper reports
alongside the PDE cost so the planner and the objective can be read apart.

<div align="center">
<img src="assets/cost_structures.png" width="820px" alt="PDE cost-to-go against latent distance"/>
<br>
<em>The roadmap cost-to-go against plain latent distance.</em>
</div>

---

## Results

Success rate averaged over the twelve cells of the main table, four environments at horizons 5, 10
and 15, 100 seeds each. Every planner gets twelve candidate action sequences and 150 iterations
over the same pretrained world model, the same seeds and the same start and goal pairs.

<div align="center">
<img src="assets/mean_success.png" width="700px" alt="Mean success rate per planner"/>
</div>

| Planner | Objective | Mean success (%) |
| :--- | :--- | :-: |
| **Ours** | PDE cost-to-go | **79.8** |
| **Ours** | $L_2$ | 72.6 |
| MPPI | as published | 72.3 |
| GD | as published | 50.9 |
| CEM | as published | 45.2 |
| GRASP | as published | 29.9 |

Per-cell numbers, confidence intervals and paired tests are in the paper.

<div align="center">
<img src="assets/success_grid.png" width="850px" alt="Success against iteration"/>
<br>
<em>Fraction of seeds solved at or before each iteration, every planner holding twelve particles.
Each environment is shown at the horizon where the margin over the strongest baseline is largest.</em>
<br><br>
<img src="assets/paired_summary.png" width="700px" alt="Paired per-seed comparison"/>
<br>
<em>Paired per-seed comparison against each baseline. Both arms of a pair share the seed, hence the
start and the goal, so the difference is within-seed.</em>
<br><br>
<img src="assets/kernels.png" width="700px" alt="Propagation kernels compared"/>
<br>
<em>Success rate by propagation kernel, inside the same planner.</em>
</div>

<div align="center">
<table>
<tr>
<td align="center"><img src="assets/frames_pointmaze.png" width="260px"/><br><em>Point maze</em></td>
<td align="center"><img src="assets/frames_pusht.png" width="260px"/><br><em>Push-T</em></td>
<td align="center"><img src="assets/frames_cube.png" width="260px"/><br><em>Cube</em></td>
</tr>
</table>
</div>

---

## Repository Structure

```
.
├── run.sh                        # one run at the paper's protocol (wraps plan.py)
├── plan.py                       # the planning entry point (Hydra)
├── train.py                      # DINO-WM world-model training (legacy path)
├── build_latent_buffer.py        # latent replay buffer -> roadmap cost-to-go
├── grasp_pseudocode.py           # minimal standalone GRASP reference
├── planning/
│   ├── em_smc.py                 # ours: SMC E-step + Adam M-step (Algorithm 1)
│   ├── fk_planner.py             # earlier Feynman-Kac variant
│   ├── mcmc_kernels.py           # ULA / MALA / RWM / HMC, verified against blackjax
│   ├── roadmap_cost.py           # the PDE cost-to-go on the latent roadmap
│   ├── em_target.py              # the tilted target and its incremental weight
│   ├── gd.py grasp.py cem.py mppi.py mpc.py   # baselines
│   ├── objectives.py             # terminal cost, roadmap term, action regularizer
│   └── evaluator.py              # episode rollout and scoring
├── models/                       # DINO-WM encoders/predictors + LeWM (JEPA) wrapper
├── env/                          # gym registrations and wrappers, plus vectorized envs
├── conf/
│   ├── plan.yaml plan_grasp.yaml # top-level entry configs
│   ├── env/*.yaml                # dataset paths and model dims
│   └── planner/*.yaml            # one config per planner per environment
├── writeup/                      # sweep launchers and the table generators
├── figs/                         # the figure generators
└── assets/                       # the images in this README
```

---

## Installation

Tested on Ubuntu with Python 3.10, PyTorch 2.11 and CUDA 12.8.

**1. Environment**
```bash
git clone https://github.com/HrishikeshVish/smcplan.git
cd smcplan
./install.sh --conda --mujoco
conda activate grasp
```

`install.sh --help` lists the steps individually (`--conda`, `--mujoco`, `--dataset-dir`,
`--download-datasets`, `--extract-pusht-frames`). `--all` runs everything.

**2. Data and checkpoints**
```bash
./install.sh --dataset-dir /path/to/base --download-datasets
```
Datasets go to `<base>/datasets` and pretrained world models to `<base>/outputs`, and
`DATASET_DIR=<base>/datasets` is written to your shell profile. The DINO-WM datasets come from the
[OSF bucket](https://osf.io/bmw48/) published with DINO-WM. LeWM checkpoints are bundled, so no
separate install is needed for Reacher and Cube.

**3. Check the install**
```bash
python planning/test_mcmc_kernels.py
```

> [!IMPORTANT]
> `MUJOCO_GL` differs by environment. Reacher and Cube need `MUJOCO_GL=egl`. Point maze, Push-T and
> Wall need it **unset** with a display available (`export DISPLAY=:0`); setting it to `egl` there
> produces silently wrong renders. The sweep scripts in `writeup/` set this per environment.

---

## Quick Start

```bash
./run.sh pointmaze ours 10                        # our planner, Point maze, horizon 10
./run.sh pointmaze ours-l2 10                     # the same cell under the plain L2 cost
./run.sh pointmaze mppi 10                        # a baseline at the same budget
./run.sh pointmaze ours 10 5 planner.kernel=mala  # 5 trials, MALA propagation kernel
```

`./run.sh <env> <planner> <horizon> [n_evals]` sets the three things that are easy to get wrong by
hand and gives every planner the same twelve candidates and 150 iterations. `MUJOCO_GL` must be
`egl` for Reacher and Cube and **unset** for Point maze, Push-T and Wall, where `egl` renders
silently wrong frames. `goal_source` differs per environment. The rollout chunk caps peak memory,
since the DINO-WM rollout kept for autograd grows superlinearly with the horizon and runs a 16 GB
card out of memory at `H=15` without it. Anything after the trial count is passed to Hydra
untouched. The call it builds for the line above is

```bash
python plan.py model_name=point_maze planner=emsmc_pointmaze goal_H=10 \
  goal_source=random_state n_evals=1 save_images=False planner.num_particles=12 \
  planner.opt_steps=150 planner.eval_every=1 planner.lambda_v=auto planner.eval_chunk=4
```

Any config field can be overridden with Hydra's dotted syntax. Results go to
`plan_outputs/<timestamp>_<model_name>_gH<goal_H>/`, or to `$GRASP_RESULTS_DIR` if it is set, which
is how the sweeps collect one directory per seed.

| Environment | `model_name` | `planner` | `goal_source` | World model |
| :--- | :--- | :--- | :--- | :--- |
| Reacher | `lewm_reacher` | `emsmc_reacher` | `random_state` | LeWM |
| Point maze | `point_maze` | `emsmc_pointmaze` | `random_state` | DINO-WM |
| Wall | `wall_single` | `emsmc_wall` | `random_state` | DINO-WM |
| Push-T | `pusht` | `emsmc_pusht` | `dset` | DINO-WM |
| Cube | `lewm_cube` | `emsmc_cube` | `dset` | LeWM |

---

## Reproducing the Paper

Each cell of the main table is 100 seeds of one `plan.py` call. The launchers run them over the
free GPUs, claim each seed with a lock directory so several workers can share a sweep, and skip any
seed whose `result.txt` already exists, so a run can be interrupted and resumed.

```bash
# the main table: four environments, three horizons, 100 seeds
GPUS="0 1" ENVS="pointmaze wall pusht reacher" bash writeup/launch_emsmc_table.sh

# the baselines at the same budget
bash writeup/launch_baselines_then_ablation.sh

# the roadmap cost applied to the baselines
bash writeup/run_pde_baselines.sh
```

Then build the tables and figures from the finished runs:

| Output | Command |
| :--- | :--- |
| Main table | `python writeup/make_equal_particle_tables.py` |
| Ablation tables | `python writeup/make_ablation_tables.py` |
| Cube table | `python writeup/make_cube_table.py` |
| Confidence intervals, paired tests | `python writeup/make_paired_table.py` |
| Success against iteration | `python figs/fig_success_grid.py` |
| Paired per-seed summary | `python figs/fig_paired_summary.py` |
| Kernel comparison | `python figs/fig_kernels.py` |
| Roadmap cost on the baselines | `python writeup/pde_baseline_stats.py && python figs/fig_pde_transfer.py` |
| This README's summary figure | `python figs/fig_readme_summary.py` |

`writeup/equal_particle_stats.py` is the single source of per-seed outcomes that the tables read,
so a number can always be traced back to the seed directory it came from.

---

## Planners

| `./run.sh` | `planner=` | Class | Objective |
| :--- | :--- | :--- | :--- |
| `ours`, `ours-l2` | `emsmc_<env>` | `planning/em_smc.py` | ours, Algorithm 1 |
| | `fk_<env>` | `planning/fk_planner.py` | earlier Feynman-Kac variant |
| `gd` | `gd`, `gd_cube` | `planning/gd.py` | gradient descent on the terminal latent distance (DINO-WM) |
| `grasp` | `grasp` | `planning/grasp.py` | lifted virtual states with dynamics-consistency penalties |
| `cem` | `cem` | `planning/cem.py` | cross-entropy method |
| `mppi` | `mppi` | `planning/mppi.py` | model-predictive path integral |
| | `mpc_cem` | `planning/mpc.py` | receding-horizon CEM |

Each baseline optimizes the objective its own authors report. The roadmap cost can be switched on
for `gd`, `cem` and `mppi` with `planner.lambda_v`, which is the study behind the appendix figure on
whether the cost transfers.

---

## Citation

```bibtex
@article{viswanath2026latentsmc,
  title   = {Latent Space Planning with Sequential Monte Carlo},
  author  = {Viswanath, Hrishikesh and Ghosh, Anindita and
             Bhattacharya, Uttaran and Bera, Aniket},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Acknowledgments

This codebase is a derivative of [DINO-WM](https://dino-wm.github.io/)
([paper](https://arxiv.org/abs/2411.04983)), and reuses its world-model training, datasets and
environment wrappers. The LeWM/JEPA path is built on
[stable-worldmodel](https://github.com/michael-psenka/le-wm-grasp). The GRASP baseline is
[Psenka et al.](https://www.michaelpsenka.io/grasp/). The MCMC kernels are ported from
[blackjax](https://github.com/blackjax-devs/blackjax) and verified against it. The Jarzynski-adjusted
Langevin scheme and the incremental weights follow Cuin et al. If you use this repository, please
also cite DINO-WM.

## License

MIT. See [LICENSE](LICENSE).
