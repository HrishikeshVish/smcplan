# JALA-EM main results (2026-09-30)

Config: ULA+Jarzynski (JALA) E-step, eta=0.18, P=12, adaptive beta<=1, score M-step (Adam lr 0.1),
150 iters, stop_on_success, seeds 1-100, protocol of writeup/run_fk_sweep_fixed.sh.
L2 arm: lambda_V=0 (sweep_emsmc/table/<env>/em_H*). PDE arm: lambda_V=auto, trajectory-weighted Vhat
(base 2), per-env calibrated roadmap k/c (writeup/calib_pde.py, calib_surrogate.py), dedupe, t=c/median(D),
surrogate 3000 steps (sweep_emsmc/table/<env>/pde_H*). p = exact McNemar on paired seeds.

| env | H | FK | L2 | PDE | PDE-only | L2-only | p |
|---|---|---|---|---|---|---|---|
| pointmaze | 5 | 75 | 79 | 88 | 13 | 4 | 0.049 |
| pointmaze | 10 | 81 | 91 | 98 | 7 | 0 | 0.016 |
| pointmaze | 15 | 61 | 95 | 96 | 2 | 1 | 1.0 |
| wall | 5 | 56 | 75 | 75 | 12 | 12 | 1.0 |
| wall | 10 | 64 | 84 | 84 | 9 | 9 | 1.0 |
| wall | 15 | 41 | 89 | 79 | 9 | 19 | 0.087 |
| reacher | 5 | 100 | 95 | 96 | 4 | 3 | 1.0 |
| reacher | 10 | 92 | 72 | 84 | 22 | 10 | 0.05 |
| reacher | 15 | 50 | 53 | 71 | 31 | 13 | 0.01 |
| pusht | 5 | 68 | 71 | 68 | 14 | 17 | 0.72 |
| pusht | 10 | 27 | 18 | 15 | 6 | 9 | 0.607 |
| pusht | 15 | 5 | 3 | 3 | 1 | 1 | 1.0 |

## Cube (LeFlow protocol, 50 seeds; non-trivial = block moves > 0.04 m, n=29)

| arm | all % | non-trivial % |
|---|---|---|
| em | 84 | 72 |
| fk | 88 | 79 |
| gd | 82 | 69 |
| mpc_cem | 76 | 59 |
| grasp | 50 | 14 |
