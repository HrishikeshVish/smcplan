# Kernel ablation (appendix), 2026-10-07

Fixed JALA-EM setting (pathwise M-step, m_smooth_samples=1, calibrated PDE cost,
P=12, adaptive beta<=1, 150 iters); only the E-step kernel changes. H=10 on the four
table envs (100 paired seeds), cube H=5 under the LeFlow protocol (50 seeds).
Runs: sweep_emsmc/table/<env>/kfix_<kernel>_H10, sweep_cube_leflow/kfix_<kernel>;
JALA = pdefix. p = exact McNemar on paired seeds. "iters" = mean iterations to
stop (stop at first success; HMC iterations cost 5 rollouts each).

| cell | JALA | MALA | RWM | HMC | JALA vs MALA | JALA vs RWM | JALA vs HMC | iters JALA/MALA/RWM/HMC |
|---|---|---|---|---|---|---|---|---|
| pointmaze H10 | 95 | 85 | 79 | 91 | +10/-0 p=0.002 | +19/-3 p=0.001 | +5/-1 p=0.22 | 14/31/48/26 |
| wall H10 | 81 | 79 | 60 | 89 | +14/-12 p=0.85 | +26/-5 p<0.001 | +4/-12 p=0.077 | 43/57/80/40 |
| reacher H10 | 100 | 98 | 84 | 100 | +2/-0 p=0.50 | +16/-0 p<0.001 | +0/-0 | 24/32/67/22 |
| pusht H10 | 32 | 23 | 12 | 25 | +13/-4 p=0.049 | +21/-1 p<0.001 | +12/-5 p=0.14 | 119/128/139/127 |
| cube H5 | 88 | 84 | 84 | 86 | +2/-0 p=0.50 | +3/-1 p=0.63 | +2/-1 p=1.0 | 20/30/32/23 |
