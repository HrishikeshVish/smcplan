"""Split EM planner failures into COVERAGE vs SELECTION, and estimate the
z->s channel that bounds both.

Consumes the npz traces written by planning/em_smc.py under GRASP_EM_ORACLE.
Each trace holds, at every checkpoint step, all P particles' latent objective
(`task`), composite cost, roadmap V, real-env success, and true goal distance
(`d_true`), plus the index our argmin-task rule selected.

Why this and not the SMC metrics: ESS, acceptance, Sinkhorn and log Zhat all
score how well we approximate pi, which lives in LATENT space. Success lives in
state space. A run can pass every sampler diagnostic and still fail -- observed
directly (seed 6: ESS 15-22/32, acceptance on target, 46 resamples, 29.5 nats of
selection pressure, success 0). These three numbers are what the sampling
metrics cannot give:

  coverage        max_i success(a_i)   -- did the population CONTAIN a winner?
                                          the kernel's job
  selection       success(a_sel), and d_true[sel] - min_i d_true
                                       -- did we PICK it? the proxy's job
  z->s fidelity   Spearman(task_i, d_true_i) across the population
                                       -- CAN the latent objective rank
                                          candidates at all? caps both above

Usage:  python writeup/analyze_emsmc_oracle.py <trace.npz> [more.npz ...]
        python writeup/analyze_emsmc_oracle.py <dir>     # groups by kernel
"""
import os
import sys

import numpy as np


def spearman(x, y):
    """Rank correlation, nan-safe, no scipy dependency."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    x, y = x[m], y[m]
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else np.nan


def collect(paths):
    """Per-checkpoint rows, tagged by kernel so arms can be compared."""
    rows = []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        step, task, dt, suc = z["step"], z["task"], z["d_true"], z["success"]
        sel = z["sel_idx"]
        kern = str(z["kernel"]) if "kernel" in z.files else "?"
        fin = float(z["final_success"]) if "final_success" in z.files else np.nan
        smc = float(z["softmin_cost"]) if "softmin_cost" in z.files else np.nan
        for k in range(len(step)):
            cov = float(np.nanmax(suc[k]))
            sels = float(suc[k][sel[k]])
            best_d = float(np.nanmin(dt[k]))
            sel_d = float(dt[k][sel[k]])
            rho = spearman(np.asarray(task[k], float), np.asarray(dt[k], float))
            rows.append(dict(trace=p.split("/")[-1], kernel=kern,
                             seed=p.split("/")[-2], step=int(step[k]),
                             coverage=cov, selected=sels,
                             regret=sel_d - best_d, best_d=best_d,
                             sel_d=sel_d, rho=rho, final_success=fin,
                             softmin_cost=smc))
    return rows


def selection_delay(rows):
    """Per seed: first checkpoint the population CONTAINED a winner vs the first
    one where we PICKED a winner. Their gap is what selection regret actually
    costs, in iterations.

    A per-checkpoint selection miss is not a planning failure -- the planner
    re-selects every iteration and needs one hit -- so the per-checkpoint rate
    overstates the harm. Coverage decides whether a seed is solvable at all;
    this delay decides how long it takes.
    """
    print(f"\n--- coverage vs selection timing (per seed) ---")
    print(f"{'kernel':>7} {'seed':>9} {'1st cover':>10} {'1st picked':>11} "
          f"{'delay':>7} {'success':>8}")
    keys = sorted({(r["kernel"], r["seed"]) for r in rows})
    delays = []
    for k, sd in keys:
        rk = sorted([r for r in rows if r["kernel"] == k and r["seed"] == sd],
                    key=lambda r: r["step"])
        fc = next((r["step"] for r in rk if r["coverage"] > 0), None)
        fs = next((r["step"] for r in rk if r["selected"] > 0), None)
        d = (fs - fc) if (fc is not None and fs is not None) else None
        if d is not None:
            delays.append(d)
        print(f"{k:>7} {sd:>9} {str(fc):>10} {str(fs):>11} "
              f"{('never' if fs is None and fc is not None else str(d)):>7} "
              f"{rk[0]['final_success']:>8.0f}")
    if delays:
        print(f"\n  mean selection delay where both occur: "
              f"{np.mean(delays):.1f} iterations "
              f"(checkpoint resolution, so a lower bound on resolution)")
    ncov = sum(1 for k, sd in keys
               if any(r["coverage"] > 0 for r in rows
                      if r["kernel"] == k and r["seed"] == sd))
    nsel = sum(1 for k, sd in keys
               if any(r["selected"] > 0 for r in rows
                      if r["kernel"] == k and r["seed"] == sd))
    print(f"  seed-arms with coverage at some checkpoint: {ncov}/{len(keys)}")
    print(f"  seed-arms that ever picked a winner:        {nsel}/{len(keys)}")


def per_kernel(rows):
    """The Q1/Q2 table: coverage per kernel, and whether it reaches success."""
    ks = sorted({r["kernel"] for r in rows})
    print(f"\n--- per kernel ---")
    print(f"{'kernel':>7} {'ckpts':>6} {'coverage':>9} {'selected':>9} "
          f"{'sel|cov':>8} {'min d':>7} {'regret':>7} {'rho':>6} "
          f"{'softminC':>9} {'success':>8}")
    for k in ks:
        rk = [r for r in rows if r["kernel"] == k]
        cov = np.array([r["coverage"] for r in rk])
        sel = np.array([r["selected"] for r in rk])
        bd = np.array([r["best_d"] for r in rk], float)
        rg = np.array([r["regret"] for r in rk], float)
        rho = np.array([r["rho"] for r in rk], float)
        sm = np.array([r["softmin_cost"] for r in rk], float)
        seeds = {r["seed"]: r["final_success"] for r in rk}
        selcov = sel[cov > 0].mean() if (cov > 0).any() else np.nan
        print(f"{k:>7} {len(rk):>6} {cov.mean():>8.1%} {sel.mean():>8.1%} "
              f"{selcov:>8.1%} {np.nanmean(bd):>7.3f} {np.nanmean(rg):>7.3f} "
              f"{np.nanmean(rho):>+6.2f} {np.nanmean(sm):>9.4f} "
              f"{np.nansum(list(seeds.values())):>3.0f}/{len(seeds):<4}")
    print("\n  Q1: lower softminC / higher coverage = better candidate generation.")
    print("  Q2: a kernel leading on coverage WITHOUT leading on success means the")
    print("      advantage does not transmit -- read sel|cov and rho to see whether")
    print("      selection or the z->s channel is eating it.")


def main(paths):
    import glob
    if len(paths) == 1 and os.path.isdir(paths[0]):
        paths = sorted(glob.glob(os.path.join(paths[0], "*", "seed_*", "oracle.npz")))
        print(f"found {len(paths)} oracle traces under {sys.argv[1]}")
    rows = collect(paths)
    rhos = [r["rho"] for r in rows]
    if not rows:
        print("no checkpoints found")
        return

    if len(rows) <= 40:
        print(f"{'kernel':>6} {'seed':>9} {'step':>5} {'cover':>6} {'sel':>4} "
              f"{'min d':>7} {'d[sel]':>7} {'regret':>7} {'rho':>6}")
        for r in rows:
            print(f"{r['kernel']:>6} {r['seed']:>9} {r['step']:>5} "
                  f"{r['coverage']:>6.0f} {r['selected']:>4.0f} "
                  f"{r['best_d']:>7.3f} {r['sel_d']:>7.3f} "
                  f"{r['regret']:>7.3f} {r['rho']:>6.2f}")

    cov = np.array([r["coverage"] for r in rows])
    sel = np.array([r["selected"] for r in rows])
    reg = np.array([r["regret"] for r in rows])
    rho = np.array([r for r in rhos], dtype=float)
    n = len(rows)
    print(f"\n--- aggregate over {n} checkpoints ---")
    print(f"  coverage (population contained a winner): {cov.mean():.1%}")
    print(f"  selected a winner:                        {sel.mean():.1%}")
    both = cov > 0
    if both.any():
        print(f"  of the {both.sum()} with coverage, selection found it: "
              f"{sel[both].mean():.1%}   <-- SELECTION efficiency")
    print(f"  mean selection regret in true distance:   {np.nanmean(reg):.3f}")
    print(f"  Spearman(latent objective, true distance): "
          f"mean {np.nanmean(rho):+.3f}  median {np.nanmedian(rho):+.3f}")
    print(f"     fraction of checkpoints with rho > 0.3: "
          f"{np.mean(rho > 0.3):.1%}")
    print("\n  Reading: coverage ~0 => the KERNEL never produces a winner, so rank")
    print("  kernels on log Zhat / coverage. coverage high but selected low =>")
    print("  the latent objective cannot identify it, and kernel gains are capped")
    print("  no matter which sampler wins on coverage.")
    if len({r["kernel"] for r in rows}) > 1:
        per_kernel(rows)
    selection_delay(rows)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
