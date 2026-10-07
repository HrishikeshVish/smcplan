"""Compare the heat-V and raw-objective FK arms on OPTIMALITY, not just success.

Success rate barely moves here by construction (the seeds were chosen so real FK
solves them), so the questions are:
  1. On seeds BOTH arms solve, which reaches the goal in fewer iterations
     (exit_step, the quantity the paper's tab:main reports) and ends closer
     (state_dist)?  Paired, per seed.
  2. What does the tilt actually DO differently?  cumKL (total selection
     pressure in nats), resample count, and final ESS are reported as primary
     outcomes -- the np=2 runs showed the raw objective tilting systematically
     weaker (cumKL 0.18 vs 0.66), and whether that weakness costs anything is
     the open question.

Usage: python writeup/analyze_ablation.py [root] [H]
"""
import os, re, sys, glob, statistics as st

ARMS = ["none", "objective_raw"]
LABEL = {"none": "heat V (ours)", "objective_raw": "L2 objective"}


def read_seed(d):
    """-> dict of metrics for one run dir, or None."""
    log = os.path.join(d, "seed_success_log.txt")
    run = os.path.join(d, "run.log")
    if not os.path.exists(log):
        return None
    try:
        ok, dist, vis, prop, ex = open(log).read().split()[:5]
    except ValueError:
        return None
    m = {"success": int(float(ok)), "dist": float(dist), "exit": int(ex)}
    if os.path.exists(run):
        txt = open(run, errors="ignore").read()
        last = None
        for last in re.finditer(
            r"done \| beta=([\d.]+) \| resamples=(\d+) \| cumKL=([\d.]+) nats \| final ESS=([\d.]+)", txt):
            pass
        if last:
            m["beta"], m["resamples"] = float(last.group(1)), int(last.group(2))
            m["cumkl"], m["ess"] = float(last.group(3)), float(last.group(4))
    return m


def fmt(vals, prec=2):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    med = st.median(vals)
    return f"{st.mean(vals):.{prec}f} (med {med:.{prec}f})"


def main(root, H):
    data = {a: {} for a in ARMS}
    for a in ARMS:
        for d in glob.glob(f"{root}/{a}_H{H}/seed_*"):
            m = read_seed(d)
            if m:
                data[a][int(d.rsplit("_", 1)[1])] = m
    both = sorted(set(data[ARMS[0]]) & set(data[ARMS[1]]))
    if not both:
        print(f"no paired seeds yet under {root} (H{H})")
        for a in ARMS:
            print(f"  {a}: {len(data[a])} runs")
        return

    print(f"paired seeds: {len(both)}\n")
    print(f"{'metric':<26} {'heat V (ours)':<22} {'L2 objective':<22} {'delta'}")
    print("-" * 82)
    sr = {a: sum(data[a][s]['success'] for s in both) for a in ARMS}
    print(f"{'success / n':<26} {str(sr['none'])+'/'+str(len(both)):<22} "
          f"{str(sr['objective_raw'])+'/'+str(len(both)):<22} {sr['none']-sr['objective_raw']:+d}")

    # Optimality is only comparable where BOTH solved it.
    solved = [s for s in both if data['none'][s]['success'] and data['objective_raw'][s]['success']]
    print(f"\n-- on the {len(solved)} seeds BOTH arms solved --")
    for key, prec in (("exit", 1), ("dist", 3)):
        v0 = [data['none'][s][key] for s in solved]
        v1 = [data['objective_raw'][s][key] for s in solved]
        d = st.mean(v0) - st.mean(v1) if solved else 0
        name = "exit_step (lower=better)" if key == "exit" else "state_dist (lower=better)"
        print(f"{name:<26} {fmt(v0,prec):<22} {fmt(v1,prec):<22} {d:+.{prec}f}")
    if solved:
        w = sum(data['none'][s]['exit'] < data['objective_raw'][s]['exit'] for s in solved)
        l = sum(data['none'][s]['exit'] > data['objective_raw'][s]['exit'] for s in solved)
        print(f"{'  exit_step win/tie/loss':<26} {w}/{len(solved)-w-l}/{l}")

    print("\n-- tilt behaviour (all paired seeds) --")
    for key, prec in (("cumkl", 2), ("resamples", 1), ("ess", 2), ("beta", 3)):
        v0 = [data['none'][s].get(key) for s in both]
        v1 = [data['objective_raw'][s].get(key) for s in both]
        a0 = [x for x in v0 if x is not None]; a1 = [x for x in v1 if x is not None]
        d = (st.mean(a0) - st.mean(a1)) if a0 and a1 else 0
        print(f"{key:<26} {fmt(v0,prec):<22} {fmt(v1,prec):<22} {d:+.{prec}f}")
    inert0 = sum(1 for s in both if (data['none'][s].get('cumkl') or 0) < 0.01)
    inert1 = sum(1 for s in both if (data['objective_raw'][s].get('cumkl') or 0) < 0.01)
    print(f"{'seeds w/ inert tilt (~0)':<26} {inert0:<22} {inert1:<22} {inert0-inert1:+d}")


if __name__ == "__main__":
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(repo, "sweep_fk", "ablate_np3"),
         sys.argv[2] if len(sys.argv) > 2 else "10")
