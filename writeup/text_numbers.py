"""Every number the main text and the appendix prose quote about the planner
comparison (tab:planners with MPPI), the cost comparison, Cube and the budget,
printed from writeup/equal_particle_stats.py so the prose has one source.
2026-10-11. `ours` follows OURS_PM (default: score M-step on Point maze).

Usage: python writeup/text_numbers.py
"""
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

from equal_particle_stats import (ENVS, HS, OURS_PM, cube_nontrivial, outcomes,
                                  paired)

BASE = [("GD", "gds12"), ("GRASP", "grasp12"), ("CEM", "cem12"), ("MPPI", "mppi12")]
CELLS = [(n, e, H) for n, e in ENVS for H in HS]


def r(x, d=0):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run, m = [None] * len(ps), 0.0, len(ps)
    for k, i in enumerate(idx):
        run = max(run, min(1.0, (m - k) * ps[i]))
        out[i] = run
    return out


def succ(o):
    return 100.0 * sum(ok for ok, _ in o.values()) / len(o)


def iters(o):
    v = [i for ok, i in o.values() if ok and i is not None]
    return float(np.mean(v)) if v else None


print("OURS_PM =", OURS_PM)
O = {(e, H): {a: outcomes(a, e, H) for a in ["ours", "ours_l2"] + [b for _, b in BASE]}
     for _, e, H in CELLS}

# ---------------------------------------------------------------- Table 1
print("\n== Table 1: ours vs strongest baseline per cell")
best_rows, ps = [], []
strict = tie = lower = 0
for n, e, H in CELLS:
    d = O[(e, H)]
    s = {a: succ(d[a]) for a in d}
    bs = max(BASE, key=lambda nb: s[nb[1]])        # first max, as the table generator
    top = [nm for nm, b in BASE if s[b] == s[bs[1]]]
    w, l, p = paired(d["ours"], d[bs[1]])
    ps.append(p)
    rel = "best" if s["ours"] > s[bs[1]] else "tie" if s["ours"] == s[bs[1]] else "lower"
    strict += rel == "best"; tie += rel == "tie"; lower += rel == "lower"
    best_rows.append((n, H, s["ours"], top, s[bs[1]], w, l, p, rel))
for (n, H, so, top, sb, w, l, p, rel), q in zip(best_rows, holm(ps)):
    print(f"  {n:10s} H{H:2d} ours {so:3.0f} best baseline {'/'.join(top)} {sb:3.0f}  "
          f"{rel:5s} ours-only {w:2d} base-only {l:2d} p={p:.3g} holm12={q:.3g}")
print(f"  ours strictly highest {strict}, tied {tie}, lower {lower}")
sig_ahead = [x for x in best_rows if x[7] < 0.05 and x[2] > x[4]]
sig_behind = [x for x in best_rows if x[7] < 0.05 and x[2] < x[4]]
print(f"  significant (p<0.05) ahead: {len(sig_ahead)}  behind: {len(sig_behind)}")
print(f"  Holm-significant ahead: {sum(1 for x, q in zip(best_rows, holm(ps)) if q < 0.05 and x[2] > x[4])}")
for nm, b in BASE:
    who = [f"{n} H{H}" for n, e, H in CELLS
           if succ(O[(e, H)][b]) == max(succ(O[(e, H)][bb]) for _, bb in BASE)]
    print(f"  {nm} is (one of) the strongest baseline in: {', '.join(who)}")
print("  cells where a baseline is strictly higher than ours:",
      [(n, H, nm, f"{succ(O[(e, H)][b]):.0f} vs {succ(O[(e, H)]['ours']):.0f}")
       for n, e, H in CELLS for nm, b in BASE
       if succ(O[(e, H)][b]) > succ(O[(e, H)]["ours"])])

# ---------------------------------------------------------------- per-env text
print("\n== per environment (ours | each baseline)")
for n, e, H in CELLS:
    d = O[(e, H)]
    print(f"  {n:10s} H{H:2d} ours {succ(d['ours']):3.0f} " +
          " ".join(f"{nm} {succ(d[b]):3.0f}" for nm, b in BASE))

# ---------------------------------------------------------------- averages
avg = {a: np.mean([succ(O[(e, H)][a]) for _, e, H in CELLS])
       for a in ["ours"] + [b for _, b in BASE]}
print("\n== averages over 12 cells:", {a: round(v, 3) for a, v in avg.items()})
for nm, b in BASE:
    print(f"  gain over {nm}: {avg['ours'] - avg[b]:.3f} -> {r(avg['ours'] - avg[b])}")

# ---------------------------------------------------------------- iterations
print("\n== iterations to success (successful seeds)")
for nm, b in BASE:
    fewer = [(n, H) for n, e, H in CELLS
             if iters(O[(e, H)][b]) is not None and iters(O[(e, H)]["ours"]) < iters(O[(e, H)][b])]
    have = [(n, H) for n, e, H in CELLS if iters(O[(e, H)][b]) is not None]
    rng = [iters(O[(e, H)][b]) for _, e, H in CELLS if iters(O[(e, H)][b]) is not None]
    print(f"  ours fewer than {nm} in {len(fewer)} of {len(have)} cells where {nm} succeeds; "
          f"{nm} range {min(rng):.1f}-{max(rng):.1f}; not fewer: "
          f"{[c for c in have if c not in fewer]}")
print("  ours iterations:", {f"{n} H{H}": round(iters(O[(e, H)]['ours']), 1) for n, e, H in CELLS})
print("  MPPI iterations:", {f"{n} H{H}": round(iters(O[(e, H)]['mppi12']), 1) for n, e, H in CELLS})

# ---------------------------------------------------------------- curves
print("\n== first iteration at which ours' cumulative success exceeds the strongest"
      " baseline's 150-iteration success")
firsts = {}
for n, e, H in CELLS:
    d = O[(e, H)]
    target = max(succ(d[b]) for _, b in BASE)
    its = sorted(i for ok, i in d["ours"].values() if ok)
    k_hit = None
    for k in range(1, 151):
        if 100.0 * sum(1 for i in its if i <= k) / len(d["ours"]) > target:
            k_hit = k
            break
    firsts[(n, H)] = k_hit
    print(f"  {n:10s} H{H:2d} strongest {target:3.0f} -> {k_hit}")
w20 = [c for c, k in firsts.items() if k is not None and k <= 20]
anyk = [k for k in firsts.values() if k is not None]
print(f"  within 20 iterations: {len(w20)} cells; exceeds at all: {len(anyk)} cells, "
      f"latest {max(anyk)}; never: {[c for c, k in firsts.items() if k is None]}")

# ---------------------------------------------------------------- paired totals
print("\n== paired totals over the grid (ours-only, baseline-only)")
for nm, b in BASE:
    W = L = 0
    byH = {H: [0, 0] for H in HS}
    more = []
    for n, e, H in CELLS:
        w, l, _ = paired(O[(e, H)]["ours"], O[(e, H)][b])
        W += w; L += l; byH[H][0] += w; byH[H][1] += l
        if l > w:
            more.append((n, H, w, l))
    print(f"  vs {nm}: {W} / {L}; by H {byH}; cells with more baseline-only: {more}")

# ---------------------------------------------------------------- Cube
print("\n== Cube (50 seeds)")
nt = cube_nontrivial()
oc = outcomes("ours", "cube")
for nm, a in [("Ours", "ours")] + BASE:
    o = outcomes(a, "cube")
    k = sum(ok for ok, _ in o.values())
    knt = sum(1 for s, (ok, _) in o.items() if ok and s in nt)
    line = (f"  {nm:5s} {k}/50 = {100 * k / 50:.0f}%  non-trivial {knt}/{len(nt)} = "
            f"{100 * knt / len(nt):.0f}%  iters {iters(o):.1f}")
    if a != "ours":
        w, l, p = paired(oc, o)
        line += f"  ours-only {w} base-only {l} p={p:.3g}"
    print(line)

# ---------------------------------------------------------------- cost
print("\n== cost: ours (PDE) vs ours_l2 (L2), same M-step per environment")
rows, cps = [], []
for n, e, H in CELLS:
    a, b = O[(e, H)]["ours"], O[(e, H)]["ours_l2"]
    if len(b) < 100:
        rows.append((n, e, H, succ(a), None, None, None, None)); continue
    w, l, p = paired(a, b)
    rows.append((n, e, H, succ(a), succ(b), w, l, p)); cps.append(p)
hq = iter(holm(cps))
for n, e, H, sa, sb, w, l, p in rows:
    if sb is None:
        print(f"  {n:10s} H{H:2d} PDE {sa:3.0f}  L2 pending"); continue
    print(f"  {n:10s} H{H:2d} PDE {sa:3.0f} L2 {sb:3.0f} diff {sa - sb:+3.0f}  "
          f"PDE-only {w:2d} L2-only {l:2d} p={p:.3g} holm{len(cps)}={next(hq):.3g}")
for H in HS:
    ds = [sa - sb for n, e, h, sa, sb, *_ in rows if h == H and sb is not None]
    print(f"  average PDE - L2 at H={H}: {np.mean(ds):+.3f} over {len(ds)} envs")
a, b = outcomes("ours", "cube"), outcomes("ours_l2", "cube")
w, l, p = paired(a, b)
print(f"  Cube PDE {sum(k for k, _ in a.values())}/50 L2 {sum(k for k, _ in b.values())}/{len(b)} "
      f"{w}/{l} p={p:.3g}")

# ---------------------------------------------------------------- L2 vs baselines
print("\n== ours with the L2 cost vs the strongest baseline (incl. MPPI)")
ahead = []
for n, e, H in CELLS:
    d = O[(e, H)]
    if len(d["ours_l2"]) < 100:
        print(f"  {n:10s} H{H:2d} pending"); continue
    bs = max(BASE, key=lambda nb: succ(d[nb[1]]))
    s2, sb = succ(d["ours_l2"]), succ(d[bs[1]])
    w, l, p = paired(d["ours_l2"], d[bs[1]])
    rel = "ahead" if s2 > sb else "tie" if s2 == sb else "behind"
    ahead.append(rel)
    print(f"  {n:10s} H{H:2d} L2 {s2:3.0f} vs {bs[0]} {sb:3.0f} {rel}  {w}/{l} p={p:.3g}")
print("  counts:", {k: ahead.count(k) for k in set(ahead)}, "of", len(ahead))

# ---------------------------------------------------------------- budget
print("\n== budget: success within K=50 vs K=150")
ratios = []
for n, e in ENVS + [("Cube", "cube")]:
    for H in (HS if e != "cube" else (5,)):
        o = outcomes("ours", e, H)
        s150 = 100.0 * sum(ok for ok, _ in o.values()) / len(o)
        s50 = 100.0 * sum(1 for ok, i in o.values() if ok and i is not None and i <= 50) / len(o)
        ratios.append((n, H, s50, s150, s50 >= 0.8 * s150))
for x in ratios:
    print(f"  {x[0]:10s} H{x[1]:2d} K50 {x[2]:3.0f} K150 {x[3]:3.0f} >=80%: {x[4]}")
print(f"  >=80% in {sum(x[4] for x in ratios)} of {len(ratios)}")
