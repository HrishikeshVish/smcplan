"""Composed per-cell rates at N=25, for the main-text bar figure.

Composition (as specified): keep every seed where raw-L2 FAILED in the
roadmap-succeeds stratum, fill the rest of that stratum with both-succeed seeds,
then add roadmap-failure seeds until the roadmap's success/failure split matches
the main table.

Three strata feed each cell:
  BAND  sweep_aft/rawl2      seeds the roadmap solves in 5-120 iterations
  FAST  sweep_aft/faststratum seeds it solves in under 5 -- excluded by the
        original [5,120] filter, which dropped 21% of all solved seeds and
        ~half of Point maze's, so the success stratum was biased toward hard
        instances and L2's failure rate was measured only on those
  FAIL  sweep_aft/failstratum seeds the roadmap does not solve

Reacher is the one env whose failure stratum is already inside sweep_aft/rawl2:
its goals do not reproduce across runs, so the slow_seeds filter never bound and
the sweep contains roadmap failures of its own. Those are used directly; the
failstratum sweep only tops it up. Every other env has an all-success stratum by
construction and draws its failures entirely from failstratum.
"""
import csv, glob, collections

MAIN = {('pointmaze','H5'):75, ('pointmaze','H10'):81, ('pointmaze','H15'):61,
        ('wall','H5'):56, ('wall','H10'):64, ('wall','H15'):41,
        ('pusht','H5'):68, ('pusht','H10'):27, ('pusht','H15'):5,
        ('reacher','H5'):100, ('reacher','H10'):92, ('reacher','H15'):50,
        ('cube','H5'):88}
N = 25

S, Fown = collections.defaultdict(list), collections.defaultdict(list)
got = collections.defaultdict(dict)
for arm in ('roadmap', 'rawl2'):
    for f in glob.glob(f'sweep_aft/rawl2/{arm}/*/H*/seed_*/result.txt'):
        p = f.split('/'); got[(p[-4], p[-3], p[-2])][arm] = int(open(f).read().split()[0])
for (e, H, s), v in got.items():
    if len(v) != 2:
        continue
    (S if v['roadmap'] == 1 else Fown)[(e, H)].append(v['rawl2'])

Fnew = collections.defaultdict(list)
for f in glob.glob('sweep_aft/failstratum/rawl2/*/H*/seed_*/result.txt'):
    p = f.split('/'); Fnew[(p[-4], p[-3])].append(int(open(f).read().split()[0]))

# FAST: roadmap solves these too, so they join the success stratum.
Sfast = collections.defaultdict(list)
for f in glob.glob('sweep_aft/faststratum/rawl2/*/H*/seed_*/result.txt'):
    p = f.split('/'); Sfast[(p[-4], p[-3])].append(int(open(f).read().split()[0]))

rows = {}
for k in sorted(MAIN, key=lambda x: (x[0], int(x[1][1:]))):
    s = S.get(k, []) + Sfast.get(k, [])
    f = Fown.get(k, []) + Fnew.get(k, [])          # reacher contributes its own
    if not s:
        continue
    rate = MAIN[k]
    want_ok = round(N * rate / 100); want_fail = N - want_ok
    if want_fail and len(f) < want_fail:
        status = f'partial ({len(f)}/{want_fail} fail seeds)'
    else:
        status = 'complete'
    lf = [x for x in s if x == 0]
    bo = [x for x in s if x == 1]
    comp = lf + bo[:max(0, want_ok - len(lf))] + f[:want_fail]
    rows[f'{k[0]}_{k[1]}'] = dict(env=k[0], H=int(k[1][1:]), roadmap=rate,
                                  l2=100 * sum(comp) / len(comp), n_S=len(s),
                                  n_F=len(f), want_fail=want_fail, status=status)

# Written as CSV so the figure reads a fixed table rather than recomposing from
# the sweep directories on every run: the numbers in the paper then come from a
# file that can be diffed and checked, and a rerun of the figure cannot silently
# change them because a sweep directory grew.
with open('writeup/composed.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['cell', 'env', 'H', 'roadmap', 'l2', 'n_success_stratum',
                'n_fail_stratum', 'fail_seeds_needed', 'status'])
    for k, r in rows.items():
        w.writerow([k, r['env'], r['H'], f"{r['roadmap']:.1f}", f"{r['l2']:.4f}",
                    r['n_S'], r['n_F'], r['want_fail'], r['status']])
print(f'{"cell":15s} {"roadmap":>8s} {"L2":>7s} {"gap":>6s}  {"status":>26s}')
for k, r in rows.items():
    print(f'{k:15s} {r["roadmap"]:7d}% {r["l2"]:6.1f}% {r["roadmap"]-r["l2"]:+6.1f}  {r["status"]:>26s}')
