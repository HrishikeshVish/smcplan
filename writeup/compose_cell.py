"""Per-cell composition at N=25, as specified: keep every seed where raw-L2
FAILED in the roadmap-succeeds stratum, fill the rest of that stratum with
both-succeed seeds, and add roadmap-failure seeds so the roadmap's success
split matches the main table (writeup/table1_iclr.tex).

Prints the composed rate AND the unmodified stratum rate, because the composed
denominator has had L2's wins removed and the two differ.
"""
import glob, collections, sys

MAIN = {('pointmaze','H5'):75, ('pointmaze','H10'):81, ('pointmaze','H15'):61,
        ('wall','H5'):56, ('wall','H10'):64, ('wall','H15'):41,
        ('pusht','H5'):68, ('pusht','H10'):27, ('pusht','H15'):5,
        ('reacher','H5'):100, ('reacher','H10'):92, ('reacher','H15'):50,
        ('cube','H5'):88}
N = 25

def stratum_S():
    got = collections.defaultdict(dict)
    for arm in ('roadmap','rawl2'):
        for f in glob.glob(f'sweep_aft/rawl2/{arm}/*/H*/seed_*/result.txt'):
            p = f.split('/'); got[(p[-4],p[-3],p[-2])][arm] = int(open(f).read().split()[0])
    out = collections.defaultdict(list)
    for (e,H,s), v in got.items():
        if len(v) == 2 and v['roadmap'] == 1:
            out[(e,H)].append(v['rawl2'])
    return out

def stratum_F():
    out = collections.defaultdict(list)
    for f in glob.glob('sweep_aft/failstratum/rawl2/*/H*/seed_*/result.txt'):
        p = f.split('/'); out[(p[-4],p[-3])].append(int(open(f).read().split()[0]))
    return out

S, F = stratum_S(), stratum_F()
print(f'{"cell":15s} {"main%":>6s} {"F: L2 ok":>9s} {"S: L2 fails":>12s} '
      f'{"composed L2%":>13s} {"true strat L2%":>15s}')
for k in sorted(set(S) | set(F), key=lambda x: (x[0], int(x[1][1:]))):
    s, f = S.get(k, []), F.get(k, [])
    if not f:
        continue
    rate = MAIN.get(k)
    want_ok = round(N * rate / 100); want_fail = N - want_ok
    lf = [x for x in s if x == 0]                 # L2 failed: always kept
    bo = [x for x in s if x == 1]                 # both succeeded: droppable
    keep_bo = max(0, want_ok - len(lf))
    comp = lf + bo[:keep_bo] + f[:want_fail]
    print(f'{k[0]+" "+k[1]:15s} {rate:5d}% {sum(f)}/{len(f):<7d} '
          f'{len(lf)}/{len(s):<10d} {100*sum(comp)/len(comp):12.1f}% '
          f'{100*sum(s)/len(s):14.1f}%')
