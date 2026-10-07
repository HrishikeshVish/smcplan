"""Ground-truth cost-to-go along executed trajectories, and the progress fraction.

DEFINITION (the quantity the main-text claim is about)
-----------------------------------------------------
For an executed trajectory s_0..s_T and a cost-to-go c defined by the
ENVIRONMENT's geometry (never by either arm's own cost -- using the roadmap's V
would make the comparison circular):

    d_t = c(s_{t+1}) - c(s_t)
    D   = sum_t max(0, -d_t)        total descent
    U   = sum_t max(0, +d_t)        total ascent
    desc = D / (D + U)

desc lies in [0, 1]; it is 1 exactly when the cost-to-go never increases, and
0.5 when the path moves backwards as much as forwards. It is equivalent to a
normalised net progress, since D - U = c_0 - c_T and therefore

    2*desc - 1 = (c_0 - c_T) / (D + U)

i.e. net progress per unit of cost-to-go variation. It is scale-free, so seeds
with different start-goal geometry are directly comparable.

Steps with d_t = 0 enter neither D nor U. That is deliberate and is why desc is
used in place of the fraction of strictly-improving steps: in Push-T the block
does not move while the pusher repositions, and those pauses are not regressions.

COST-TO-GO PER ENVIRONMENT
--------------------------
point maze  true geodesic through the maze, by Dijkstra on an occupancy grid.
            Free space is the wrapper's declared init/goal region dilated by
            0.18 -- the largest clearance any trajectory actually uses. The
            dilation preserves the topology: the central block stays 1.04 wide,
            so the two corridors remain joined only across the top.
wall        true geodesic with exact geometry: the slab x in [29, 35] blocks
            everything except the door y in [26, 34]. Both were read off the
            trajectories, since the config is ambiguous (fix_door_location is 30
            in DEFAULT_CFG and 10 in the wrapper signature); in the slab the dot
            is only ever observed at y in [26, 34].
Push-T      the block is a point object in free 2D space, so its geodesic IS the
            euclidean distance from the block to its goal position. The pusher's
            own position is not scored here.

Cube and Reacher are excluded: their tasks are not navigation in a space where a
configuration distance is an honest cost-to-go.
"""
import glob
import os

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = "/home/hviswan/Documents/grasp"
SWEEP = os.path.join(ROOT, "sweep_aft/rawl2")
ENVS = (("Point maze", "pointmaze"), ("Push-T", "pusht"), ("Wall", "wall"))


def _grid_geodesic(free, g_x, g_y, res, goals):
    """Dijkstra from each goal over an 8-connected occupancy grid."""
    ny, nx = free.shape
    idx = -np.ones(free.shape, np.int64)
    idx[free] = np.arange(free.sum())
    n = int(free.sum())
    rows, cols, vals = [], [], []
    for dxi, dyi in ((1, 0), (0, 1), (1, 1), (1, -1)):
        sa = (slice(max(0, -dxi), ny - max(0, dxi)), slice(max(0, -dyi), nx - max(0, dyi)))
        sb = (slice(max(0, dxi), ny - max(0, -dxi)), slice(max(0, dyi), nx - max(0, -dyi)))
        m = free[sa] & free[sb]
        ia, ib = idx[sa][m], idx[sb][m]
        w = res * np.hypot(dxi, dyi)
        rows += [ia, ib]; cols += [ib, ia]
        vals += [np.full(ia.shape, w)] * 2
    adj = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                     shape=(n, n)).tocsr()
    fi, fj = np.nonzero(free)
    fp = np.stack([g_x[fi], g_y[fj]], 1)

    def snap(p):
        p = np.atleast_2d(p)
        i = np.clip(((p[:, 0] - g_x[0]) / res).round().astype(int), 0, ny - 1)
        j = np.clip(((p[:, 1] - g_y[0]) / res).round().astype(int), 0, nx - 1)
        out = idx[i, j].copy()
        bad = ~free[i, j]
        if bad.any():
            q = p[bad]
            k = ((q[:, None, 0] - fp[None, :, 0]) ** 2 + (q[:, None, 1] - fp[None, :, 1]) ** 2).argmin(1)
            out[bad] = idx[fi[k], fj[k]]
        return out

    return dijkstra(adj, indices=snap(np.asarray(goals)), directed=False), snap


def _pointmaze_solver(goals):
    res, pad = 0.01, 0.18
    gx = np.arange(0.30, 3.31 + 1e-9, res)
    gy = np.arange(0.30, 3.31 + 1e-9, res)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    declared = ((((0.5 <= GX) & (GX <= 1.1)) | ((2.5 <= GX) & (GX <= 3.1)))
                & ((0.5 <= GY) & (GY <= 3.1))) | ((1.1 < GX) & (GX < 2.5) & (2.5 <= GY) & (GY <= 3.1))
    free = distance_transform_edt(~declared) * res <= pad
    return _grid_geodesic(free, gx, gy, res, goals)


def _wall_solver(goals):
    res = 0.25
    g = np.arange(3.0, 61.0 + 1e-9, res)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    free = ~(((GX >= 29) & (GX <= 35)) & ~((GY >= 26) & (GY <= 34)))
    return _grid_geodesic(free, g, g, res, goals)


def desc(c):
    """Progress fraction of one cost-to-go sequence; nan if the cost never moves."""
    d = np.diff(np.asarray(c, float))
    down, up = -d[d < 0].sum(), d[d > 0].sum()
    return float(down / (down + up)) if down + up > 0 else np.nan


def load(env):
    """[{H, seed, stratum, c_roadmap, c_rawl2, desc_roadmap, desc_rawl2}] for one env.

    Pairs whose two arms selected a bit-identical action sequence are dropped:
    the tilt reweighted the population but the argmin picked the same particle,
    so the pair carries no information either way.
    """
    got = {}
    for arm in ("roadmap", "rawl2"):
        for f in glob.glob(f"{SWEEP}/{arm}/{env}/H*/seed_*/result.txt"):
            p = f.split("/")
            got.setdefault((p[-3], p[-2]), {})[arm] = (f.rsplit("/", 1)[0],
                                                       int(open(f).read().split()[0]))
    pairs = {k: v for k, v in got.items() if len(v) == 2}
    keys = sorted(pairs)
    z0 = {k: np.load(pairs[k]["roadmap"][0] + "/traj.npz", allow_pickle=True) for k in keys}

    if env == "pusht":                    # block in free space -> euclidean
        cost = lambda k, es: np.linalg.norm(es[:, 2:4] - np.asarray(z0[k]["state_g"])[0][2:4], axis=1)
    else:
        goals = [np.asarray(z0[k]["state_g"])[0][:2] for k in keys]
        D, snap = (_pointmaze_solver if env == "pointmaze" else _wall_solver)(goals)
        cost = lambda k, es, _D=D, _s=snap: _D[keys.index(k)][_s(es[:, :2])]

    out = []
    for k in keys:
        rec, acts = {}, {}
        for arm in ("roadmap", "rawl2"):
            z = np.load(pairs[k][arm][0] + "/traj.npz", allow_pickle=True)
            es = np.asarray(z["e_states"])[0]
            rec[arm] = (cost(k, es), pairs[k][arm][1])
            acts[arm] = np.asarray(z["actions"])
        if np.array_equal(acts["roadmap"], acts["rawl2"]):
            continue
        ok_r, ok_l = rec["roadmap"][1], rec["rawl2"][1]
        out.append(dict(H=int(k[0][1:]), seed=int(k[1].split("_")[1]),
                        stratum=("both" if ok_r and ok_l else
                                 "l2_fails" if ok_r else
                                 "road_fails" if ok_l else "both_fail"),
                        c_roadmap=rec["roadmap"][0], c_rawl2=rec["rawl2"][0],
                        desc_roadmap=desc(rec["roadmap"][0]), desc_rawl2=desc(rec["rawl2"][0])))
    return out
