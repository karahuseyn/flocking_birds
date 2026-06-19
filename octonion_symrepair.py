# octonion_symrepair.py
# ============================================================================
# Symmetry / periodicity OCCLUSION REPAIR -- one of ARC's most frequent archetypes
# in BOTH splits, and (because symmetry is universal) the mechanism with the best
# chance of transferring to evaluation.  Gradient-free, no predefined named ops.
#
# Archetype: the grid is a symmetric or periodic pattern with a rectangular region
# hidden behind a single occluder colour.  The visible cells determine the symmetry
# group; we reconstruct the hidden cells from it.  Output is either the whole
# repaired grid, or just the content under the hole (cropped).
#
# Mechanism (all learned / detected from the task's own data, exact-verified):
#   * occluder colour    : searched over the colours present (the hole = its cells).
#   * symmetry group     : detected per grid as the subset of a parametric candidate
#       family { horizontal mirror, vertical mirror, 180 rotation, transpose,
#       anti-transpose, all horizontal periods, all vertical periods } that the
#       VISIBLE cells satisfy exactly.  Nothing is named per task -- we search the
#       family and keep whichever symmetries the data actually obeys.
#   * fill               : propagate known values along the detected symmetries to a
#       fixed point (orbit closure).
#   * output mode        : whole repaired grid, or the hole's bounding-box crop.
# Accepted only if it reproduces every demonstration EXACTLY, then applied to test.
# ============================================================================
import json, time, sys
import numpy as np
from octonion_arc import A, eq

def _coord_syms(H, W):
    """Reflection/rotation source-coordinate maps (src for each cell), as (H,W) index
    pairs; only geometric ones here -- periods handled separately."""
    rr, cc = np.mgrid[0:H, 0:W]
    syms = {"mh": (rr, W - 1 - cc), "mv": (H - 1 - rr, cc), "r180": (H - 1 - rr, W - 1 - cc)}
    if H == W:
        syms["T"] = (cc, rr); syms["aT"] = (W - 1 - cc, H - 1 - rr)
    return syms

def _valid_geom(g, known, src, thr=4):
    sr, sc = src; vsrc = known[sr, sc] & known
    if vsrc.sum() < thr: return False
    return bool(np.all(g[sr, sc][vsrc] == g[vsrc]))

def _valid_period(g, known, axis, p, thr=4):
    if axis == 0:
        if p >= g.shape[0]: return False
        a = known[:-p, :] & known[p:, :]
        if a.sum() < thr: return False
        return bool(np.all(g[:-p, :][a] == g[p:, :][a]))
    else:
        if p >= g.shape[1]: return False
        a = known[:, :-p] & known[:, p:]
        if a.sum() < thr: return False
        return bool(np.all(g[:, :-p][a] == g[:, p:][a]))

def _detect(g, known):
    """Return source-coordinate maps for every symmetry the visible cells satisfy."""
    H, W = g.shape; maps = []
    for name, src in _coord_syms(H, W).items():
        if _valid_geom(g, known, src): maps.append(src)
    for p in range(1, W):                             # horizontal periods (both directions)
        if _valid_period(g, known, 1, p):
            rr, cc = np.mgrid[0:H, 0:W]
            maps.append((rr, np.clip(cc - p, 0, W - 1))); maps.append((rr, np.clip(cc + p, 0, W - 1)))
    for p in range(1, H):                             # vertical periods
        if _valid_period(g, known, 0, p):
            rr, cc = np.mgrid[0:H, 0:W]
            maps.append((np.clip(rr - p, 0, H - 1), cc)); maps.append((np.clip(rr + p, 0, H - 1), cc))
    return maps

def _repair(g, occ):
    """Fill cells of colour `occ` from the detected symmetries; return (filled, ok)."""
    g = A(g).copy(); known = g != occ
    if known.all() or not known.any(): return g, False
    maps = _detect(g, known)
    if not maps: return g, False
    for _ in range(12):
        changed = False
        for sr, sc in maps:
            src_known = known[sr, sc]
            fillable = (~known) & src_known
            if fillable.any():
                g[fillable] = g[sr, sc][fillable]; known[fillable] = True; changed = True
        if not changed: break
    return g, bool(known.all())

def _bbox(mask):
    r = np.where(mask.any(1))[0]; c = np.where(mask.any(0))[0]
    return r.min(), r.max(), c.min(), c.max()

def solve(task):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    tests = [A(tp["input"]) for tp in task["test"]]
    cols = set()
    for i, _ in pairs: cols |= set(np.unique(i).tolist())
    for occ in sorted(cols):
        for mode in ("full", "crop"):
            ok = True
            for i, o in pairs:
                if not (A(i) == occ).any(): ok = False; break
                filled, done = _repair(i, occ)
                if not done: ok = False; break
                if mode == "full":
                    if not eq(filled, o): ok = False; break
                else:
                    r0, r1, c0, c1 = _bbox(A(i) == occ)
                    if not eq(filled[r0:r1 + 1, c0:c1 + 1], o): ok = False; break
            if not ok: continue
            def emit(g, occ=occ, mode=mode):
                filled, done = _repair(g, occ)
                if not done: return None
                if mode == "full": return A(filled)
                r0, r1, c0, c1 = _bbox(A(g) == occ); return A(filled[r0:r1 + 1, c0:c1 + 1])
            pr = [emit(t) for t in tests]
            if all(p is not None for p in pr): return pr
    return None


if __name__ == "__main__":
    DIR = "arc_data/"; split = sys.argv[1] if len(sys.argv) > 1 else "training"
    ch = json.load(open(DIR + "arc-agi_%s_challenges.json" % split))
    sol = json.load(open(DIR + "arc-agi_%s_solutions.json" % split))
    t0 = time.time(); solved = []
    for tid, task in ch.items():
        try: pr = solve(task)
        except Exception: pr = None
        if pr and all(eq(pr[i], A(g)) for i, g in enumerate(sol[tid])): solved.append(tid)
    print("SYMMETRY / OCCLUSION REPAIR (gradient-free): %d / %d  %s  (%.0fs)"
          % (len(solved), len(ch), split, time.time() - t0))
    open("/tmp/symrepair_%s.ids" % split, "w").write(" ".join(solved))
