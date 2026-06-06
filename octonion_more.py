# octonion_more.py
# ============================================================================
# More high-frequency ARC mechanisms, gradient-free, un-named, exact-verified --
# growing the "model" the only way that has ever paid off: new MECHANISMS.
#
#   (1) KALEIDOSCOPE TILING -- output is an a×b grid of input-sized tiles, each a
#       (position-specific) element of the dihedral group of the square applied to
#       the input.  The per-tile transform is learned from the data (so mirror-tiling,
#       rotation-tiling, kaleidoscopes, etc. all fall out); verified exactly.
#   (2) SYMMETRY COMPLETION (overlay) -- complete the grid to be invariant under a
#       detected symmetry subgroup by overlaying the orbit: a background cell takes
#       the (consistent) non-background value of its orbit partners.
# Accepted only on EXACT reproduction of every demonstration, then applied to test.
# ============================================================================
import json, time, sys
import numpy as np
from octonion_arc import A, eq, bg_color, DIHEDRAL

DIH = list(DIHEDRAL.values())

# ----------------------------------------------------------------- (1) kaleidoscope tiling
def _kaleidoscope(pairs, tests):
    H0, W0 = pairs[0][0].shape; Ho, Wo = pairs[0][1].shape
    if Ho % H0 or Wo % W0: return None
    a, b = Ho // H0, Wo // W0
    if a * b < 2: return None
    for i, o in pairs:                                   # shape ratios must be constant
        if o.shape != (a * i.shape[0], b * i.shape[1]): return None
    grid = {}                                            # (ti,tj) -> dihedral index
    for ti in range(a):
        for tj in range(b):
            chosen = None
            for di, d in enumerate(DIH):
                ok = True
                for i, o in pairs:
                    h, w = i.shape; tile = o[ti * h:(ti + 1) * h, tj * w:(tj + 1) * w]
                    di_g = d(A(i))
                    if di_g.shape != tile.shape or not np.array_equal(di_g, tile): ok = False; break
                if ok: chosen = di; break
            if chosen is None: return None
            grid[(ti, tj)] = chosen
    def emit(g):
        g = A(g); h, w = g.shape; out = np.zeros((a * h, b * w), int)
        for (ti, tj), di in grid.items():
            t = DIH[di](g)
            if t.shape != (h, w): return None
            out[ti * h:(ti + 1) * h, tj * w:(tj + 1) * w] = t
        return out
    if all(eq(emit(i), o) for i, o in pairs):
        pr = [emit(t) for t in tests]
        if all(p is not None for p in pr): return pr
    return None

# ----------------------------------------------------------------- (2) symmetry completion
def _sym_maps(H, W):
    rr, cc = np.mgrid[0:H, 0:W]
    m = {"mh": (rr, W - 1 - cc), "mv": (H - 1 - rr, cc), "r180": (H - 1 - rr, W - 1 - cc)}
    if H == W: m["T"] = (cc, rr); m["aT"] = (W - 1 - cc, H - 1 - rr)
    return m

# subgroups as sets of map-names (closed under composition for these choices)
_GROUPS = [["mh"], ["mv"], ["r180"], ["mh", "mv", "r180"], ["T"], ["aT"],
           ["mh", "mv", "r180", "T", "aT"]]

def _complete(g, names, maps, bg):
    g = A(g).copy()
    for _ in range(4):
        changed = False
        for nm in names:
            if nm not in maps: continue
            sr, sc = maps[nm]; src = g[sr, sc]
            fillable = (g == bg) & (src != bg)
            if fillable.any(): g[fillable] = src[fillable]; changed = True
        if not changed: break
    return g

def _symcomplete(pairs, tests):
    if not all(i.shape == o.shape for i, o in pairs): return None
    def _bg(g): return 0 if (A(g) == 0).any() else bg_color(A(g))
    for names in _GROUPS:
        ok = True
        for i, o in pairs:
            H, W = i.shape; maps = _sym_maps(H, W)
            if any(nm not in maps for nm in names): ok = False; break
            if not eq(_complete(i, names, maps, _bg(i)), o): ok = False; break
        if not ok: continue
        def emit(g, names=names):
            g = A(g); H, W = g.shape; maps = _sym_maps(H, W)
            if any(nm not in maps for nm in names): return None
            return _complete(g, names, maps, _bg(g))
        pr = [emit(t) for t in tests]
        if all(p is not None for p in pr): return pr
    return None

def solve(task):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    tests = [A(tp["input"]) for tp in task["test"]]
    for fn in (_kaleidoscope, _symcomplete):
        try: pr = fn(pairs, tests)
        except Exception: pr = None
        if pr is not None: return pr
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
    print("MORE (kaleidoscope tiling / symmetry completion, gradient-free): %d / %d  %s  (%.0fs)"
          % (len(solved), len(ch), split, time.time() - t0))
    open("/tmp/more_%s.ids" % split, "w").write(" ".join(solved))
