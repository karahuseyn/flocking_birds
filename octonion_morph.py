# octonion_morph.py
# ============================================================================
# Morphology + colour-frequency + counting mechanisms, gradient-free, un-named,
# exact-verified.
#
#   (1) DILATE   -- spread every non-background cell's colour to its neighbours
#       (4/8-connectivity, 1-2 steps); a geometric family.
#   (2) OUTLINE  -- paint a halo of a learned colour on the background cells adjacent
#       to objects.
#   (3) KEEP / REMOVE by colour frequency -- keep only (or delete) the most/least
#       frequent non-background colour.
#   (4) COUNT -> LINE -- emit a 1xN / Nx1 strip of a learned colour whose length N is
#       the number of objects (or of cells of the dominant colour).
# Accepted only on EXACT reproduction of every demonstration, then applied to test.
# ============================================================================
import json, time, sys
from collections import Counter
import numpy as np
from octonion_arc import A, eq, bg_color, objects

def _bg(g): return bg_color(A(g))

# ----------------------------------------------------------------- (1) dilate
def _dilate(g, conn8, steps):
    g = A(g); bg = _bg(g); H, W = g.shape; out = g.copy()
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)] + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if conn8 else [])
    for _ in range(steps):
        cur = out.copy()
        for dr, dc in dirs:
            r0, r1 = max(0, dr), min(H, H + dr); c0, c1 = max(0, dc), min(W, W + dc)
            src = cur[r0:r1, c0:c1]; dst = out[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
            mask = (dst == bg) & (src != bg)
            dst[mask] = src[mask]
    return out

def _dilate_m(pairs, tests):
    for conn8 in (False, True):
        for steps in (1, 2):
            if all(i.shape == o.shape and eq(_dilate(i, conn8, steps), o) for i, o in pairs):
                return [A(_dilate(t, conn8, steps)) for t in tests]
    return None

# ----------------------------------------------------------------- (2) outline / halo
def _outline(g, col, conn8):
    g = A(g); bg = _bg(g); H, W = g.shape; out = g.copy()
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)] + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if conn8 else [])
    halo = np.zeros((H, W), bool)
    for dr, dc in dirs:
        r0, r1 = max(0, dr), min(H, H + dr); c0, c1 = max(0, dc), min(W, W + dc)
        src = g[r0:r1, c0:c1]
        halo[r0 - dr:r1 - dr, c0 - dc:c1 - dc] |= (src != bg)
    paint = halo & (g == bg); out[paint] = col
    return out

def _outline_m(pairs, tests):
    for conn8 in (False, True):
        col = None; ok = True
        for i, o in pairs:
            if i.shape != o.shape: ok = False; break
            changed = A(o)[(A(i) == _bg(i)) & (A(o) != _bg(i))]
            if len(changed) == 0: ok = False; break
            u = set(changed.tolist())
            if len(u) != 1: ok = False; break
            v = u.pop()
            if col is None: col = v
            elif col != v: ok = False; break
        if ok and col is not None and all(eq(_outline(i, col, conn8), o) for i, o in pairs):
            return [A(_outline(t, col, conn8)) for t in tests]
    return None

# ----------------------------------------------------------------- (3) keep / remove by frequency
def _freq_target(g, which):
    g = A(g); bg = _bg(g); nz = g[g != bg]
    if len(nz) == 0: return None
    cnt = Counter(nz.tolist())
    items = cnt.most_common()
    return items[0][0] if which == "most" else items[-1][0]

def _freqop(pairs, tests):
    for which in ("most", "least"):
        for op in ("keep", "remove"):
            def emit(g, which=which, op=op):
                g = A(g); bg = _bg(g); t = _freq_target(g, which)
                if t is None: return None
                return np.where(g == t, g, bg) if op == "keep" else np.where(g == t, bg, g)
            if all(i.shape == o.shape and (emit(i) is not None) and eq(emit(i), o) for i, o in pairs):
                pr = [emit(t) for t in tests]
                if all(p is not None for p in pr): return [A(p) for p in pr]
    return None

# ----------------------------------------------------------------- (4) count -> line
def _count(g, mode):
    g = A(g); bg = _bg(g)
    if mode == "objects": return len(objects(g, bg, True, True))
    if mode == "objects_any": return len(objects(g, bg, True, False))
    if mode == "colors": return len(set(g[g != bg].tolist()))
    return 0

def _countline(pairs, tests):
    for mode in ("objects", "objects_any", "colors"):
        for orient in ("row", "col"):
            col = None; ok = True
            for i, o in pairs:
                n = _count(i, mode)
                exp = (1, n) if orient == "row" else (n, 1)
                if o.shape != exp or n == 0: ok = False; break
                u = set(A(o).ravel().tolist())
                if len(u) != 1: ok = False; break
                v = u.pop()
                if col is None: col = v
                elif col != v: ok = False; break
            if ok and col is not None:
                def emit(g, mode=mode, orient=orient, col=col):
                    n = _count(g, mode)
                    if n == 0: return None
                    return np.full((1, n) if orient == "row" else (n, 1), col, int)
                pr = [emit(t) for t in tests]
                if all(p is not None for p in pr): return [A(p) for p in pr]
    return None

def solve(task):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    tests = [A(tp["input"]) for tp in task["test"]]
    for fn in (_dilate_m, _outline_m, _freqop, _countline):
        try: pr = fn(pairs, tests)
        except Exception: pr = None
        if pr is not None and all(p is not None for p in pr): return pr
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
    print("MORPH (dilate / outline / freq keep-remove / count-line, gradient-free): %d / %d  %s  (%.0fs)"
          % (len(solved), len(ch), split, time.time() - t0))
    open("/tmp/morph_%s.ids" % split, "w").write(" ".join(solved))
