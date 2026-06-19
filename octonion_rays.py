# octonion_rays.py
# ============================================================================
# More high-frequency ARC mechanisms, gradient-free, un-named, exact-verified.
#
#   (1) RAY DRAWING -- from every coloured cell, cast rays in a learned direction
#       set (horizontal / vertical / orthogonal / diagonal / all-8), painting the
#       cell's colour to the border, optionally stopping at obstacles and optionally
#       only over background.  The direction set + stop/overwrite mode is fit from
#       data (a geometric family, not a per-task named op).
#   (2) CONNECT PAIRS -- for every pair of equal-coloured cells aligned on a row or
#       column, fill the gap between them (with that colour), when the gap is empty.
#   (3) DENOISE -- cells with no orthogonal same-colour neighbour (isolated specks)
#       are removed to background.
# Accepted only on EXACT reproduction of every demonstration, then applied to test.
# ============================================================================
import json, time, sys
import numpy as np
from octonion_arc import A, eq, bg_color

def _bg(g): return 0 if (A(g) == 0).any() else bg_color(A(g))

# ----------------------------------------------------------------- (1) ray drawing
DIRSETS = {
    "H": [(0, 1), (0, -1)], "V": [(-1, 0), (1, 0)],
    "HV": [(0, 1), (0, -1), (-1, 0), (1, 0)],
    "D": [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    "ALL": [(0, 1), (0, -1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)],
}

def _ray(g, dirs, stop_obstacle, bgonly):
    g = A(g); bg = _bg(g); out = g.copy(); H, W = g.shape
    seeds = list(zip(*np.where(g != bg)))
    for r, c in seeds:
        v = g[r, c]
        for dr, dc in dirs:
            rr, cc = r + dr, c + dc
            while 0 <= rr < H and 0 <= cc < W:
                if stop_obstacle and g[rr, cc] != bg: break
                if bgonly and g[rr, cc] != bg: break
                out[rr, cc] = v; rr += dr; cc += dc
    return out

def _rays(pairs, tests):
    if not all(i.shape == o.shape for i, o in pairs): return None
    for name, dirs in DIRSETS.items():
        for stop in (False, True):
            for bgonly in (True, False):
                if all(eq(_ray(i, dirs, stop, bgonly), o) for i, o in pairs):
                    pr = [A(_ray(t, dirs, stop, bgonly)) for t in tests]
                    return pr
    return None

# ----------------------------------------------------------------- (2) connect pairs
def _connect_grid(g):
    g = A(g); bg = _bg(g); out = g.copy(); H, W = g.shape
    for v in np.unique(g):
        if v == bg: continue
        pts = list(zip(*np.where(g == v)))
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                (r1, c1), (r2, c2) = pts[a], pts[b]
                if r1 == r2:
                    lo, hi = sorted((c1, c2))
                    if all(g[r1, x] in (v, bg) for x in range(lo + 1, hi)):
                        out[r1, lo + 1:hi] = np.where(g[r1, lo + 1:hi] == bg, v, out[r1, lo + 1:hi])
                elif c1 == c2:
                    lo, hi = sorted((r1, r2))
                    if all(g[x, c1] in (v, bg) for x in range(lo + 1, hi)):
                        out[lo + 1:hi, c1] = np.where(g[lo + 1:hi, c1] == bg, v, out[lo + 1:hi, c1])
    return out

def _connect(pairs, tests):
    if not all(i.shape == o.shape for i, o in pairs): return None
    if all(eq(_connect_grid(i), o) for i, o in pairs):
        return [A(_connect_grid(t)) for t in tests]
    return None

# ----------------------------------------------------------------- (3) denoise
def _denoise_grid(g):
    g = A(g); bg = _bg(g); out = g.copy(); H, W = g.shape
    for r in range(H):
        for c in range(W):
            v = g[r, c]
            if v == bg: continue
            iso = True
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and g[rr, cc] == v: iso = False; break
            if iso: out[r, c] = bg
    return out

def _denoise(pairs, tests):
    if not all(i.shape == o.shape for i, o in pairs): return None
    if all(eq(_denoise_grid(i), o) for i, o in pairs):
        return [A(_denoise_grid(t)) for t in tests]
    return None

def solve(task):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    tests = [A(tp["input"]) for tp in task["test"]]
    for fn in (_rays, _connect, _denoise):
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
    print("RAYS (ray draw / connect pairs / denoise, gradient-free): %d / %d  %s  (%.0fs)"
          % (len(solved), len(ch), split, time.time() - t0))
    open("/tmp/rays_%s.ids" % split, "w").write(" ".join(solved))
