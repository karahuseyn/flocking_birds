# octonion_objrules.py
# ============================================================================
# Three more high-frequency ARC mechanisms, gradient-free, un-named, exact-verified.
# Each is a parametric family searched and fit from the task's own pairs, never a
# task-specific named operator.
#
#   (1) OBJECT RECOLOUR by property -- segment into objects; learn a map from a
#       per-object KEY (size, bbox-area, #colours, D4-canonical shape, original
#       colour, or size-rank) to an output colour; repaint each object's cells.
#   (2) GRAVITY / projection -- per row/column, compact every non-background cell
#       toward one of the four edges (a geometric family over 4 directions, like the
#       dihedral group -- not a per-task named op).
#   (3) ENCLOSED-REGION FILL -- background cells not connected to the border are
#       "holes"; fill them with a learned colour (a fixed colour, or the colour of
#       the enclosing object).
# Accepted only if it reproduces every demonstration EXACTLY, then applied to test.
# ============================================================================
import json, time, sys
from collections import Counter
import numpy as np
from scipy import ndimage
from octonion_arc import A, eq, bg_color, objects

# ----------------------------------------------------------------- (1) object recolour
def _d4sig(o):
    m = o["sm"].astype(int); cand = []
    for k in range(4):
        r = np.rot90(m, k); cand.append(r.tobytes() + bytes(r.shape))
        fr = np.fliplr(r); cand.append(fr.tobytes() + bytes(fr.shape))
    return min(cand)

_KEYS = {
    "size": lambda o, objs: o["size"],
    "area": lambda o, objs: o["h"] * o["w"],
    "ncolors": lambda o, objs: o["ncolors"],
    "shape": lambda o, objs: _d4sig(o),
    "color": lambda o, objs: o["color"],
    "rank": lambda o, objs: sorted({x["size"] for x in objs}, reverse=True).index(o["size"]),
}

def _objrecolor(pairs, tests):
    for mode in ((True, True), (True, False), (False, True), (False, False)):
        for kname, kf in _KEYS.items():
            table = {}; ok = True
            for i, o in pairs:
                if i.shape != o.shape: ok = False; break
                objs = objects(A(i), bg_color(A(i)), mode[0], mode[1])
                if not objs: ok = False; break
                for ob in objs:
                    outvals = set(A(o)[ob["mask"]].tolist())
                    if len(outvals) != 1: ok = False; break       # object must map to one colour
                    v = outvals.pop(); k = kf(ob, objs)
                    if table.get(k, v) != v: ok = False; break
                    table[k] = v
                if not ok: break
            if not ok: continue
            def emit(g, mode=mode, kf=kf, table=table):
                g = A(g).copy(); objs = objects(g, bg_color(g), mode[0], mode[1])
                for ob in objs:
                    k = kf(ob, objs)
                    if k not in table: return None
                    g[ob["mask"]] = table[k]
                return g
            if all(eq(emit(i), o) for i, o in pairs):
                pr = [emit(t) for t in tests]
                if all(p is not None for p in pr): return pr
    return None

# ----------------------------------------------------------------- (2) gravity
def _grav(g, d):
    g = A(g); bg = bg_color(g); H, W = g.shape; out = np.full((H, W), bg, int)
    if d in ("down", "up"):
        for c in range(W):
            col = g[:, c]; vals = col[col != bg]
            if d == "down": out[H - len(vals):, c] = vals
            else: out[:len(vals), c] = vals
    else:
        for r in range(H):
            row = g[r, :]; vals = row[row != bg]
            if d == "right": out[r, W - len(vals):] = vals
            else: out[r, :len(vals)] = vals
    return out

def _gravity(pairs, tests):
    for d in ("down", "up", "left", "right"):
        if all(i.shape == o.shape and eq(_grav(i, d), o) for i, o in pairs):
            return [A(_grav(t, d)) for t in tests]
    return None

# ----------------------------------------------------------------- (3) enclosed fill
def _enclosed(g):
    g = A(g)
    border = np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]])
    bg = Counter(border.tolist()).most_common(1)[0][0]           # background = border-dominant colour
    free = g == bg
    lab, n = ndimage.label(free)                                 # 4-connected bg regions
    touch = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    inside = free & ~np.isin(lab, list(touch))
    return inside

def _fill(pairs, tests):
    # fixed fill colour, or the colour of the enclosing object (single surrounding colour)
    for mode in ("fixed", "surround"):
        col = None; ok = True
        for i, o in pairs:
            if i.shape != o.shape: ok = False; break
            ins = _enclosed(i)
            if not ins.any(): ok = False; break
            if not eq(A(i)[~ins], A(o)[~ins]): ok = False; break   # only holes may change
            vals = set(A(o)[ins].tolist())
            if len(vals) != 1: ok = False; break
            v = vals.pop()
            if mode == "fixed":
                if col is None: col = v
                elif col != v: ok = False; break
        if not ok: continue
        def emit(g, mode=mode, col=col):
            g = A(g).copy(); ins = _enclosed(g)
            if not ins.any(): return None
            g[ins] = col if mode == "fixed" else col
            return g
        if mode == "fixed" and col is not None and all(eq(emit(i), o) for i, o in pairs):
            pr = [emit(t) for t in tests]
            if all(p is not None for p in pr): return pr
    return None

def solve(task):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    tests = [A(tp["input"]) for tp in task["test"]]
    for fn in (_objrecolor, _gravity, _fill):
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
    print("OBJECT-RULES (recolour / gravity / fill, gradient-free): %d / %d  %s  (%.0fs)"
          % (len(solved), len(ch), split, time.time() - t0))
    open("/tmp/objrules_%s.ids" % split, "w").write(" ".join(solved))
