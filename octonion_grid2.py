# octonion_grid2.py
# ============================================================================
# Three more ARC mechanism families, gradient-free, un-named, exact-verified.
#
#   (1) BLOCK REDUCE -- the input is a regular grid of blocks (by an integer size
#       ratio, or split by full-line separators of one colour); the output is the
#       coarse grid in which each block is summarised to a single cell (its unique
#       non-background colour, or its dominant colour).  "region -> cell".
#   (2) FRAME -- add a border of a learned colour/thickness, or remove a border.
#   (3) BBOX FILL -- solidify each object by filling its bounding box with its
#       colour (or draw only the bounding-box outline).
# Accepted only on EXACT reproduction of every demonstration, then applied to test.
# ============================================================================
import json, time, sys
from collections import Counter
import numpy as np
from octonion_arc import A, eq, bg_color, objects

# ----------------------------------------------------------------- (1) block reduce
def _summ(block, bg, kind):
    vals = block.ravel()
    nz = vals[vals != bg]
    if kind == "uniq_nonbg":
        u = set(nz.tolist())
        if len(u) == 1: return u.pop()
        if len(u) == 0: return bg
        return None
    if kind == "dom":
        return Counter(vals.tolist()).most_common(1)[0][0]
    if kind == "dom_nonbg":
        if len(nz) == 0: return bg
        return Counter(nz.tolist()).most_common(1)[0][0]
    return None

def _sep_lines(g, bg):
    """Indices of full rows/cols that are a single constant colour (candidate separators)."""
    H, W = g.shape
    rows = [r for r in range(H) if len(set(g[r, :].tolist())) == 1]
    cols = [c for c in range(W) if len(set(g[:, c].tolist())) == 1]
    return rows, cols

def _blocks_by_ratio(g, Ho, Wo):
    H, W = g.shape
    if Ho == 0 or Wo == 0 or H % Ho or W % Wo: return None
    bh, bw = H // Ho, W // Wo
    if bh * bw < 2: return None
    return [[g[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw] for c in range(Wo)] for r in range(Ho)]

def _block_reduce(pairs, tests):
    Ho, Wo = pairs[0][1].shape
    if any(o.shape != (Ho, Wo) for _, o in pairs): return None
    for kind in ("uniq_nonbg", "dom_nonbg", "dom"):
        def build(g):
            g = A(g); bg = bg_color(g); blocks = _blocks_by_ratio(g, Ho, Wo)
            if blocks is None: return None
            out = np.zeros((Ho, Wo), int)
            for r in range(Ho):
                for c in range(Wo):
                    v = _summ(blocks[r][c], bg, kind)
                    if v is None: return None
                    out[r, c] = v
            return out
        ok = True
        for i, o in pairs:
            b = build(i)
            if b is None or not eq(b, o): ok = False; break
        if ok:
            pr = [build(t) for t in tests]
            if all(p is not None for p in pr): return pr
    return None

# ----------------------------------------------------------------- (2) frame add / remove
def _frame(pairs, tests):
    # remove a t-thick border
    for t in (1, 2):
        if all(i.shape[0] > 2 * t and i.shape[1] > 2 * t and eq(A(i)[t:-t, t:-t], o) for i, o in pairs):
            return [A(A(x)[t:-t, t:-t]) for x in tests]
    # add a t-thick border of colour col
    for t in (1, 2):
        col = None; ok = True
        for i, o in pairs:
            H, W = i.shape
            if o.shape != (H + 2 * t, W + 2 * t): ok = False; break
            if not eq(o[t:-t, t:-t], A(i)): ok = False; break
            border = np.concatenate([o[0, :], o[-1, :], o[:, 0], o[:, -1]])
            u = set(border.tolist())
            if len(u) != 1: ok = False; break
            v = u.pop()
            if col is None: col = v
            elif col != v: ok = False; break
        if ok and col is not None:
            def emit(g, t=t, col=col):
                g = A(g); H, W = g.shape; out = np.full((H + 2 * t, W + 2 * t), col, int); out[t:-t, t:-t] = g; return out
            return [emit(x) for x in tests]
    return None

# ----------------------------------------------------------------- (3) bbox fill / outline
def _bbox_fill(pairs, tests):
    # per-connected-object bbox
    for mode in ("fill", "outline"):
        for sc in (True, False):
            def emit(g, mode=mode, sc=sc):
                g = A(g); bg = bg_color(g); out = g.copy()
                for ob in objects(g, bg, True, sc):
                    r0, c0, r1, c1 = ob["bbox"]; col = ob["color"]
                    if col is None: col = int(Counter(g[ob["mask"]].tolist()).most_common(1)[0][0])
                    if mode == "fill":
                        out[r0:r1 + 1, c0:c1 + 1] = col
                    else:
                        out[r0:r1 + 1, c0] = col; out[r0:r1 + 1, c1] = col
                        out[r0, c0:c1 + 1] = col; out[r1, c0:c1 + 1] = col
                return out
            if all(i.shape == o.shape and eq(emit(i), o) for i, o in pairs):
                return [A(emit(x)) for x in tests]
    # per-COLOUR bbox (all cells of a colour share one bounding box)
    for mode in ("fill", "outline"):
        def emitc(g, mode=mode):
            g = A(g); bg = bg_color(g); out = g.copy()
            for col in np.unique(g):
                if col == bg: continue
                rs, cs = np.where(g == col)
                r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
                if mode == "fill":
                    out[r0:r1 + 1, c0:c1 + 1] = col
                else:
                    out[r0:r1 + 1, c0] = col; out[r0:r1 + 1, c1] = col
                    out[r0, c0:c1 + 1] = col; out[r1, c0:c1 + 1] = col
            return out
        if all(i.shape == o.shape and eq(emitc(i), o) for i, o in pairs):
            return [A(emitc(x)) for x in tests]
    return None

def solve(task):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    tests = [A(tp["input"]) for tp in task["test"]]
    for fn in (_block_reduce, _frame, _bbox_fill):
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
    print("GRID2 (block-reduce / frame / bbox-fill, gradient-free): %d / %d  %s  (%.0fs)"
          % (len(solved), len(ch), split, time.time() - t0))
    open("/tmp/grid2_%s.ids" % split, "w").write(" ".join(solved))
