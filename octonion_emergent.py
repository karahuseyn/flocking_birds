# octonion_emergent.py -- a family of EMERGENT operators, each SOLVED from the task's data and
# verified EXACTLY, with NO predefined task-transform labels. Extends the emergent direction beyond
# whole-grid affine (octonion_incontext) and object re-pose paths (octonion_paths) to the high-yield
# ARC families: integer scaling, mirror/dihedral tiling (fractal), content cropping, and
# symmetry repair (occlusion fill). Every operator infers its parameters from the train pairs and is
# accepted only if it reproduces every pair exactly -- verification is the only learning signal.
import json, time
import numpy as np
from octonion_arc import A, eq, bg_color, DIHEDRAL

DH = list(DIHEDRAL.values())

# ---- operator 1: integer upscale (each cell -> kr x kc block), k inferred from shape ratio ----
def op_upscale(pairs):
    i0, o0 = pairs[0]
    if o0.shape[0] % i0.shape[0] or o0.shape[1] % i0.shape[1]: return None
    kr, kc = o0.shape[0] // i0.shape[0], o0.shape[1] // i0.shape[1]
    if (kr, kc) == (1, 1): return None
    fn = lambda g: np.kron(g, np.ones((kr, kc), int))
    return fn if all(eq(fn(i), o) for i, o in pairs) else None

# ---- operator 2: dihedral tiling (output is a grid of dihedral re-poses of the input) ----
def op_tile(pairs):
    i0, o0 = pairs[0]
    if o0.shape[0] % i0.shape[0] or o0.shape[1] % i0.shape[1]: return None
    nr, nc = o0.shape[0] // i0.shape[0], o0.shape[1] // i0.shape[1]
    if (nr, nc) == (1, 1): return None
    h, w = i0.shape
    arr = {}                                                          # (bi,bj) -> dihedral fn chosen from pair 0
    for bi in range(nr):
        for bj in range(nc):
            block = o0[bi*h:(bi+1)*h, bj*w:(bj+1)*w]
            ch = next((fn for fn in DH if fn(i0).shape == block.shape and np.array_equal(fn(i0), block)), None)
            if ch is None: return None
            arr[(bi, bj)] = ch
    def fn(g):
        h, w = g.shape; out = np.zeros((h*nr, w*nc), int)
        for (bi, bj), f in arr.items(): out[bi*h:(bi+1)*h, bj*w:(bj+1)*w] = f(g)
        return out
    return fn if all(eq(fn(i), o) for i, o in pairs) else None

# ---- operator 3: content crop -- output is a data-derived crop window of the input ----
def _crop_rules(g):
    bg = bg_color(g); m = g != bg
    if not m.any(): return {}
    r = np.where(m.any(1))[0]; c = np.where(m.any(0))[0]
    out = {"nonbg": g[r.min():r.max()+1, c.min():c.max()+1]}
    for v in np.unique(g):
        mm = g == v
        rr = np.where(mm.any(1))[0]; cc = np.where(mm.any(0))[0]
        out["c%d" % v] = g[rr.min():rr.max()+1, cc.min():cc.max()+1]   # bbox of colour v
    return out

def op_crop(pairs):
    keys = set(_crop_rules(pairs[0][0]))
    for k in list(keys):
        try:
            if all(eq(_crop_rules(i).get(k), o) for i, o in pairs):
                return (lambda g, _k=k: _crop_rules(g).get(g.shape and _k))
        except Exception: pass
    return None

# ---- operator 4: symmetry repair (occlusion fill) -- the classic ARC symmetry family ----
def _syms(shape):
    h, w = shape; S = []
    S.append(lambda r, c: (r, w-1-c)); S.append(lambda r, c: (h-1-r, c))            # mirrors
    S.append(lambda r, c: (h-1-r, w-1-c))                                            # 180
    if h == w:
        S.append(lambda r, c: (c, r)); S.append(lambda r, c: (w-1-c, h-1-r))        # transposes
    return S

def op_symfill(pairs):
    # detect the single hole colour: cells that change must all be that colour in the input
    hole = None
    for i, o in pairs:
        if i.shape != o.shape: return None
        d = i != o
        if not d.any(): continue
        hv = set(np.unique(i[d]).tolist())
        if len(hv) != 1: return None
        h = hv.pop()
        if (o[d] == h).any(): return None
        if hole is None: hole = h
        elif hole != h: return None
    if hole is None: return None

    def periods(g, mask):                                            # smallest valid h/v periods on known cells
        h, w = g.shape; pr = pc = None
        for p in range(1, h):
            ok = all(g[r, c] == g[r-p, c] for r in range(p, h) for c in range(w)
                     if mask[r, c] and mask[r-p, c])
            if ok: pr = p; break
        for p in range(1, w):
            ok = all(g[r, c] == g[r, c-p] for r in range(h) for c in range(p, w)
                     if mask[r, c] and mask[r, c-p])
            if ok: pc = p; break
        return pr, pc

    def fill(g):
        g = g.copy(); h, w = g.shape; known = g != hole
        syms = _syms(g.shape)
        # validate which mirror/rotation symmetries hold on known overlap
        valid = [s for s in syms if all(
            (lambda rc: not (0 <= rc[0] < h and 0 <= rc[1] < w) or not known[rc] or g[rc] == g[r, c])(s(r, c))
            for r in range(h) for c in range(w) if known[r, c])]
        pr, pc = periods(g, known)
        for _ in range(4):
            for r in range(h):
                for c in range(w):
                    if g[r, c] != hole: continue
                    votes = {}
                    for s in valid:
                        rr, cc = s(r, c)
                        if 0 <= rr < h and 0 <= cc < w and g[rr, cc] != hole:
                            votes[g[rr, cc]] = votes.get(g[rr, cc], 0) + 1
                    for p, ax in ((pr, 0), (pc, 1)):
                        if p:
                            for k in range(1, max(h, w)):
                                for sgn in (-1, 1):
                                    rr, cc = (r + sgn*k*p, c) if ax == 0 else (r, c + sgn*k*p)
                                    if 0 <= rr < h and 0 <= cc < w and g[rr, cc] != hole:
                                        votes[g[rr, cc]] = votes.get(g[rr, cc], 0) + 1
                    if votes: g[r, c] = max(votes, key=votes.get)
            if (g != hole).all(): break
        return g
    return fill if all(eq(fill(i), o) for i, o in pairs) else None

OPS = [op_upscale, op_tile, op_crop, op_symfill]

def solve(task):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    for op in OPS:
        try: fn = op(pairs)
        except Exception: fn = None
        if fn is None: continue
        try: return [A(fn(A(tp["input"]))) for tp in task["test"]]
        except Exception: continue
    return None


if __name__ == "__main__":
    import sys
    D = "arc_data/"; split = sys.argv[1] if len(sys.argv) > 1 else "training"
    ch = json.load(open(D + "arc-agi_%s_challenges.json" % split))
    sol = json.load(open(D + "arc-agi_%s_solutions.json" % split))
    t0 = time.time(); by = {op.__name__: [] for op in OPS}; solved = []
    for tid, task in ch.items():
        pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
        try: preds = solve(task)
        except Exception: preds = None
        if preds is None: continue
        if all(eq(preds[i], A(g)) for i, g in enumerate(sol[tid])):
            solved.append(tid)
            for op in OPS:
                try:
                    if op(pairs) is not None: by[op.__name__].append(tid); break
                except Exception: pass
    print("EMERGENT operator family (no predefined task-transforms): %d / %d  %s  (%.0fs)"
          % (len(solved), len(ch), split, time.time() - t0))
    for k, v in by.items(): print("   %-12s %d" % (k, len(v)))
