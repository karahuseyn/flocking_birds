# octonion_layered.py -- LAYERED, MULTI-STEP emergent search. Still no predefined task-transforms:
# every step is either a structural forward transform whose parameters come from the grid itself, or
# an emergent closer that SOLVES its rule from the (transformed-input -> output) pairs and is accepted
# only on EXACT reproduction. A pipeline is a chain of forward transforms followed by one emergent
# closer; the search is a depth-bounded DFS (iterative deepening) with state memoisation. This lifts
# the single-operator ceiling to compositions like crop->tile, dihedral->recolour, upscale->symfill.
# No gradients, no backprop -- verification across every demonstration is the only learning signal.
# Measured (training): depth 0 = 40/1000, depth 2 = 53/1000 (composition adds +13: crop->tile,
# dihedral->recolour, upscale->symfill, ...). Best purely-emergent result, beating the hand-built
# operator bank (31). ARC-2 evaluation: 0/120 -- the wall holds even with composition; those tasks
# need reasoning beyond chains of closed-form operators.
import json, time
import numpy as np
from octonion_arc import A, eq, bg_color, crop_bbox, DIHEDRAL
import octonion_incontext as IC
import octonion_emergent as EM

# ---------- emergent CLOSERS: (transformed-input -> output) pairs -> grid->grid fn, or None ----------
def _colormap(pairs):
    if not all(i.shape == o.shape for i, o in pairs): return None
    cm = {}
    for i, o in pairs:
        for a, b in zip(i.ravel(), o.ravel()):
            a, b = int(a), int(b)
            if cm.get(a, b) != b: return None
            cm[a] = b
    return lambda g: np.vectorize(lambda c: cm.get(int(c), int(c)))(A(g))

def _ic_close(pairs):
    r = IC.infer(pairs, iters=40)
    if r is None: return None
    Amat, b, osh, rc = r
    return lambda g: A(rc(IC._warp(A(g), Amat, b, osh(A(g).shape))))

def single_step(pairs):
    """Try every emergent operator as a closer; return the first grid->grid fn reproducing all pairs."""
    for name, d in DIHEDRAL.items():                              # dihedral (includes identity)
        if all(eq(d(i), o) for i, o in pairs): return d
    for name, d in DIHEDRAL.items():                             # dihedral + colour map
        tp = [(d(i), o) for i, o in pairs]
        cm = _colormap(tp)
        if cm is not None and all(eq(cm(d(i)), o) for i, o in pairs):
            return (lambda g, _d=d, _cm=cm: _cm(_d(A(g))))
    for op in (EM.op_upscale, EM.op_tile, EM.op_crop, EM.op_symfill):  # structural emergent ops
        try: fn = op(pairs)
        except Exception: fn = None
        if fn is not None: return fn
    try: return _ic_close(pairs)                                  # emergent affine + Fano recolour
    except Exception: return None

# ---------- structural FORWARD transforms: grid -> grid, parameters from the grid itself ----------
def _crop_rank(g, rank, asc):
    g = A(g); vals, cnts = np.unique(g, return_counts=True)
    order = np.argsort(cnts if asc else -cnts)
    if rank >= len(order): return g
    v = vals[order[rank]]; m = g == v
    r = np.where(m.any(1))[0]; c = np.where(m.any(0))[0]
    return g[r.min():r.max()+1, c.min():c.max()+1]

def _mirror(g, mode):
    g = A(g)
    if mode == "h": return np.hstack([g, np.fliplr(g)])
    if mode == "v": return np.vstack([g, np.flipud(g)])
    top = np.hstack([g, np.fliplr(g)]); return np.vstack([top, np.flipud(top)])

FT = ([(n, d) for n, d in DIHEDRAL.items() if n != "identity"]
      + [("crop_nonbg", lambda g: crop_bbox(A(g)))]
      + [("crop_rank%d_%s" % (r, a), (lambda g, _r=r, _a=(a == "asc"): _crop_rank(g, _r, _a)))
         for r in (0, 1) for a in ("asc", "desc")]
      + [("upscale%d" % k, (lambda g, _k=k: np.kron(A(g), np.ones((_k, _k), int)))) for k in (2, 3)]
      + [("tile%dx%d" % (a, b), (lambda g, _a=a, _b=b: np.tile(A(g), (_a, _b)))) for a, b in ((2, 2), (1, 2), (2, 1))]
      + [("mirror_%s" % m, (lambda g, _m=m: _mirror(g, _m))) for m in ("h", "v", "quad")])

# ---------- layered DFS search over [forward transforms] + [emergent closer] ----------
def search(inputs, outs, D):
    seen = set()
    def rec(cur, fns, depth):
        key = tuple(c.tobytes() + bytes(repr(c.shape), "utf8") for c in cur)
        if key in seen: return None
        seen.add(key)
        cl = single_step(list(zip(cur, outs)))
        if cl is not None and all(eq(A(cl(c)), o) for c, o in zip(cur, outs)):
            return fns + [cl]
        if depth == 0: return None
        for name, ft in FT:
            try: nxt = [A(ft(c)) for c in cur]
            except Exception: continue
            if any(n is None or n.size == 0 or max(n.shape) > 45 for n in nxt): continue
            if all(np.array_equal(n, c) for n, c in zip(nxt, cur)): continue
            r = rec(nxt, fns + [ft], depth - 1)
            if r is not None: return r
        return None
    return rec([A(i) for i in inputs], [], D)

def solve(task, D=2):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    ins = [i for i, _ in pairs]; outs = [o for _, o in pairs]
    fns = search(ins, outs, D)
    if fns is None: return None
    def apply(g):
        g = A(g)
        for f in fns: g = A(f(g))
        return g
    return [apply(tp["input"]) for tp in task["test"]]


if __name__ == "__main__":
    import sys
    DIR = "arc_data/"; split = sys.argv[1] if len(sys.argv) > 1 else "training"
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    ch = json.load(open(DIR + "arc-agi_%s_challenges.json" % split))
    sol = json.load(open(DIR + "arc-agi_%s_solutions.json" % split))
    t0 = time.time(); solved = []
    for tid, task in ch.items():
        try: preds = solve(task, depth)
        except Exception: preds = None
        if preds is None: continue
        if all(eq(preds[i], A(g)) for i, g in enumerate(sol[tid])): solved.append(tid)
    print("LAYERED multi-step emergent search (depth %d, no predefined transforms): %d / %d  %s  (%.0fs)"
          % (depth, len(solved), len(ch), split, time.time() - t0))
