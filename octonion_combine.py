# octonion_combine.py
# ============================================================================
# Growing the vocabulary the RIGHT way -- a new MECHANISM, not a variant of an old
# one.  (Multiplying CA symmetry-subgroups saturated at +0; coverage needs a
# genuinely different computational primitive.)
#
# PANEL COMBINATION.  Many ARC tasks present the input as several equal PANELS
# (split by a separator line, or as equal halves/thirds) that are merged cellwise
# into one output: logical AND / OR / XOR, overlay, difference, ...  We do NOT name
# any of these.  We detect the partition from the grids themselves and LEARN the
# cellwise combination table
#       T : (p_1[r,c], p_2[r,c], ..., p_k[r,c]) -> out[r,c]
# from the task's own pairs, accepting only on EXACT reproduction of every
# demonstration.  The table is un-named and fit from data; gradient-free, no
# backprop.  This is a binary/k-ary cellwise operator over sub-grids -- a mechanism
# absent from the CA (single grid), substitution (expansion) and affine families.
# ============================================================================
import json, time, sys
import numpy as np
from octonion_arc import A, eq

def _split_by_separators(g):
    """Panels from removing full separator lines.  The separator colour is the one
    whose full rows/cols actually split the grid into >=2 EQUAL bands (so an
    accidental constant line of another colour does not abort detection)."""
    g = A(g); res = []
    for axis in (0, 1):
        line = g if axis == 0 else g.T                       # rows of `line` = lines along this axis
        for c in np.unique(g):
            seps = [k for k in range(line.shape[0]) if np.all(line[k] == c)]
            if not seps: continue
            bands = []; start = 0
            for s in seps + [line.shape[0]]:
                if s > start: bands.append((start, s))
                start = s + 1
            panels = [g[a:b, :] if axis == 0 else g[:, a:b] for a, b in bands]
            shp = panels[0].shape
            if len(panels) >= 2 and all(p.shape == shp for p in panels): res.append(panels)
    return res

def _split_equal(g, out_shape):
    """Panels from an equal n-way split matching the output shape."""
    g = A(g); H, W = g.shape; oh, ow = out_shape; res = []
    if oh == H and ow and W % ow == 0 and W // ow >= 2:
        n = W // ow; res.append([g[:, k * ow:(k + 1) * ow] for k in range(n)])
    if ow == W and oh and H % oh == 0 and H // oh >= 2:
        n = H // oh; res.append([g[k * oh:(k + 1) * oh, :] for k in range(n)])
    return res

def _partitions(g, out_shape):
    parts = [p for p in _split_by_separators(g) if p[0].shape == out_shape]
    parts += _split_equal(g, out_shape)
    return parts

def _learn(pairs, choose):
    """choose: index into the partition list (so the SAME partition scheme is used
    for every pair).  Learn the cellwise k-ary table; None on contradiction."""
    T = {}; arity = None
    for i, o in pairs:
        parts = _partitions(i, o.shape)
        if choose >= len(parts): return None
        panels = parts[choose]
        if arity is None: arity = len(panels)
        if len(panels) != arity: return None
        stacks = [p.ravel() for p in panels]; ov = A(o).ravel()
        for j in range(len(ov)):
            key = tuple(int(s[j]) for s in stacks); v = int(ov[j])
            if T.get(key, v) != v: return None
            T[key] = v
    return T, arity

def _apply(g, out_shape, T, arity, choose):
    parts = _partitions(g, out_shape)
    if choose >= len(parts): return None
    panels = parts[choose]
    if len(panels) != arity: return None
    stacks = [p.ravel() for p in panels]; n = len(stacks[0]); out = np.empty(n, int)
    for j in range(n):
        key = tuple(int(s[j]) for s in stacks); out[j] = T.get(key, key[0])
    return out.reshape(panels[0].shape)

def combine_solver(pairs):
    osh = pairs[0][1].shape
    if not all(o.shape == osh for _, o in pairs): return None     # constant output shape
    if any(i.shape == osh for i, _ in pairs): return None         # input must be larger (panelled)
    for choose in range(6):
        r = _learn(pairs, choose)
        if r is None: continue
        T, arity = r
        fn = lambda g, T=T, ar=arity, c=choose, os=osh: _apply(A(g), os, T, ar, c)
        try:
            if all(fn(i) is not None and eq(fn(i), o) for i, o in pairs): return fn
        except Exception: continue
    return None

def solve(task):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    fn = combine_solver(pairs)
    if fn is None: return None
    try:
        out = [fn(A(tp["input"])) for tp in task["test"]]
        return None if any(o is None for o in out) else [A(o) for o in out]
    except Exception: return None


if __name__ == "__main__":
    DIR = "arc_data/"; split = sys.argv[1] if len(sys.argv) > 1 else "training"
    ch = json.load(open(DIR + "arc-agi_%s_challenges.json" % split))
    sol = json.load(open(DIR + "arc-agi_%s_solutions.json" % split))
    t0 = time.time(); solved = []
    for tid, task in ch.items():
        try: pr = solve(task)
        except Exception: pr = None
        if pr and all(eq(pr[i], A(g)) for i, g in enumerate(sol[tid])): solved.append(tid)
    print("PANEL COMBINATION (cellwise k-ary, gradient-free): %d / %d  %s  (%.0fs)"
          % (len(solved), len(ch), split, time.time() - t0))
    open("/tmp/combine_%s.ids" % split, "w").write(" ".join(solved))
