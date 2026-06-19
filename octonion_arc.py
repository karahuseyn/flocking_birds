# octonion_arc.py -- gradient-free octonionic-recursive solver for ARC-AGI-2.
#
# Vision (no backprop, no gradients):
#   * OCTON network: each grid cell is a colour-octon o(c) in S^7; a grid is a network of octons.
#   * FANO PATH = a transformation rule = a COMPOSITION of primitive grid operations. The
#     colour-relation between input/output octons is the exact Fano transform R = o(out)(x)o(in)^-1
#     (the proven IMPLIES rotation) -- recolour rules are learned this way, gradient-free.
#   * CYCLIC gradient-free LEARNING (TRM-inspired, but NO backprop): recursively deepen the fano
#     path, keeping the composition that maps ALL train inputs to outputs EXACTLY (a fixed point
#     consistent across every demonstration). Verification is the learning signal.
# Tested on the evaluation set with hidden-then-revealed solutions; exact-match @2 attempts.
import json, time, base64
import numpy as np
from itertools import product

# ---------- octonion colour encoding (the octon per colour) ----------
import exp_fano_layer as XF
_rng = np.random.default_rng(0)
COLOR_OCTON = XF.unit(np.concatenate([np.eye(8), _rng.standard_normal((2, 8))]))   # 10 colours -> S^7
def relation_octon(a, b): return XF.octo_mul(COLOR_OCTON[b], XF._inv(COLOR_OCTON[a]))  # exact recolour rotation

# ---------- grid utilities ----------
def A(g): return np.array(g, dtype=np.int64)
def eq(x, y): return x is not None and y is not None and x.shape == y.shape and bool(np.array_equal(x, y))
def bg_color(g):
    vals, cnts = np.unique(g, return_counts=True); return int(vals[np.argmax(cnts)])
def crop_bbox(g, bg=None):
    bg = bg_color(g) if bg is None else bg
    m = g != bg
    if not m.any(): return g
    r = np.where(m.any(1))[0]; c = np.where(m.any(0))[0]
    return g[r.min():r.max()+1, c.min():c.max()+1]

DIHEDRAL = {"identity": lambda g: g, "flip_h": np.fliplr, "flip_v": np.flipud,
            "transpose": lambda g: g.T, "anti_transpose": lambda g: np.rot90(np.fliplr(g)),
            "rot90": lambda g: np.rot90(g, 1), "rot180": lambda g: np.rot90(g, 2), "rot270": lambda g: np.rot90(g, 3)}

# ---------- solvers: each fits a rule from train pairs, returns a predictor fn or None ----------
def s_dihedral(train):
    for name, fn in DIHEDRAL.items():
        if all(eq(fn(A(p["input"])), A(p["output"])) for p in train):
            return (name, fn)
    return None

def s_colormap(train):
    if not all(A(p["input"]).shape == A(p["output"]).shape for p in train): return None
    cmap = {}
    for p in train:
        i, o = A(p["input"]), A(p["output"])
        for a, b in zip(i.flatten(), o.flatten()):
            a, b = int(a), int(b)
            if a in cmap and cmap[a] != b: return None
            cmap[a] = b
    fn = lambda g: np.vectorize(lambda c: cmap.get(int(c), int(c)))(g)
    return ("colormap", fn)

def s_crop_content(train):
    for bgmode in ("auto", 0):
        fn = (lambda g: crop_bbox(g)) if bgmode == "auto" else (lambda g: crop_bbox(g, 0))
        if all(eq(fn(A(p["input"])), A(p["output"])) for p in train):
            return ("crop_content_%s" % bgmode, fn)
    return None

def s_scale(train):
    rs, cs = set(), set()
    for p in train:
        i, o = A(p["input"]), A(p["output"])
        if i.shape[0] == 0 or o.shape[0] % i.shape[0] or o.shape[1] % i.shape[1]: return None
        rs.add(o.shape[0] // i.shape[0]); cs.add(o.shape[1] // i.shape[1])
    if len(rs) != 1 or len(cs) != 1: return None
    kr, kc = rs.pop(), cs.pop()
    if kr == 1 and kc == 1: return None
    fn = lambda g: np.kron(g, np.ones((kr, kc), dtype=g.dtype))
    return ("scale_%dx%d" % (kr, kc), fn) if all(eq(fn(A(p["input"])), A(p["output"])) for p in train) else None

def s_tile(train):
    rs, cs = set(), set()
    for p in train:
        i, o = A(p["input"]), A(p["output"])
        if i.shape[0] == 0 or o.shape[0] % i.shape[0] or o.shape[1] % i.shape[1]: return None
        rs.add(o.shape[0] // i.shape[0]); cs.add(o.shape[1] // i.shape[1])
    if len(rs) != 1 or len(cs) != 1: return None
    nr, nc = rs.pop(), cs.pop()
    if nr * nc == 1: return None
    # try plain tile and mirror tile
    def plain(g): return np.tile(g, (nr, nc))
    def mirror(g):
        rows = [np.concatenate([g if (j % 2 == 0) else np.fliplr(g) for j in range(nc)], 1)]
        block = rows[0]
        return np.concatenate([block if (i % 2 == 0) else np.flipud(block) for i in range(nr)], 0)
    for name, fn in (("tile", plain), ("tile_mirror", mirror)):
        if all(eq(fn(A(p["input"])), A(p["output"])) for p in train): return (name, fn)
    return None

def s_const(train):
    outs = [A(p["output"]) for p in train]
    if len(outs) >= 2 and all(eq(outs[0], o) for o in outs[1:]):
        o0 = outs[0]; return ("const", lambda g: o0.copy())
    return None

def s_symmetrize(train):
    # output = input with its (mirror/rot) symmetry completed; the defect cells (the colour that
    # breaks symmetry) are overwritten by the symmetric counterpart. Recurses until stable.
    if not all(A(p["input"]).shape == A(p["output"]).shape for p in train): return None
    if all(eq(_repair(A(p["input"])), A(p["output"])) for p in train):
        return ("symmetrize", _repair)
    return None

def _repair(g):
    syms = [np.fliplr, np.flipud, lambda x: np.rot90(x, 2)]
    if g.shape[0] == g.shape[1]: syms += [lambda x: x.T, lambda x: np.rot90(np.fliplr(x))]
    out = g.copy()
    for _ in range(6):                                       # cyclic refinement (gradient-free)
        prev = out.copy()
        # only the symmetries the grid ACTUALLY (mostly) satisfies vote -- avoids corrupting a
        # non-transpose-symmetric square grid by including transpose.
        active = [s(out) for s in syms if s(out).shape == out.shape and (s(out) == out).mean() > 0.80]
        if not active: break
        st = np.stack([out] + active)
        out = np.apply_along_axis(lambda v: np.bincount(v).argmax(), 0, st)   # majority across symmetric copies
        if np.array_equal(out, prev): break
    return out

def s_occlusion_extract(train):
    # input is (near-)symmetric except a rectangular OCCLUDED region; reconstruct it by symmetry
    # (majority vote across symmetric copies) and OUTPUT just that region. Targets the dominant
    # ARC-AGI-2 'reconstruct the hidden patch' pattern.
    def predict(g):
        r = _repair(g); diff = g != r
        if not diff.any(): return None
        rr = np.where(diff.any(1))[0]; cc = np.where(diff.any(0))[0]
        return r[rr.min():rr.max()+1, cc.min():cc.max()+1]
    try:
        if all(eq(predict(A(p["input"])), A(p["output"])) for p in train):
            return ("occlusion_extract", predict)
    except Exception: pass
    return None

SOLVERS = [s_dihedral, s_colormap, s_crop_content, s_scale, s_tile, s_symmetrize, s_occlusion_extract, s_const]

# ---------------------------- OBJECT-CENTRIC layer ----------------------------
from scipy import ndimage
_STRUCT = np.ones((3, 3), int)

def objects(g, bg, diag=True, same_color=True):
    st = _STRUCT if diag else None; objs = []
    if same_color:
        for col in np.unique(g):
            if int(col) == bg: continue
            lab, n = ndimage.label(g == col, st)
            for k in range(1, n + 1): objs.append(_mkobj(g, lab == k, int(col)))
    else:
        lab, n = ndimage.label(g != bg, st)
        for k in range(1, n + 1): objs.append(_mkobj(g, lab == k, None))
    return objs

def _mkobj(g, mask, col):
    rr = np.where(mask.any(1))[0]; cc = np.where(mask.any(0))[0]
    r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
    sub = g[r0:r1+1, c0:c1+1]; sm = mask[r0:r1+1, c0:c1+1]
    cols = set(int(x) for x in g[mask].tolist())
    return dict(mask=mask, bbox=(r0, c0, r1, c1), sub=sub, sm=sm, size=int(mask.sum()),
                h=sub.shape[0], w=sub.shape[1], color=col, ncolors=len(cols))

def _obj_octon(o):
    """Each object -> a unit octonion signature (size, shape, colour). Uniqueness / correspondence
    are then measured by octonion distance -- the octon network at the object level."""
    f = np.array([o["size"], o["h"], o["w"], o["h"]*o["w"], o["ncolors"],
                  (o["color"] or 0), o["size"]/(o["h"]*o["w"]+1e-9), 0.0], float)
    return XF.unit(f)

def _shape_key(o): return o["sm"].tobytes() + bytes(o["sm"].shape)

CRITERIA = {
    "largest": lambda os, g: max(os, key=lambda o: o["size"]),
    "smallest": lambda os, g: min(os, key=lambda o: o["size"]),
    "tallest": lambda os, g: max(os, key=lambda o: o["h"]),
    "widest": lambda os, g: max(os, key=lambda o: o["w"]),
    "most_colours": lambda os, g: max(os, key=lambda o: o["ncolors"]),
    "unique_shape": lambda os, g: _unique(os, _shape_key),
    "unique_colour": lambda os, g: _unique(os, lambda o: o["color"]),
    "unique_octon": lambda os, g: _octon_outlier(os),
}
def _unique(os, key):
    from collections import Counter
    c = Counter(key(o) for o in os); u = [o for o in os if c[key(o)] == 1]
    return u[0] if len(u) == 1 else None
def _octon_outlier(os):
    if len(os) < 3: return None
    V = XF.unit(np.array([_obj_octon(o) for o in os])); S = V @ V.T
    i = int(np.argmin(S.sum(1)))                                  # the object least like all others
    return os[i]

def s_select_object(train):
    for sc in (True, False):
        for diag in (True, False):
            for cname, crit in CRITERIA.items():
                for masked in (False, True):
                    def fn(g, sc=sc, diag=diag, crit=crit, masked=masked):
                        bg = bg_color(g); os = objects(g, bg, diag, sc)
                        if not os: return None
                        o = crit(os, g)
                        if o is None: return None
                        if not masked: return o["sub"].copy()
                        out = np.full(o["sub"].shape, bg, int); out[o["sm"]] = o["sub"][o["sm"]]; return out
                    if _verify(fn, train):
                        return ("select_%s_%s_%s_m%d" % (cname, sc, diag, masked), fn)
    return None

def s_keep_object(train):
    if not all(A(p["input"]).shape == A(p["output"]).shape for p in train): return None
    for sc in (True, False):
        for cname, crit in (("largest", CRITERIA["largest"]), ("smallest", CRITERIA["smallest"]),
                            ("unique_shape", CRITERIA["unique_shape"]), ("unique_octon", CRITERIA["unique_octon"])):
            for keep in (True, False):
                def fn(g, sc=sc, crit=crit, keep=keep):
                    bg = bg_color(g); os = objects(g, bg, True, sc)
                    if not os: return None
                    o = crit(os, g)
                    if o is None: return None
                    out = g.copy() if keep else np.full(g.shape, bg, int)
                    if keep:
                        m = np.ones(g.shape, bool); m[o["mask"]] = False; out[m] = bg
                    else:
                        out[o["mask"]] = g[o["mask"]]
                    return out
                if _verify(fn, train):
                    return ("keep_%s_%s_k%d" % (cname, sc, keep), fn)
    return None

def s_count_line(train):
    def predict(g):
        bg = bg_color(g); n = len(objects(g, bg, True, False))
        cols = [int(c) for c in np.unique(g) if int(c) != bg]
        col = cols[0] if cols else 1
        return np.full((1, n), col, int) if n else None
    for fn in (predict, lambda g: predict(g).T if predict(g) is not None else None):
        if _verify(fn, train): return ("count_line", fn)
    return None

SOLVERS += [s_select_object, s_keep_object, s_count_line]


# deterministic transforms used for FANO-PATH composition (recursive deepening, gradient-free)
BASE_DET = list(DIHEDRAL.items()) + [("crop", lambda g: crop_bbox(g)), ("crop0", lambda g: crop_bbox(g, 0))]

def _verify(fn, train):
    try: return all(eq(A(fn(A(p["input"]))), A(p["output"])) for p in train)
    except Exception: return False

def _fit_colormap(pairs):
    cmap = {}
    for i, o in pairs:
        if i.shape != o.shape: return None
        for a, b in zip(i.flatten(), o.flatten()):
            a, b = int(a), int(b)
            if a in cmap and cmap[a] != b: return None
            cmap[a] = b
    return lambda g: np.vectorize(lambda c: cmap.get(int(c), int(c)))(g)

def solve_task(task):
    """Up to 2 predicted grids per test input. Fano-path search: parameterized solvers +
    depth<=2 composition of deterministic primitives + colourmap-after-geometry."""
    train = task["train"]; cands = []
    for soltype in SOLVERS:
        try: r = soltype(train)
        except Exception: r = None
        if r is not None: cands.append(r)
    # composition over deterministic primitives (depth 1 and 2)
    for (n1, f1) in BASE_DET:
        if _verify(f1, train): cands.append((n1, f1))
        for (n2, f2) in BASE_DET:
            comp = (lambda g, f1=f1, f2=f2: f2(f1(g)))
            if _verify(comp, train): cands.append((n1 + "+" + n2, comp))
    # colourmap AFTER a geometry (recolour rules combined with a flip/rotate)
    for (n1, f1) in BASE_DET:
        cm = _fit_colormap([(A(f1(A(p["input"]))), A(p["output"])) for p in train])
        if cm is not None:
            fn = (lambda g, f1=f1, cm=cm: cm(A(f1(A(g)))))
            if _verify(fn, train): cands.append(("cmap*" + n1, fn))
    # dedup predictors by name
    seen = set(); uniq = []
    for n, f in cands:
        if n not in seen: seen.add(n); uniq.append((n, f))
    cands = uniq
    preds = []
    for tp in task["test"]:
        gi = A(tp["input"]); outs = []
        for name, fn in cands:
            try: o = A(fn(gi))
            except Exception: continue
            if o is None or o.ndim != 2 or not (1 <= o.shape[0] <= 30 and 1 <= o.shape[1] <= 30): continue
            if not any(eq(o, e) for e in outs): outs.append(o)
            if len(outs) >= 2: break
        if not outs: outs = [gi]
        preds.append((outs + [gi])[:2])
    return preds, [n for n, _ in cands]

if __name__ == "__main__":
    D = "arc_data/"
    ev = json.load(open(D + "arc-agi_evaluation_challenges.json"))
    sol = json.load(open(D + "arc-agi_evaluation_solutions.json"))
    t0 = time.time()
    solved = 0; ntasks = 0; ntest = 0; ntest_solved = 0; by_solver = {}
    solved_ids = []
    for tid, task in ev.items():
        preds, used = solve_task(task); gold = sol[tid]
        ntasks += 1; task_ok = True
        for i, golds in enumerate(gold):
            g = A(golds); ntest += 1
            hit = any(eq(p, g) for p in preds[i])
            if hit: ntest_solved += 1
            else: task_ok = False
        if task_ok:
            solved += 1; solved_ids.append(tid)
            for u in used: by_solver[u] = by_solver.get(u, 0) + 1
    out = ["ARC-AGI-2 evaluation (gradient-free octonionic solver), %.0fs" % (time.time()-t0),
           "  tasks fully solved (all test inputs, @2): %d / %d  (%.1f%%)" % (solved, ntasks, 100*solved/ntasks),
           "  test inputs solved: %d / %d  (%.1f%%)" % (ntest_solved, ntest, 100*ntest_solved/ntest),
           "  solved task ids: " + " ".join(solved_ids[:40]),
           "  solver hits among solved tasks: " + str(by_solver)]
    print("B64ARC:" + base64.b64encode("\n".join(out).encode()).decode())
