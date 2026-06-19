# octonion_fractal.py
# ============================================================================
# Wolfram, used UNCONVENTIONALLY: substitution systems and nested / self-similar
# (fractal) rewrites, rather than the shape-preserving cellular automata of
# octonion_wolfram.py.  This is the other half of Wolfram's computational universe
# -- the one that PRODUCES fractals (Rule 90 -> Sierpinski, nested tilings).
#
# Mathematical setting.  A substitution system is a map sigma : F -> F^{a x b}
# expanding every cell into an a x b block.  Applied to a grid g of shape (H,W) it
# yields a grid of shape (aH, bW).  Two rule families, both LEARNED from the task's
# own pairs and accepted only on EXACT reproduction (gradient-free, un-named):
#
#   * FIXED STAMP            sigma(c) is a constant block, independent of the input
#     ("every pixel of colour c becomes this icon").
#
#   * SELF-REFERENTIAL / FRACTAL   the block written for a cell is the WHOLE input
#     grid itself (optionally recoloured), gated by a learned per-colour predicate:
#         out[block r,c] = T_c(g)   with T_c in { g, recolour(g), constant fill }.
#     Choosing T_c = g on a foreground predicate and constant elsewhere makes the
#     output a self-similar nesting of g inside g -- the canonical ARC fractal, and
#     a genuine fixed-shape-ratio nonlinear (input-dependent) transform: the same
#     rule yields a different, scale-coupled output for every grid, exactly the
#     non-linearity a colour bijection or an octonionic rotation cannot express.
#
# We unify both via a per-colour classification of the observed output blocks:
# constant-across-pairs -> fixed stamp;  equal to the pair's input (up to a fixed
# recolour) -> fractal;  else reject.
# ============================================================================
import json, time, sys
import numpy as np
from octonion_arc import A, eq

def _ratios(pairs):
    """Integer, consistent block size (a,b) with out = (a*H, b*W); else None."""
    a0 = pairs[0][1].shape[0] / pairs[0][0].shape[0]; b0 = pairs[0][1].shape[1] / pairs[0][0].shape[1]
    if a0 < 1 or b0 < 1 or a0 != int(a0) or b0 != int(b0): return None
    a, b = int(a0), int(b0)
    for i, o in pairs:
        if o.shape != (a * i.shape[0], b * i.shape[1]): return None
    return a, b

def _blocks(i, o, a, b):
    """The a x b output block for every input cell: dict (r,c) -> block."""
    return {(r, c): o[r * a:(r + 1) * a, c * b:(c + 1) * b] for r in range(i.shape[0]) for c in range(i.shape[1])}

def _colormap(src, dst):
    """A consistent colour bijection src->dst over equal-shaped grids, or None."""
    if src.shape != dst.shape: return None
    cm = {}
    for x, y in zip(src.ravel().tolist(), dst.ravel().tolist()):
        if cm.get(x, y) != y: return None
        cm[x] = y
    return cm

def _learn(pairs, a, b):
    """Classify each colour's substitution: 'stamp' (constant block), 'self' (the
    input grid, identity), or 'recolour' (input under a fixed colour map)."""
    rule = {}                                    # colour -> ('stamp', block) | ('self', None) | ('recolour', cm)
    for i, o in pairs:
        i = A(i); o = A(o); B = _blocks(i, o, a, b)
        for (r, c), blk in B.items():
            col = int(i[r, c])
            same_in = (a, b) == i.shape and np.array_equal(blk, i)
            if col not in rule:
                if same_in: rule[col] = ("self", None)
                elif (a, b) == i.shape and (cm := _colormap(i, blk)) is not None and any(k != v for k, v in cm.items()):
                    rule[col] = ("recolour", cm)
                else: rule[col] = ("stamp", blk.copy())
            else:
                kind, payload = rule[col]
                if kind == "self":
                    if not same_in: return None
                elif kind == "recolour":
                    cm = _colormap(i, blk)
                    if cm is None or any(payload.get(k, v) != v for k, v in cm.items()): return None
                else:                            # stamp must stay constant across every occurrence/pair
                    if blk.shape != payload.shape or not np.array_equal(blk, payload): return None
    return rule

def _apply(g, rule, a, b):
    g = A(g); H, W = g.shape; out = np.zeros((a * H, b * W), int)
    for r in range(H):
        for c in range(W):
            col = int(g[r, c]); spec = rule.get(col)
            if spec is None: blk = np.zeros((a, b), int)
            elif spec[0] == "self": blk = g
            elif spec[0] == "recolour": blk = np.vectorize(lambda x: spec[1].get(int(x), int(x)))(g)
            else: blk = spec[1]
            if blk.shape != (a, b): blk = np.zeros((a, b), int)   # 'self'/'recolour' need a==H,b==W
            out[r * a:(r + 1) * a, c * b:(c + 1) * b] = blk
    return out

def substitution_solver(pairs):
    r = _ratios(pairs)
    if r is None: return None
    a, b = r
    rule = _learn(pairs, a, b)
    if rule is None: return None
    if all(eq(_apply(i, rule, a, b), o) for i, o in pairs):
        return lambda g, R=rule, A_=a, B_=b: _apply(g, R, A_, B_)
    return None

def solve(task):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    fn = substitution_solver(pairs)
    if fn is None: return None
    try: return [A(fn(A(tp["input"]))) for tp in task["test"]]
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
    print("FRACTAL / substitution system (gradient-free): %d / %d  %s  (%.0fs)"
          % (len(solved), len(ch), split, time.time() - t0))
    open("/tmp/fractal_%s.ids" % split, "w").write(" ".join(solved))
