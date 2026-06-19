# octonion_eml.py
# ============================================================================
# Integrating Odrzywolek's EML operator (arXiv:2603.21852, "All elementary
# functions from a single operator") into our octonionic transformation space.
#
# The paper's result: a SINGLE binary operator
#       eml(x, y) = exp(x) - ln(y)
# together with the constant 1 generates the whole elementary-function repertoire
# (constants e, pi, i; +, -, *, /, ^; exp, ln, the trig family via Euler).  Every
# expression is a uniform binary tree, grammar  S -> 1 | eml(S, S),  and the author
# *found the operator by exhaustive search* -- a gradient-free program search, the
# same philosophy we use for ARC.
#
# We develop two layers, kept honestly separate:
#
#  (1) OCTONIONIC EML  (the mathematical core).  Lift exp, log to the octonions O
#      (a normed division algebra).  For q = a + v, v = Im(q):
#          exp(q) = e^a (cos|v| + (v/|v|) sin|v|)
#          log(q) = ln|q| + (v/|v|) * atan2(|v|, a)
#      (well defined because, by Artin's theorem, 1 and v generate a commutative
#      C-isomorphic subalgebra, so the complex identities transport verbatim.)  Then
#          EML_O(x, y) = exp(x) - log(y),  constant 1.
#      The paper's single imaginary unit i becomes the SEVEN Fano units e1..e7:
#      EML_O realises Euler rotations e^{e_k t} = cos t + e_k sin t in each of the 7
#      octonionic planes -- the algebraic source of periodicity, and the bridge to
#      our Fano/octonion ARC encoding.  We verify these identities numerically.
#
#  (2) GRADIENT-FREE EML SYMBOLIC REGRESSION (the ARC test).  The grammar is
#      enumerable, so we do bottom-up enumeration with OBSERVATIONAL EQUIVALENCE
#      (dedup expressions by their value vector) -- faithful to the paper's own
#      exhaustive search, and strictly no backprop.  We search a closed form
#          colour(r, c, v) = round(Re( EML-tree over {1, i, r, c, v} ))
#      that reproduces EVERY train cell exactly, then apply it to the test grid.
#      Because oscillation needs i (real EML cannot oscillate), the imaginary unit
#      -- the paper's, generalised to ours -- is exactly what unlocks ARC's
#      periodic / positional colourings, which the CA and substitution solvers miss.
# ============================================================================
import json, time, sys, itertools
import numpy as np
from octonion_arc import A, eq

# ----------------------------------------------------------------- (1) octonionic EML
import exp_fano_layer as XF                       # octo_mul, _conj, unit

def _split(q): return q[..., 0], q[..., 1:]
def onorm(q):  return np.sqrt((q * q).sum(-1))

def oexp(q):
    a, v = _split(q); nv = np.sqrt((v * v).sum(-1, keepdims=True))
    ea = np.exp(a)[..., None]
    dir = np.where(nv > 1e-12, v / np.where(nv > 1e-12, nv, 1.0), 0.0)
    out = np.concatenate([np.cos(nv), dir * np.sin(nv)], -1)
    return ea * out

def olog(q):
    a, v = _split(q); n = onorm(q)[..., None]; nv = np.sqrt((v * v).sum(-1, keepdims=True))
    dir = np.where(nv > 1e-12, v / np.where(nv > 1e-12, nv, 1.0), 0.0)
    ang = np.arctan2(nv[..., 0], a)[..., None]
    return np.concatenate([np.log(np.maximum(n[..., 0], 1e-30))[..., None], dir * ang], -1)

def EML_O(x, y): return oexp(x) - olog(y)

def _verify_octonionic_eml():
    rng = np.random.default_rng(0); ok = []
    one = np.zeros(8); one[0] = 1.0
    # e^x = EML_O(x, 1)
    x = rng.standard_normal(8)
    ok.append(("exp = EML_O(.,1)", np.allclose(EML_O(x, one), oexp(x), atol=1e-9)))
    # log/exp inverse on a domain
    q = np.concatenate([[abs(rng.standard_normal())], 0.3 * rng.standard_normal(7)])
    ok.append(("exp(log q) = q", np.allclose(oexp(olog(q)), q, atol=1e-7)))
    # the 7-unit Euler formula  e^{e_k t} = cos t + e_k sin t
    eul = True
    for k in range(1, 8):
        t = 0.7; q = np.zeros(8); q[k] = t; r = oexp(q)
        exp_r = np.zeros(8); exp_r[0] = np.cos(t); exp_r[k] = np.sin(t)
        eul &= np.allclose(r, exp_r, atol=1e-9)
    ok.append(("7-unit Euler e^{e_k t}=cos t+e_k sin t", eul))
    # multiplication law on a commutative C-subalgebra spanned by 1,e_k: x*y = exp(log x+log y)
    k = 3; x = np.zeros(8); x[0] = 1.2; x[k] = 0.4; y = np.zeros(8); y[0] = 0.7; y[k] = -0.9
    xy = XF.octo_mul(x, y); law = oexp(olog(x) + olog(y))
    ok.append(("x*y = exp(log x + log y) on C-subalgebra", np.allclose(xy, law, atol=1e-7)))
    return ok

# ----------------------------------------------------------------- (2) EML symbolic regression
# Bottom-up enumeration over leaves {1, i, r, c, v} with the single operator eml,
# evaluated on the task's cells (complex), deduped by observational equivalence.
def eml_c(x, y):                                  # complex EML
    with np.errstate(all="ignore"):
        return np.exp(x) - np.log(y)

def _enumerate(env, target, max_exprs=500, max_layer=3):
    """env: dict name->complex vector over N samples.  target: int vector (the
    output colours).  Returns an AST whose round(Re(.)) equals target on all
    samples, or None.  Observational-equivalence dedup keeps it tractable."""
    N = len(target)
    leaves = {"1": np.ones(N, complex), "i": np.full(N, 1j)}
    leaves.update({k: v.astype(complex) for k, v in env.items()})
    exprs = []; seen = set()
    def sig(vec):
        z = np.round(vec.real, 4) + 1j * np.round(vec.imag, 4)
        return z.tobytes()
    def add(ast, vec):
        if not np.all(np.isfinite(vec)) or np.max(np.abs(vec)) > 1e6: return False
        s = sig(vec)
        if s in seen: return False
        seen.add(s); exprs.append((ast, vec)); return True
    def match(vec):
        pred = np.round(vec.real).astype(int)
        return np.array_equal(np.clip(pred, 0, 9), target) and np.all((pred >= 0) & (pred <= 9))
    for name, vec in leaves.items():
        if add((name,), vec) and match(vec): return (name,)
    layer_start = 0
    for _ in range(max_layer):
        cur = list(exprs); produced = []
        for ia in range(len(cur)):
            for ib in range(len(cur)):
                va, vb = cur[ia][1], cur[ib][1]
                vec = eml_c(va, vb)
                ast = ("eml", cur[ia][0], cur[ib][0])
                if add(ast, vec):
                    produced.append((ast, vec))
                    if match(vec): return ast
                    if len(exprs) >= max_exprs: return None
        if not produced: break
    return None

def _eval_ast(ast, env):
    if ast[0] == "1": return np.ones_like(next(iter(env.values())), complex)
    if ast[0] == "i": return np.full_like(next(iter(env.values())), 1j, complex)
    if ast[0] in env: return env[ast[0]].astype(complex)
    return eml_c(_eval_ast(ast[1], env), _eval_ast(ast[2], env))

def _features(grids):
    """Per-cell features for a list of grids: row, col, value (+ normalised r,c)."""
    R = []; C = []; V = []
    for g in grids:
        g = A(g); H, W = g.shape
        rr, cc = np.mgrid[0:H, 0:W]
        R.append(rr.ravel()); C.append(cc.ravel()); V.append(g.ravel())
    return {"r": np.concatenate(R).astype(float), "c": np.concatenate(C).astype(float),
            "v": np.concatenate(V).astype(float)}

def regress_solver(pairs, max_exprs=500):
    """Discover colour(r,c,v) as an EML closed form reproducing every train cell."""
    if not all(i.shape == o.shape for i, o in pairs): return None
    ins = [i for i, _ in pairs]; outs = [o for _, o in pairs]
    env = _features(ins); target = np.concatenate([A(o).ravel() for o in outs]).astype(int)
    if target.min() < 0 or target.max() > 9: return None
    ast = _enumerate(env, target, max_exprs=max_exprs)
    if ast is None: return None
    def fn(g, ast=ast):
        e = _features([g]); z = _eval_ast(ast, e)
        rp = np.where(np.isfinite(z.real), z.real, -1)
        return np.clip(np.round(rp).astype(int), 0, 9).reshape(A(g).shape)
    if all(eq(fn(i), o) for i, o in pairs): return fn
    return None

def solve(task):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    fn = regress_solver(pairs)
    if fn is None: return None
    try: return [A(fn(A(tp["input"]))) for tp in task["test"]]
    except Exception: return None


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "verify":
        print("Octonionic EML identity checks:")
        for name, val in _verify_octonionic_eml(): print("  [%s] %s" % ("OK" if val else "XX", name))
    else:
        DIR = "arc_data/"; split = cmd
        ch = json.load(open(DIR + "arc-agi_%s_challenges.json" % split))
        sol = json.load(open(DIR + "arc-agi_%s_solutions.json" % split))
        t0 = time.time(); solved = []
        for tid, task in ch.items():
            try: pr = solve(task)
            except Exception: pr = None
            if pr and all(eq(pr[i], A(g)) for i, g in enumerate(sol[tid])): solved.append(tid)
        print("EML symbolic regression (gradient-free): %d / %d  %s  (%.0fs)"
              % (len(solved), len(ch), split, time.time() - t0))
        open("/tmp/eml_%s.ids" % split, "w").write(" ".join(solved))
