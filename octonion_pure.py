# octonion_pure.py
# ============================================================================
# A PURE octonionic network -- no predefined transforms, no mechanism bank, no named
# operations.  The ONLY operation is octonion multiplication; the only learning signal
# is exact reproduction of the demonstrations.  Back to the original thesis, informed
# by everything since.
#
# Architecture: FANO-PATH OCTONION SUB-NETWORKS produce a hierarchy of SUB-
# REPRESENTATIONS of the grid, and the pure network maps input sub-representations to
# output ones by a single octonionic operator solved in closed form.
#
#   sub-representation s : ctx_s(cell) = unit( sum_{d in N_s} ROLE_s[d] (x) Phi(nb_d) )
#       Phi = COLOR_OCTON ;  ROLE_s = fixed Fano-point role octons ;  N_s is the
#       receptive field of sub-network s (self / von Neumann / Moore).  Each sub-
#       network is a holographic Fano-path code at its own scale.
#
#   pure operator : per input colour, the relation octon r with  o_out = r (x) ctx
#       is LINEAR in r (R(ctx) r = r (x) ctx), so r is the closed-form least-squares
#       solution -- no gradient, no named transform.  Network DEPTH = an octonionic
#       walk r2 (x) (r1 (x) ctx), fit by alternating least squares.
#
# A prediction is accepted only if the octonionic operator reproduces EVERY
# demonstration exactly; then it is applied to the test grid and decoded to the
# nearest colour.  That EXACT-verification is the sole supervision -- nothing is
# matched to a transform name.
# ============================================================================
import json, time, sys
import numpy as np
import exp_fano_layer as XF
from octonion_arc import A, eq, bg_color
from octonion_arc_phys import COLOR_OCTON

def omul(a, b): return XF.octo_mul(a, b)
_E = np.eye(8)
def Rmat_batch(X):                                   # (N,8)->(N,8,8): columns omul(e_k, X)
    return np.stack([omul(np.broadcast_to(_E[k], X.shape), X) for k in range(8)], axis=2)
def Lmat(a):                                         # (8,8): omul(a, e_k) columns
    return np.stack([omul(a, _E[k]) for k in range(8)], axis=1)

# ----- Fano-path sub-networks: role octons + receptive fields -----
_E0 = np.zeros(8); _E0[0] = 1.0
def _roles(level):
    offs = [(0, 0)]
    if level >= 1: offs += [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if level >= 2: offs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    R = {}
    for i, o in enumerate(offs):
        e = np.zeros(8); e[i % 8] = 1.0; R[o] = e        # Fano-point roles (e0 self = identity bind)
    return R, offs

def _ctx(g, level):
    g = A(g); H, W = g.shape; F = COLOR_OCTON[g]; bg = COLOR_OCTON[bg_color(g)]
    R, offs = _roles(level); acc = np.zeros((H, W, 8))
    for d in offs:
        dr, dc = d; nb = np.empty((H, W, 8)); nb[:] = bg
        r0, r1 = max(0, dr), min(H, H + dr); c0, c1 = max(0, dc), min(W, W + dc)
        nb[r0 - dr:r1 - dr, c0 - dc:c1 - dc] = F[r0:r1, c0:c1]
        acc = acc + omul(R[d], nb)
    n = np.linalg.norm(acc, axis=-1, keepdims=True)
    return acc / (n + 1e-12)

def _decode(F): return (F.reshape(-1, 8) @ COLOR_OCTON.T).argmax(1).reshape(F.shape[:-1])

def _ridge(M, Y, lam=1e-3):
    G = np.einsum("nij,nik->jk", M, M) + lam * np.eye(8); h = np.einsum("nij,ni->j", M, Y)
    return np.linalg.solve(G, h)

def _collect(pairs, level):
    if not all(i.shape == o.shape for i, o in pairs): return None
    data = {}
    for gi, go in pairs:
        ctx = _ctx(gi, level); Fo = COLOR_OCTON[A(go)]; gi = A(gi)
        for c in np.unique(gi):
            m = gi == c; X = ctx[m].reshape(-1, 8); Y = Fo[m].reshape(-1, 8)
            if c in data: data[c] = (np.vstack([data[c][0], X]), np.vstack([data[c][1], Y]))
            else: data[c] = (X, Y)
    return data

def _solve1(data): return {c: _ridge(Rmat_batch(X), Y) for c, (X, Y) in data.items()}

def _solve2(data, iters=6):                          # octonionic depth: r2 (x) (r1 (x) ctx), ALS
    out = {}
    for c, (X, Y) in data.items():
        Rc = Rmat_batch(X); r1 = _ridge(Rc, Y); r2 = _E0.copy()
        for _ in range(iters):
            mid = omul(np.broadcast_to(r1, X.shape), X); r2 = _ridge(Rmat_batch(mid), Y)
            M = np.einsum("ij,njk->nik", Lmat(r2), Rc); r1 = _ridge(M, Y)
        out[c] = (r1, r2)
    return out

def _apply(g, paths, level):
    g = A(g); ctx = _ctx(g, level); out = np.empty(g.shape + (8,))
    for c in np.unique(g):
        m = g == c; v = ctx[m]
        p = paths.get(c, COLOR_OCTON[int(c)]); chain = p if isinstance(p, tuple) else (p,)
        for r in chain: v = omul(np.broadcast_to(r, v.shape), v)
        out[m] = v
    return _decode(out)

def solve(task):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    tests = [A(tp["input"]) for tp in task["test"]]
    for level in (0, 1, 2):                           # the Fano sub-networks (scales)
        data = _collect(pairs, level)
        if data is None: continue
        for paths in (_solve1(data), _solve2(data)):  # octonionic depth 1 then 2
            if all(eq(_apply(i, paths, level), o) for i, o in pairs):
                try: return [A(_apply(t, paths, level)) for t in tests]
                except Exception: pass
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
    print("PURE octonionic network (Fano sub-reps, no predefined transforms): %d / %d  %s  (%.0fs)"
          % (len(solved), len(ch), split, time.time() - t0))
    open("/tmp/pure_%s.ids" % split, "w").write(" ".join(solved))
