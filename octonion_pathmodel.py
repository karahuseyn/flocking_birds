# octonion_pathmodel.py
# ============================================================================
# An octonionic PATH-TRANSFORM model.  The thesis (user's): picture one large
# octonionic space in which any object -- a pixel, a whole pixel-object, a token,
# a sentence -- is representable, and in which a FANO PATH runs from every object
# to every other.  A path does not store a relation explicitly; it IMPLIES it
# (relational / contextual / correlational), so the bond is encoded INDIRECTLY.
# Training a base "universe of paths" on a huge stream of synthetic object->object
# transforms then gives, for any ARC task, a way to read the path off the train
# pairs and push the test input through it -- fast, gradient-free, no backprop.
#
# Concretely, three pieces of (deliberately advanced) mathematics:
#
#  (1) FANO-ROLE BINDING  (a vector-symbolic / holographic code over octonions).
#      A cell's CONTEXT octon bundles its neighbourhood by octonion-multiplying
#      each neighbour's colour octon with a fixed Fano-point role octon (an
#      imaginary unit e_k) and superposing:  ctx(p) = unit( sum_d  R_d (x) Phi(p+d) ).
#      The bond between centre and surround is carried indirectly, in the
#      interference pattern of the bundle -- not as an explicit rule.
#
#  (2) CLOSED-FORM OCTONIONIC OPERATOR REGRESSION  (the path itself).
#      A path is a single relation octon r with  o_out = r (x) ctx.  Octonion
#      product is LINEAR in r through the right-multiplication matrix R(ctx)
#      (R(ctx) @ r == r (x) ctx), so from the task's own cells we SOLVE
#          r* = argmin_r  sum_i || R(ctx_i) r - o_out_i ||^2
#      by the normal equations (Tikhonov-regularised) -- 8 unknowns, closed form,
#      no gradients.  r* IS the Fano path; we apply it to the test grid and decode
#      to the nearest colour, accepting only on EXACT reproduction of every demo.
#
#  (3) A SYNTHETIC BASE UNIVERSE OF PATHS  (the "big transform model").
#      A small gradient-free codebook of path atoms r, harvested by online
#      competitive clustering from a broad stream of un-named parametric
#      object->object transforms.  It supplies a PRIOR (regularising the in-context
#      regression toward a known path) and CLEAN-UP (snapping a noisy in-context
#      path onto the nearest learned atom) -- this is where the base model can
#      generalise a path to a context the task's train pairs never showed.
#
# Everything is gradient-free / no backprop.  The model is small (a few hundred
# path atoms, 8-D each) -- much smaller than the 2048-node paramnet.
# ============================================================================
import os, sys, time, json
import numpy as np
from octonion_arc import A, eq, bg_color, DIHEDRAL
from octonion_arc_phys import COLOR_OCTON
import exp_fano_layer as XF
import octonion_incontext as IC

DIH = list(DIHEDRAL.values())

# ------------------------------------------------------------------ octonion ops
def omul(a, b): return XF.octo_mul(a, b)

_E = np.eye(8)
def Rmat(x):                                   # right-mult matrix: Rmat(x) @ r == r (x) x
    return np.stack([omul(_E[k], x) for k in range(8)], axis=1)

def Rmat_batch(X):                             # (N,8) -> (N,8,8), columns omul(e_k, X)
    return np.stack([omul(np.broadcast_to(_E[k], X.shape), X) for k in range(8)], axis=2)

def Lmat_batch(a):                             # (8,) -> (8,8), Lmat(a) @ x == a (x) x ; columns omul(a, e_k)
    A8 = np.broadcast_to(a, (8, 8))
    return np.stack([omul(A8[k], _E[k]) for k in range(8)], axis=1)

# Fano-point role octons for neighbourhood offsets.  Self gets the real unit e0
# (binding by e0 is the identity); each neighbour gets a distinct imaginary unit
# e1..e7 -- the seven Fano points -- so the bundle is a holographic 7-slot code.
def _roles(level):
    offs = [(0, 0)]
    if level >= 1: offs += [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if level >= 2: offs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    R = {}
    for i, o in enumerate(offs):
        e = np.zeros(8); e[i % 8] = 1.0; R[o] = e
    return R, offs

def _field(g):                                 # grid -> (H,W,8) octonion field
    return COLOR_OCTON[A(g)]

def _context(g, level):
    """Per-cell context octons via Fano-role binding of the neighbourhood (with the
    grid's own background octon padded in for out-of-grid neighbours)."""
    g = A(g); F = _field(g); H, W = g.shape; bg = COLOR_OCTON[bg_color(g)]
    R, offs = _roles(level)
    acc = np.zeros((H, W, 8))
    for d in offs:
        dr, dc = d
        nb = np.empty((H, W, 8)); nb[:] = bg
        r0, r1 = max(0, dr), min(H, H + dr); c0, c1 = max(0, dc), min(W, W + dc)
        nb[r0 - dr:r1 - dr, c0 - dc:c1 - dc] = F[r0:r1, c0:c1]
        acc += omul(R[d], nb)                   # bind neighbour to its Fano role, superpose
    n = np.linalg.norm(acc, axis=-1, keepdims=True)
    return acc / (n + 1e-12)

def decode(F):                                 # (.,8) -> nearest colour
    return (F.reshape(-1, 8) @ COLOR_OCTON.T).argmax(1).reshape(F.shape[:-1])

# ----------------------------------------------- closed-form per-colour path solve
def _solve_paths(pairs, level, prior=None, lam=1e-3):
    """For each input colour c, solve the relation octon r_c (the Fano path) that
    best maps the Fano-role context to the output octon, closed-form via the
    normal equations.  prior (codebook) optionally Tikhonov-pulls r_c toward a
    known atom.  Returns {c: r_c} or None if shapes mismatch."""
    if not all(i.shape == o.shape for i, o in pairs): return None
    G = {}; h = {}
    for gi, go in pairs:
        ctx = _context(gi, level); Fout = _field(go)
        gi = A(gi)
        for c in np.unique(gi):
            m = gi == c
            X = ctx[m].reshape(-1, 8); Y = Fout[m].reshape(-1, 8)
            Rb = Rmat_batch(X)                                  # (N,8,8)
            Gc = np.einsum("nij,nik->jk", Rb, Rb)               # sum R^T R
            hc = np.einsum("nij,ni->j", Rb, Y)                  # sum R^T y
            G[c] = G.get(c, 0) + Gc; h[c] = h.get(c, 0) + hc
    paths = {}
    for c in G:
        Gc = G[c] + lam * np.eye(8); hc = h[c]
        if prior is not None and c in prior:   # pull toward the codebook atom
            Gc = Gc + lam * np.eye(8); hc = hc + lam * prior[c]
        paths[c] = np.linalg.solve(Gc, hc)
    return paths

def _collect(pairs, level):
    """Gather per-input-colour (context, output) octon stacks across all train pairs."""
    if not all(i.shape == o.shape for i, o in pairs): return None
    data = {}
    for gi, go in pairs:
        ctx = _context(gi, level); Fout = _field(go); gi = A(gi)
        for c in np.unique(gi):
            m = gi == c
            X = ctx[m].reshape(-1, 8); Y = Fout[m].reshape(-1, 8)
            if c in data: data[c] = (np.vstack([data[c][0], X]), np.vstack([data[c][1], Y]))
            else: data[c] = (X, Y)
    return data

def _ridge(M, Y, lam, prior=None):             # (N,8,8),(N,8) -> r  via normal equations
    G = np.einsum("nij,nik->jk", M, M) + lam * np.eye(8)
    h = np.einsum("nij,ni->j", M, Y)
    if prior is not None: G = G + lam * np.eye(8); h = h + lam * prior
    return np.linalg.solve(G, h)

def _solve1(data, lam=1e-3, prior=None):       # one relation octon r:  o_out = r (x) ctx
    return {c: _ridge(Rmat_batch(X), Y, lam, None if prior is None else prior.get(c))
            for c, (X, Y) in data.items()}

# ------------------------------------------ octonionic KALMAN / RLS path estimator
# The relation octon r is a hidden state with the linear measurement model
#     o_out_i = H_i r + noise,   H_i = R(ctx_i)   (the right-multiplication matrix).
# A Kalman filter is the recursive-Bayes form of the batch ridge solve, and it adds
# two things ridge cannot give: (i) the POSTERIOR COVARIANCE P -- how well the data
# pins the path, a principled overfit gate; (ii) optimal FUSION of a base-universe
# prior, pulling r toward the learned atom ONLY in directions the data leaves
# unconstrained (large P), never disturbing the data-constrained ones.  Gradient-free.
def _kalman_path(X, Y, r0, P0, rmeas=2e-2, q=0.0, cap=400):
    if len(X) > cap:                                          # subsample cells: r is 8-D, a few hundred pin it
        idx = np.linspace(0, len(X) - 1, cap).astype(int); X, Y = X[idx], Y[idx]
    r = r0.astype(float).copy(); P = P0.astype(float).copy(); I = np.eye(8); Rm = rmeas * I
    for i in range(len(X)):
        if q: P = P + q * I                                   # process noise -> tracks a drifting path
        H = Rmat(X[i])
        S = H @ P @ H.T + Rm
        K = P @ H.T @ np.linalg.inv(S)                        # Kalman gain
        r = r + K @ (Y[i] - H @ r)
        P = (I - K @ H) @ P
    return r, P

def _solve_kalman(data, uni=None, p0=10.0, rmeas=2e-2, q=0.0):
    """Per-colour Kalman estimate of the path, prior-initialised from the base universe
    (Bayesian fusion).  Returns {c:r} plus {c:trace(P)} as an uncertainty read-out."""
    paths = {}; unc = {}
    for c, (X, Y) in data.items():
        r0 = np.zeros(8); r0[0] = 1.0                          # default prior: identity-ish path
        if uni is not None:                                   # warm-start from nearest learned atom
            rb = _ridge(Rmat_batch(X), Y, 1e-2)
            rb = rb / (np.linalg.norm(rb) + 1e-12)
            r0 = uni.atoms[int((uni.atoms @ rb).argmax())]
        r, P = _kalman_path(X, Y, r0, p0 * np.eye(8), rmeas=rmeas, q=q)
        paths[c] = r; unc[c] = float(np.trace(P))
    return paths, unc

E0 = np.zeros(8); E0[0] = 1.0
def _solve2(data, iters=6, lam=1e-3):
    """Two-step Fano walk  o_out = r2 (x) (r1 (x) ctx), fit by ALTERNATING LEAST SQUARES.
    Each half-step is closed-form (the product is bilinear in (r1,r2)); no gradients.
    r1 (x) ctx = R(ctx) r1 ;  r2 (x) (r1(x)ctx) = L(r2) R(ctx) r1 (linear in r1),
                                                = R(r1(x)ctx) r2 (linear in r2)."""
    out = {}
    for c, (X, Y) in data.items():
        Rc = Rmat_batch(X)                                  # (N,8,8): R(ctx)
        r1 = _ridge(Rc, Y, lam); r2 = E0.copy()
        for _ in range(iters):
            mid = omul(np.broadcast_to(r1, X.shape), X)                # mid = r1 (x) ctx
            r2 = _ridge(Rmat_batch(mid), Y, lam)                        # solve r2 | r1
            Lr2 = Lmat_batch(r2)                                        # (8,8)
            M = np.einsum("ij,njk->nik", Lr2, Rc)                       # L(r2) R(ctx), (N,8,8)
            r1 = _ridge(M, Y, lam)                                      # solve r1 | r2
        out[c] = (r1, r2)
    return out

def _step(r, ctx):                              # r (x) ctx, broadcast over a cell stack
    return omul(np.broadcast_to(r, ctx.shape), ctx)

def _apply(g, paths, level, fallback=None):
    """paths[c] is either a single octon r (1-step) or a tuple (r1,..,rk) walked
    inside-out: o = rk (x) (... (x) (r1 (x) ctx)).  Decode to nearest colour."""
    g = A(g); ctx = _context(g, level); H, W = g.shape
    out = np.empty((H, W, 8))
    for c in np.unique(g):
        m = g == c; v = ctx[m]
        p = paths.get(c, fallback if fallback is not None else COLOR_OCTON[int(c)])
        chain = p if isinstance(p, tuple) else (p,)
        for r in chain: v = _step(r, v)
        out[m] = v
    return decode(out)

def _path_solver(pairs, prior=None, uni=None):
    """Search context levels (recolour -> local -> wider local) and estimator
    (1-step ridge, 2-step ALS Fano walk, then the octonionic KALMAN path with
    base-universe prior fusion); accept the first path that reproduces every
    demonstration EXACTLY."""
    for level in (0, 1, 2):
        data = _collect(pairs, level)
        if data is None: continue
        cands = [_solve1(data, prior=prior), _solve2(data)]
        if uni is not None:
            cands.append(_solve_kalman(data, uni)[0])             # Kalman, prior-fused, q=0 (RLS)
            cands.append(_solve_kalman(data, uni, q=1e-3)[0])     # Kalman with drift (tracks varying r)
        for paths in cands:
            if all(eq(_apply(i, paths, level), o) for i, o in pairs):
                return lambda g, P=paths, L=level: _apply(g, P, L)
    return None

# ------------------------------------------- synthetic base universe of paths
def _rand_grid(rng):
    h, w = int(rng.integers(2, 12)), int(rng.integers(2, 12))
    k = int(rng.integers(2, 6))
    return rng.integers(0, k, size=(h, w))

def _rand_transform(rng, g):
    """A broad, UN-NAMED parametric object->object transform (shape-preserving so it
    yields a value path): random colour bijection o random local cellular rule."""
    g = A(g)
    rule = rng.integers(0, g.max() + 1 if g.max() > 0 else 1, size=(int(g.max()) + 1, 5))
    P = np.pad(g, 1)
    nz = sum((P[1 + dr:1 + dr + g.shape[0], 1 + dc:1 + dc + g.shape[1]] > 0).astype(int)
             for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
    g2 = rule[g, np.minimum(nz, 4)]
    perm = np.arange(10); perm[1:] = rng.permutation(perm[1:])
    return perm[g2]

class PathUniverse:
    """Small gradient-free codebook of path atoms r (the base transform universe),
    grown by online competitive clustering of the relation octons that the synthetic
    object->object transforms induce."""
    def __init__(self, n=256, seed=0):
        rng = np.random.default_rng(seed)
        self.atoms = XF.unit(rng.standard_normal((n, 8))); self.cnt = np.zeros(n); self.n = n
    def _commit(self, r):
        r = r / (np.linalg.norm(r) + 1e-12)
        w = int((self.atoms @ r).argmax())
        self.cnt[w] += 1; self.atoms[w] = XF.unit(self.atoms[w] + (r - self.atoms[w]) / self.cnt[w])
    def train(self, n, seed=1, log_every=200000):
        rng = np.random.default_rng(seed); t0 = time.time()
        for done in range(0, n, 1):
            gi = _rand_grid(rng); go = _rand_transform(rng, gi)
            paths = _solve_paths([(gi, go)], level=int(rng.integers(0, 3)))
            if paths:
                for r in paths.values(): self._commit(r)
            if (done + 1) % log_every == 0:
                print("  paths %d/%d (%.0fs, %d atoms used)" %
                      (done + 1, n, time.time() - t0, int((self.cnt > 0).sum())), flush=True)
    def prior_for(self, paths):
        """Clean-up: snap each solved path onto its nearest learned atom."""
        out = {}
        for c, r in paths.items():
            rn = r / (np.linalg.norm(r) + 1e-12)
            out[c] = self.atoms[int((self.atoms @ rn).argmax())] * np.linalg.norm(r)
        return out
    def save(self, p): np.savez_compressed(p, atoms=self.atoms, cnt=self.cnt)
    @classmethod
    def load(cls, p):
        d = np.load(p); o = cls.__new__(cls)
        o.atoms, o.cnt = d["atoms"], d["cnt"]; o.n = len(o.atoms); return o

# ----------------------- discrete grid-operator: VQ octonionic-context cellular rule
def _symbols(g, level, uni):
    """Vector-quantize each cell's Fano-context octon against the learned base
    universe -> a discrete symbol (the path-universe 'word' for that local context).
    The relational bond is carried indirectly by WHICH atom the context lands on."""
    ctx = _context(g, level).reshape(-1, 8)
    return (ctx @ uni.atoms.T).argmax(1).reshape(A(g).shape)

def _sym_rule_solver(pairs, uni, levels=(1, 2)):
    """Fit a discrete cellular operator keyed by (centre colour, context symbol) ->
    output colour, then by symbol alone.  Un-named (the key is a learned octonionic
    codeword, the table is fit from the task), gradient-free, accepted only on EXACT
    reproduction of every demonstration."""
    if uni is None or not all(i.shape == o.shape for i, o in pairs): return None
    P = uni.n
    for level in levels:
        for keymode in ("c", "cs", "s"):
            table = {}; ok = True
            for gi, go in pairs:
                S = _symbols(gi, level, uni); gi_ = A(gi).ravel(); go_ = A(go).ravel()
                k = gi_ if keymode == "c" else (gi_ * P + S.ravel() if keymode == "cs" else S.ravel())
                for kk, o in zip(k, go_):
                    kk = int(kk)
                    if table.get(kk, int(o)) != int(o): ok = False; break
                    table[kk] = int(o)
                if not ok: break
            if not ok: continue
            def fn(g, table=table, level=level, keymode=keymode):
                g = A(g); S = _symbols(g, level, uni)
                k = g.ravel() if keymode == "c" else ((g.ravel() * P + S.ravel()) if keymode == "cs" else S.ravel())
                out = np.array([table.get(int(kk), int(c)) for kk, c in zip(k, g.ravel())])
                return out.reshape(g.shape)
            if all(eq(fn(i), o) for i, o in pairs): return fn
    return None

# ------------------------------------------- discrete walk through the path universe
def _apply_atom(g, r, level):
    """One discrete path step: push the grid through atom r at a context level and
    DECODE to the nearest colour.  The decode is the discretisation that stops the
    continuous overfit -- each step lands back on a real grid."""
    g = A(g); ctx = _context(g, level)
    return decode(_step(r, ctx.reshape(-1, 8)).reshape(g.shape + (8,)))

def _match(a, b):
    return float((A(a) == A(b)).mean()) if a.shape == b.shape else -1.0

def _walk_solver(pairs, uni, depth=3, beam=8, topk=48, max_expand=700):
    """Beam search a SHARED discrete program (sequence of learned atoms+levels) that
    reproduces every demonstration EXACTLY.  The alphabet is the synthetic base
    universe -- task-independent, learned, un-named -- so the only thing fitted per
    task is the WALK (a discrete object), not a continuous operator.  Gradient-free."""
    if uni is None or not all(i.shape == o.shape for i, o in pairs): return None
    order = np.argsort(-uni.cnt)[:topk]                       # most-used atoms
    actions = [(uni.atoms[a], L) for a in order for L in (0, 1)]
    ins = [i for i, _ in pairs]; outs = [o for _, o in pairs]
    def score(state): return float(np.mean([_match(s, o) for s, o in zip(state, outs)]))
    start = tuple(ins)
    beam_states = [(score(start), start, [])]; seen = {tuple(map(lambda x: x.tobytes(), start))}
    expanded = 0
    for _ in range(depth):
        cand = []
        for sc, state, prog in beam_states:
            if sc >= 1.0: continue
            for ai, (r, L) in enumerate(actions):
                if expanded >= max_expand: break
                try: ns = tuple(_apply_atom(s, r, L) for s in state)
                except Exception: continue
                expanded += 1
                key = tuple(x.tobytes() for x in ns)
                if key in seen: continue
                seen.add(key); s2 = score(ns)
                if s2 >= 1.0:                                  # exact on all demos -> program found
                    return [r2 for r2 in prog] + [(r, L)]
                cand.append((s2, ns, prog + [(r, L)]))
            if expanded >= max_expand: break
        if not cand: break
        cand.sort(key=lambda t: -t[0]); beam_states = cand[:beam]
    return None

def _run_walk(g, program):
    for r, L in program: g = _apply_atom(g, r, L)
    return A(g)

# ------------------------------------------------------------------- full solve
def solve(task, uni=None):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    tests = [A(tp["input"]) for tp in task["test"]]
    # (A) value path on raw pairs (ridge / ALS / Kalman with base-universe prior fusion)
    fn = _path_solver(pairs, uni=uni)
    if fn is None and uni is not None:          # base universe as clean-up prior
        p0 = _solve_paths(pairs, 1)
        if p0: fn = _path_solver(pairs, prior=uni.prior_for(p0), uni=uni)
    if fn is not None:
        try: return [A(fn(t)) for t in tests]
        except Exception: pass
    # (B) coordinate path (emergent octonionic affine + Fano colour), then value path
    r = IC.infer(pairs, iters=120)
    if r is not None:
        Amat, b, osh, rc = r
        warp = lambda g: A(rc(IC._warp(A(g), Amat, b, osh(A(g).shape))))
        if all(eq(warp(i), o) for i, o in pairs):
            try: return [A(warp(t)) for t in tests]
            except Exception: pass
        mids = [warp(i) for i, _ in pairs]
        if all(m.shape == o.shape for m, (_, o) in zip(mids, pairs)):
            fn2 = _path_solver(list(zip(mids, [o for _, o in pairs])))
            if fn2 is not None:
                try: return [A(fn2(warp(t))) for t in tests]
                except Exception: pass
    # (D) discrete grid-operator: VQ octonionic-context cellular rule (learned, un-named)
    fn = _sym_rule_solver(pairs, uni)
    if fn is not None:
        try: return [A(fn(t)) for t in tests]
        except Exception: pass
    # (C) discrete walk through the learned path universe (shared program, beam search)
    prog = None if os.environ.get("NOWALK") == "1" else _walk_solver(pairs, uni)
    if prog is not None:
        try: return [_run_walk(t, prog) for t in tests]
        except Exception: pass
    return None


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    P = "octonion_pathmodel.npz"
    if cmd == "train":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000000
        uni = PathUniverse(n=int(os.environ.get("ATOMS", "256")))
        print("PATH universe: %d atoms, %d synthetic object->object transforms" % (uni.n, N), flush=True)
        uni.train(N); uni.save(P); print("saved", P, flush=True)
    else:
        uni = PathUniverse.load(P) if os.path.exists(P) else None
        for split in ("training", "evaluation"):
            ch = json.load(open("arc_data/arc-agi_%s_challenges.json" % split))
            sol = json.load(open("arc_data/arc-agi_%s_solutions.json" % split))
            t0 = time.time(); s = []
            for tid, task in ch.items():
                try: pr = solve(task, uni)
                except Exception: pr = None
                if pr and all(eq(pr[i], A(g)) for i, g in enumerate(sol[tid])): s.append(tid)
            print("pathmodel ARC %s: %d/%d (%.0fs)" % (split, len(s), len(ch), time.time() - t0), flush=True)
            open("/tmp/pathmodel_%s.ids" % split, "w").write(" ".join(s))
