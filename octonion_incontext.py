# octonion_incontext.py -- in-context transformation inference with NO predefined transforms.
#
# Perspective shift: we do not enumerate named operators (flip, rotate, ...). Instead each ARC task's
# transformation is SOLVED from its own example pairs as an operator on the octonionic field
#     Phi(out) = L_R o M o Phi(in)
#   Phi(g)  : grid -> per-cell octonion field (COLOR_OCTON[colour]); invertible (decode = nearest colour).
#   M       : a general integer AFFINE map on coordinates  p' = A p + b  -- flips, rotations, translations,
#             scalings and shears all live in ONE continuous family; nothing is hard-coded. A,b are
#             solved gradient-free by least squares from landmark-cell correspondences.
#   L_R     : an octonionic colour rotation (the per-colour relation octon R_c = o_out (x) o_in^-1),
#             solved after the geometry is fixed.
# The operator EMERGES from the examples; we then apply it to the test input and decode. Honest: a task
# is accepted only if the inferred operator reproduces every train pair EXACTLY. No gradients, no backprop.
import json, time, itertools, numpy as np
import exp_fano_layer as XF
from octonion_arc import A, eq, bg_color
from octonion_arc_phys import COLOR_OCTON

def decode_field(F):                                             # (H,W,8) -> nearest-colour grid
    return (F.reshape(-1, 8) @ COLOR_OCTON.T).argmax(1).reshape(F.shape[:2])

def _landmarks(g):
    """Cells whose colour is UNIQUE in the grid -- stable anchors that survive any colour bijection."""
    g = A(g); vals, cnts = np.unique(g, return_counts=True); bg = bg_color(g)
    pts = []
    for v, c in zip(vals, cnts):
        if c == 1 and v != bg:
            r, cc = np.argwhere(g == v)[0]; pts.append((int(r), int(cc), int(v)))
    return pts

def _fit_affine(src, dst):
    """Closed-form least-squares affine A,b with src(n,2)->dst(n,2). Returns (A,b) or None."""
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    if len(src) < 3: return None
    Xs = np.hstack([src, np.ones((len(src), 1))])               # (n,3)
    sol, *_ = np.linalg.lstsq(Xs, dst, rcond=None)              # (3,2)
    Amat = sol[:2].T; b = sol[2]
    if abs(np.linalg.det(Amat)) < 1e-6: return None
    return Amat, b

def _octo_recolour(states, outs):
    """Octonionic colour layer: per-colour relation octon R_c, applied as decode(R_c (x) o(c))."""
    if not all(s.shape == o.shape for s, o in zip(states, outs)): return None
    Rc = {}
    for s, o in zip(states, outs):
        for a, b in zip(s.ravel(), o.ravel()):
            a, b = int(a), int(b); R = XF.octo_mul(COLOR_OCTON[b], XF._inv(COLOR_OCTON[a]))
            if a in Rc and not np.allclose(Rc[a], R, atol=1e-6): return None
            Rc[a] = R
    def fn(g):
        g = A(g); out = np.empty_like(g)
        for c in np.unique(g):
            pred = XF.octo_mul(Rc[int(c)], COLOR_OCTON[int(c)]); out[g == c] = int((COLOR_OCTON @ pred).argmax())
        return out
    return fn

def _warp(g, Amat, b, oshape):
    """Apply the affine by INVERSE mapping the output lattice (no holes). Out colours = input colours."""
    g = A(g); bg = bg_color(g); H, W = g.shape; Ho, Wo = oshape
    Ai = np.linalg.inv(Amat); out = np.full((Ho, Wo), bg, int)
    rr, cc = np.mgrid[0:Ho, 0:Wo]
    P = np.stack([rr.ravel(), cc.ravel()], 1).astype(float) - b   # (Ho*Wo,2)
    S = (P @ Ai.T)                                                # source coords
    sr = np.round(S[:, 0]).astype(int); sc = np.round(S[:, 1]).astype(int)
    ok = (sr >= 0) & (sr < H) & (sc >= 0) & (sc < W)
    flat = out.ravel(); flat[ok] = g[sr[ok], sc[ok]]
    return flat.reshape(Ho, Wo)

def _oshape(pairs):
    """Output shape model: same as input, or a constant integer/rational scale -- inferred, not assumed."""
    if all(o.shape == i.shape for i, o in pairs): return lambda s: s
    ry = pairs[0][1].shape[0] / pairs[0][0].shape[0]; rx = pairs[0][1].shape[1] / pairs[0][0].shape[1]
    if all(abs(o.shape[0] - ry * i.shape[0]) < 1e-6 and abs(o.shape[1] - rx * i.shape[1]) < 1e-6 for i, o in pairs):
        return lambda s: (int(round(ry * s[0])), int(round(rx * s[1])))
    return None

def _correspondences(gi, go):
    """Per-colour cell sets shared by input and output -- the raw material for sampling correspondences
    (no colour permutation assumed here; recolour is handled afterwards by the colour layer)."""
    bgi = bg_color(gi); out = []
    for v in np.unique(gi):
        if v == bgi: continue
        ip = np.argwhere(gi == v); op = np.argwhere(go == v)
        if len(op): out.append((ip, op))
    return out

def infer(pairs, iters=300, seed=0):
    """Solve (A,b, colour layer) from the task's own pairs -- emergently. Candidate affines come from
    (i) the identity (recolour-only), (ii) landmark bijections, (iii) RANSAC over same-colour cell
    correspondences. Each candidate is accepted only if it reproduces every train pair EXACTLY."""
    osh = _oshape(pairs)
    if osh is None: return None
    outs = [o for _, o in pairs]
    def verify(Amat, b):
        warps = [_warp(i, Amat, b, osh(i.shape)) for i, _ in pairs]
        if any(w.shape != o.shape for w, o in zip(warps, outs)): return None
        rc = _octo_recolour(warps, outs)
        if rc is None or not all(eq(rc(w), o) for w, o in zip(warps, outs)): return None
        return Amat, b, osh, rc
    # (i) identity warp -- pure recolour / no motion
    r = verify(np.eye(2), np.zeros(2))
    if r is not None: return r
    gi0, go0 = pairs[0]
    # (ii) landmark bijections (when unique-colour anchors exist)
    p0in, p0out = _landmarks(gi0), _landmarks(go0)
    if len(p0in) >= 3 and len(p0in) == len(p0out):
        src0 = [(r, c) for r, c, _ in p0in]
        for perm in list(itertools.permutations(p0out, len(p0in)))[:60]:
            fit = _fit_affine(src0, [(r, c) for r, c, _ in perm])
            if fit and (r := verify(*fit)) is not None: return r
    # (iii) RANSAC over same-colour correspondences
    corr = _correspondences(gi0, go0); rng = np.random.default_rng(seed)
    if len(corr) >= 1:
        for _ in range(iters):
            src, dst = [], []
            for _ in range(3):
                ip, op = corr[int(rng.integers(len(corr)))]
                src.append(ip[int(rng.integers(len(ip)))]); dst.append(op[int(rng.integers(len(op)))])
            fit = _fit_affine(src, dst)
            if fit and (r := verify(*fit)) is not None: return r
    return None

def solve(task):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    r = infer(pairs)
    if r is None: return None
    Amat, b, osh, rc = r
    def fn(g):
        g = A(g); return A(rc(_warp(g, Amat, b, osh(g.shape))))
    return [fn(tp["input"]) for tp in task["test"]]


if __name__ == "__main__":
    import sys
    D = "arc_data/"; split = sys.argv[1] if len(sys.argv) > 1 else "training"
    ch = json.load(open(D + "arc-agi_%s_challenges.json" % split))
    sol = json.load(open(D + "arc-agi_%s_solutions.json" % split))
    t0 = time.time(); solved = []
    for tid, task in ch.items():
        try: preds = solve(task)
        except Exception: preds = None
        if preds is None: continue
        if all(eq(preds[i], A(g)) for i, g in enumerate(sol[tid])): solved.append(tid)
    print("EMERGENT in-context operator (no predefined transforms): %d / %d  %s  (%.0fs)"
          % (len(solved), len(ch), split, time.time() - t0))
    print("  solved:", " ".join(solved[:30]))
