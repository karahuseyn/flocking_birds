# octonion_otrm.py
# ============================================================================
# Octonionic TRM (O-TRM): a TRM-style recursive refiner on the PURE octonionic
# representation, with gradients confined to the final readout -- as in TRM, where
# only the last stage carries gradients.  This is the first time this project uses a
# gradient at all; it is added deliberately and minimally to see how the octonionic
# system adapts to one.
#
#   * RECURSIVE LATENT (gradient-free).  Each cell starts at its Fano-context octon
#     h0 = ctx(cell) and is refined by T steps of octonionic message passing
#         h <- unit( h + a * sum_d ROLE[d] (x) shift_d(h) ),
#     propagating information across the grid -- the octonionic analogue of TRM's
#     latent recursion (one shared rule applied repeatedly to a fixed point).
#   * GRADIENT READOUT (1 hidden layer = 2 weight layers).  A tiny MLP maps the
#     refined per-cell octonionic features to a colour, trained by back-propagation
#     (manual Adam) on the task's own cells -- test-time training, per task.
#   * Accept only if the trained readout reproduces EVERY training cell exactly, then
#     predict the test grid.  Shape-preserving tasks.
#
# So the recursion stays pure/gradient-free octonionic; the gradient touches only the
# 2-layer head.  We report both how often the head FITS the train cells (gradient
# capacity) and how often it GENERALISES to the test (the solve count).
# ============================================================================
import json, time, sys
import numpy as np
import exp_fano_layer as XF
from octonion_arc import A, eq, bg_color
from octonion_arc_phys import COLOR_OCTON

def omul(a, b): return XF.octo_mul(a, b)

def _roles(level):
    offs = [(0, 0)]
    if level >= 1: offs += [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if level >= 2: offs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    R = {}
    for i, o in enumerate(offs):
        e = np.zeros(8); e[i % 8] = 1.0; R[o] = e
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

def _recur(h, T=4, a=0.5):
    """Gradient-free octonionic message passing to a (near) fixed point."""
    R, offs = _roles(2); H, W, _ = h.shape
    for _ in range(T):
        acc = h.copy()
        for d in offs:
            dr, dc = d
            nb = np.zeros((H, W, 8))
            r0, r1 = max(0, dr), min(H, H + dr); c0, c1 = max(0, dc), min(W, W + dc)
            nb[r0 - dr:r1 - dr, c0 - dc:c1 - dc] = h[r0:r1, c0:c1]
            acc = acc + a * omul(R[d], nb)
        n = np.linalg.norm(acc, axis=-1, keepdims=True); h = acc / (n + 1e-12)
    return h

def _feats(g, T=4):
    g = A(g); c0 = _ctx(g, 0); c1 = _ctx(g, 1); c2 = _ctx(g, 2); hT = _recur(c1.copy(), T)
    H, W = g.shape; onehot = np.eye(10)[g]                       # cell colour
    X = np.concatenate([onehot, c0, c1, c2, hT], axis=-1)        # 10+8+8+8+8 = 42 per cell
    return X.reshape(-1, X.shape[-1])

# ----- tiny 2-layer MLP head, manual Adam (the only gradients in the project) -----
def _train_head(X, y, hid=48, steps=500, lr=5e-3, seed=0):
    rng = np.random.default_rng(seed); N, F = X.shape
    W1 = rng.standard_normal((F, hid)) * (1.0 / np.sqrt(F)); b1 = np.zeros(hid)
    W2 = rng.standard_normal((hid, 10)) * (1.0 / np.sqrt(hid)); b2 = np.zeros(10)
    params = [W1, b1, W2, b2]; m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    b1m, b2m, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        Z1 = X @ W1 + b1; H1 = np.maximum(Z1, 0); Zo = H1 @ W2 + b2
        Zo -= Zo.max(1, keepdims=True); E = np.exp(Zo); P = E / E.sum(1, keepdims=True)
        G = P.copy(); G[np.arange(N), y] -= 1; G /= N                # dL/dZo
        gW2 = H1.T @ G; gb2 = G.sum(0); gH1 = G @ W2.T; gZ1 = gH1 * (Z1 > 0)
        gW1 = X.T @ gZ1; gb1 = gZ1.sum(0); grads = [gW1, gb1, gW2, gb2]
        for i, gr in enumerate(grads):
            m[i] = b1m * m[i] + (1 - b1m) * gr; v[i] = b2m * v[i] + (1 - b2m) * gr * gr
            mh = m[i] / (1 - b1m ** t); vh = v[i] / (1 - b2m ** t)
            params[i] -= lr * mh / (np.sqrt(vh) + eps)
        W1, b1, W2, b2 = params
    return params

def _predict(X, params):
    W1, b1, W2, b2 = params
    H1 = np.maximum(X @ W1 + b1, 0); Zo = H1 @ W2 + b2
    return Zo.argmax(1)

def solve(task, T=4, want_fit=False):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    tests = [A(tp["input"]) for tp in task["test"]]
    if not all(i.shape == o.shape for i, o in pairs): return (None, False) if want_fit else None
    Xs = [_feats(i, T) for i, _ in pairs]; ys = [A(o).reshape(-1) for _, o in pairs]
    X = np.concatenate(Xs); y = np.concatenate(ys).astype(int)
    params = _train_head(X, y)
    fit = bool((_predict(X, params) == y).all())                 # did gradients fit the train cells?
    if not fit:
        return (None, False) if want_fit else None
    preds = []
    for t in tests:
        Xt = _feats(t, T); pr = _predict(Xt, params).reshape(A(t).shape); preds.append(A(pr))
    return (preds, True) if want_fit else preds


if __name__ == "__main__":
    DIR = "arc_data/"; split = sys.argv[1] if len(sys.argv) > 1 else "training"
    ch = json.load(open(DIR + "arc-agi_%s_challenges.json" % split))
    sol = json.load(open(DIR + "arc-agi_%s_solutions.json" % split))
    t0 = time.time(); solved = []; fit_count = 0; n = 0
    for tid, task in ch.items():
        n += 1
        try: res = solve(task, want_fit=True)
        except Exception: res = (None, False)
        pr, fit = res if isinstance(res, tuple) else (res, False)
        if fit: fit_count += 1
        if pr and all(eq(pr[i], A(g)) for i, g in enumerate(sol[tid])): solved.append(tid)
    print("O-TRM (octonionic recursion + 2-layer gradient head): %d / %d  %s  (%.0fs)  | train-fit %d"
          % (len(solved), len(ch), split, time.time() - t0, fit_count))
    open("/tmp/otrm_%s.ids" % split, "w").write(" ".join(solved))
