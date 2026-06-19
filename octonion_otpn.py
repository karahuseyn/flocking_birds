# octonion_otpn.py
# ============================================================================
# Octonionic Target-Propagation Network (OTPN): a multi-layer octonionic network that
# LEARNS like back-propagation -- forward pass, then a BACKWARD signal through the
# layers, then per-layer weight updates -- but with ZERO gradients.  The backward
# signal is a TARGET, propagated through each layer by the octonion algebra's INVERSE
# (octonions are a division algebra, so every layer is invertible), and each layer's
# weights are fit in CLOSED FORM (octonionic least squares).  No gradient, no backprop,
# no autodiff -- credit assignment is algebraic.
#
#   forward :  y_j = unit( sum_i W1[j,i] (x) x_i ) ;  z = unit( sum_i W2[i] (x) y_i )
#   target  :  t_z = COLOR_OCTON[target] ;  back-propagate  y*_i = unit( W2[i]^{-1} (x) t_z )
#   update  :  solve W2 from (y -> t_z) and W1[j] from (x -> y*_j), each a closed-form
#              octonionic least-squares solve  (o = W (x) v  is linear in W via R(v)).
#   iterate the forward/target/solve passes -- the loss ||z - t_z|| falls each pass.
#
# Read-out decodes z to the nearest colour octon.  Per task it is trained by target
# propagation on the cells; accepted only under the held-out-pair + ensemble gate
# (over weight-init seeds) that eliminated over-fit for the gradient head.
# ============================================================================
import json, time, sys
import numpy as np
import exp_fano_layer as XF
from octonion_arc import A, eq, bg_color
from octonion_arc_phys import COLOR_OCTON
from octonion_otrm import _ctx, _recur

def omul(a, b): return XF.octo_mul(a, b)
def oinv(a): return XF._inv(a)
_E = np.eye(8)
def Rb(X):                                            # (N,8,8): Rb(X)[n] @ w == omul(w, X[n])
    return np.stack([omul(np.broadcast_to(_E[k], X.shape), X) for k in range(8)], axis=2)

def _channels(g):
    """Per-cell input as C=4 octonion channels (the Fano sub-representations)."""
    g = A(g); c0 = _ctx(g, 0); c1 = _ctx(g, 1); c2 = _ctx(g, 2); hT = _recur(c1.copy(), 4)
    X = np.stack([c0, c1, c2, hT], axis=2)            # (H,W,4,8)
    return X.reshape(-1, 4, 8)

def _solve_layer(Xc, T, lam=1e-2):
    """Solve W (C,8) so that sum_i W[i] (x) Xc[:,i] ~= T, closed form (octonionic LS)."""
    N, C, _ = Xc.shape
    M = np.concatenate([Rb(Xc[:, i]) for i in range(C)], axis=2)   # (N,8,8C)
    G = np.einsum("nij,nik->jk", M, M) + lam * np.eye(8 * C)
    h = np.einsum("nij,ni->j", M, T)
    w = np.linalg.solve(G, h)
    return w.reshape(C, 8)

def _fwd1(Xc, W1):                                    # (N,C,8),(C,C,8) -> Y (N,C,8)
    N, C, _ = Xc.shape; Y = np.empty((N, C, 8))
    for j in range(C):
        acc = sum(omul(np.broadcast_to(W1[j, i], Xc[:, i].shape), Xc[:, i]) for i in range(C))
        Y[:, j] = acc / (np.linalg.norm(acc, axis=1, keepdims=True) + 1e-9)
    return Y

def _fwd2(Y, W2):
    C = Y.shape[1]
    acc = sum(omul(np.broadcast_to(W2[i], Y[:, i].shape), Y[:, i]) for i in range(C))
    return acc / (np.linalg.norm(acc, axis=1, keepdims=True) + 1e-9)

class OTPN:
    def __init__(self, C=4, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = XF.unit(rng.standard_normal((C, C, 8))); self.W2 = XF.unit(rng.standard_normal((C, 8))); self.C = C
    def train(self, Xc, tz, passes=8):
        loss = []
        for _ in range(passes):
            Y = _fwd1(Xc, self.W1)                                  # forward
            self.W2 = _solve_layer(Y, tz)                          # solve top layer  Y -> t_z
            z = _fwd2(Y, self.W2); loss.append(float(np.mean(np.linalg.norm(z - tz, axis=1))))
            yt = np.stack([XF.unit(omul(np.broadcast_to(oinv(self.W2[i]), tz.shape), tz))
                           for i in range(self.C)], axis=1)         # target-prop via inverse
            for j in range(self.C):                                # solve bottom layer  x -> y*_j
                self.W1[j] = _solve_layer(Xc, yt[:, j])
        return loss
    def predict(self, Xc):
        z = _fwd2(_fwd1(Xc, self.W1), self.W2)
        return (z @ COLOR_OCTON.T).argmax(1)

def _fit(Xc, y, seed=0, passes=8):
    tz = COLOR_OCTON[y]                                            # per-cell target octon
    net = OTPN(seed=seed); net.train(Xc, tz, passes); return net

def solve(task):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    tests = [A(tp["input"]) for tp in task["test"]]
    if len(pairs) < 2 or not all(i.shape == o.shape for i, o in pairs): return None
    Xs = [_channels(i) for i, _ in pairs]; ys = [A(o).reshape(-1).astype(int) for _, o in pairs]
    Xc = np.concatenate(Xs); y = np.concatenate(ys); K = 2; tX = [_channels(t) for t in tests]
    hj = len(pairs) - 1; idx = list(range(hj))
    Xtr = np.concatenate([Xs[k] for k in idx]); ytr = np.concatenate([ys[k] for k in idx])
    for s in range(K):                                            # held-out-pair generalisation gate
        net = _fit(Xtr, ytr, seed=s)
        if not np.array_equal(net.predict(Xs[hj]), ys[hj]): return None
    preds_k = []
    for s in range(K):
        net = _fit(Xc, y, seed=s)
        if not np.array_equal(net.predict(Xc), y): return None     # must reproduce all train cells
        preds_k.append([net.predict(t) for t in tX])
    for ti in range(len(tests)):
        for s in range(1, K):
            if not np.array_equal(preds_k[s][ti], preds_k[0][ti]): return None
    return [A(preds_k[0][ti].reshape(A(tests[ti]).shape)) for ti in range(len(tests))]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "training"
    if cmd == "learn":                                            # demonstrate it LEARNS (loss falls)
        import numpy as np
        ch = json.load(open("arc_data/arc-agi_training_challenges.json"))
        task = ch["25ff71a9"]; pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
        Xc = np.concatenate([_channels(i) for i, _ in pairs]); y = np.concatenate([A(o).reshape(-1) for _, o in pairs]).astype(int)
        net = OTPN(seed=0); loss = net.train(Xc, COLOR_OCTON[y], passes=12)
        acc = float((net.predict(Xc) == y).mean())
        print("OTPN learning curve (||z - t_z|| per pass):", [round(l, 3) for l in loss])
        print("final train cell-accuracy: %.3f" % acc)
    else:
        DIR = "arc_data/"; split = cmd
        ch = json.load(open(DIR + "arc-agi_%s_challenges.json" % split))
        sol = json.load(open(DIR + "arc-agi_%s_solutions.json" % split))
        t0 = time.time(); solved = []; acc = 0
        for tid, task in ch.items():
            try: pr = solve(task)
            except Exception: pr = None
            if pr is None: continue
            acc += 1
            if all(eq(pr[i], A(g)) for i, g in enumerate(sol[tid])): solved.append(tid)
        print("OTPN (octonionic target-prop, gradient-free): %d / %d  %s  (%.0fs) | accepted %d, precision %.2f"
              % (len(solved), len(ch), split, time.time() - t0, acc, len(solved) / max(acc, 1)))
        open("/tmp/otpn_%s.ids" % split, "w").write(" ".join(solved))
