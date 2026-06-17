# octonion_otp.py
# ============================================================================
# Backprop replaced by a CLOSED-FORM INVERSE solve -- the "back-propagation" of the
# readout is handled algebraically, not by gradient descent.  Concretely an Extreme-
# Learning-Machine head on the octonionic recursive features: a fixed RANDOM hidden
# projection (ReLU), then the output weights are solved in ONE shot by the ridge
# normal equations W = (HᵀH + λI)⁻¹ Hᵀ Y -- the matrix inverse stands in for backprop.
# Same octonionic recursion and same held-out-pair + ensemble over-fit gate as O-TRM v2,
# so it is directly comparable on speed and accuracy.  No gradients anywhere.
# ============================================================================
import json, time, sys
import numpy as np
from octonion_arc import A, eq
from octonion_otrm import _feats                          # octonionic recursive per-cell features

def _fit(X, y, hid=64, seed=0, lam=1.0):
    rng = np.random.default_rng(seed); F = X.shape[1]
    R = rng.standard_normal((F, hid)) / np.sqrt(F); b = rng.standard_normal(hid) * 0.1
    H = np.maximum(X @ R + b, 0.0)
    Yoh = np.eye(10)[y]
    W = np.linalg.solve(H.T @ H + lam * np.eye(hid), H.T @ Yoh)   # closed-form inverse (no backprop)
    return (R, b, W)

def _pred(X, M):
    R, b, W = M; return (np.maximum(X @ R + b, 0.0) @ W).argmax(1)

def solve(task):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    tests = [A(tp["input"]) for tp in task["test"]]
    if len(pairs) < 2 or not all(i.shape == o.shape for i, o in pairs): return None
    feats = [_feats(i) for i, _ in pairs]; targs = [A(o).reshape(-1).astype(int) for _, o in pairs]
    X = np.concatenate(feats); y = np.concatenate(targs)
    K = 2; testX = [_feats(t) for t in tests]
    hj = len(pairs) - 1; idx = list(range(hj))                    # held-out-pair generalisation proof
    Xtr = np.concatenate([feats[k] for k in idx]); ytr = np.concatenate([targs[k] for k in idx])
    for s in range(K):
        M = _fit(Xtr, ytr, seed=s)
        if not np.array_equal(_pred(feats[hj], M), targs[hj]): return None
    preds_k = []
    for s in range(K):
        M = _fit(X, y, seed=s)
        if not np.array_equal(_pred(X, M), y): return None         # must reproduce all train cells
        preds_k.append([_pred(tx, M) for tx in testX])
    for ti in range(len(tests)):                                   # ensemble agreement
        for s in range(1, K):
            if not np.array_equal(preds_k[s][ti], preds_k[0][ti]): return None
    return [A(preds_k[0][ti].reshape(A(tests[ti]).shape)) for ti in range(len(tests))]


if __name__ == "__main__":
    DIR = "arc_data/"; split = sys.argv[1] if len(sys.argv) > 1 else "training"
    ch = json.load(open(DIR + "arc-agi_%s_challenges.json" % split))
    sol = json.load(open(DIR + "arc-agi_%s_solutions.json" % split))
    t0 = time.time(); solved = []; accepted = 0
    for tid, task in ch.items():
        try: pr = solve(task)
        except Exception: pr = None
        if pr is None: continue
        accepted += 1
        if all(eq(pr[i], A(g)) for i, g in enumerate(sol[tid])): solved.append(tid)
    print("O-TP (inverse/closed-form 'backprop', ELM head): %d / %d  %s  (%.0fs)  | accepted %d, precision %.2f"
          % (len(solved), len(ch), split, time.time() - t0, accepted, len(solved) / max(accepted, 1)))
    open("/tmp/otp_%s.ids" % split, "w").write(" ".join(solved))
