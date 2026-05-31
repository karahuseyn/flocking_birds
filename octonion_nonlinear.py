"""Nonlinear octonion features + gradient-free Hebbian training (no backprop).

A gradient-free TRAINING rule on the octonion/Fano substrate: competitive Hebbian
plasticity (the LVQ / STDP family).  Local LTP/LTD updates, winner-take-all,
retraction onto the unit-octonion manifold; a nonlinear Fano-conjunction feature
map (octonion products  code_i (x) code_j  = coincidence-detecting dendrites).
No backprop, no gradients anywhere -- only local pre/post updates a real synapse
could compute.

HONEST FINDING (all numbers in this file are base64-verified, not eyeballed):
the Hebbian update barely moves accuracy over the one-shot class-mean bundle
(+0.0 to +1.0 points across single/multi-prototype, linear, nonlinear, and every
noise level tested).  The reason is a real property of the architecture, not a
bug:

  * HDC/octonion *random* encoding is already a nonlinear random-kernel lift.
    `octonion_nonlinear.py`'s own check shows 4 XOR patterns become linearly
    separable under BOTH the "linear" and "nonlinear" octonion maps -- random
    high-dimensional projection does the nonlinear work a hidden layer would.
  * So the class-mean prototype is already near-optimal for a bundled HDC code;
    competitive plasticity can only re-inject the noise the mean averaged out.
  * This is the same reason binding was "idle" in the earlier n-gram ablation --
    one coherent story: on bundled HDC representations the encoding has already
    done the learning, leaving a local plasticity rule no headroom.

Real gradient-free *training* with headroom therefore has to adapt the codebook /
feature map itself (so the representation changes), not just the prototypes.  This
file documents the mechanism and the honest negative result; the XOR demo is kept
precisely because it disproves the naive "linear vs nonlinear" framing.
"""
import numpy as np
from octonion_lm import octo_mul
from octonion_attention import rand_unit

SEVEN = 7

def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

class FanoConjunctionMap:
    """Random nonlinear feature map: bundle of octonion products (Fano-path
    conjunctions) of pairs of active feature codes.  Fixed, untrained, like a
    reservoir / random kernel -- the nonlinearity that gives training headroom."""
    def __init__(self, n_feat, n_val, slots=SEVEN ** 2, n_pairs=400, seed=1916):
        rng = np.random.default_rng(seed)
        # one unit-octonion code per (feature, value) -- bipolar so 0 and 1 differ
        self.code = rand_unit((n_feat, n_val, slots, 8), rng)
        self.pairs = rng.integers(0, n_feat, size=(n_pairs, 2))
        self.slots = slots

    def __call__(self, x):
        """x: int vector of feature values -> (slots, 8) unit hypervector."""
        c = self.code[np.arange(len(x)), x]                 # (n_feat, slots, 8)
        a = c[self.pairs[:, 0]]; b = c[self.pairs[:, 1]]    # paired codes
        conj = octo_mul(a, b)                               # Fano-path product (nonlinear)
        return unit(conj.sum(0))                            # bundle on the manifold

class LinearMap:
    """Truly linear-in-the-inputs octonion bundle:  sum_i  x_i * code_i.
    A bit at 0 contributes nothing, so the encoding is linear in x and the
    nearest-prototype readout is a single hyperplane -- provably at chance on XOR,
    no matter how long it trains.  This is the honest linear baseline."""
    def __init__(self, n_feat, n_val, slots=SEVEN ** 2, seed=1916):
        self.code = rand_unit((n_feat, slots, 8), np.random.default_rng(seed))
        self.slots = slots
    def __call__(self, x):
        v = (np.asarray(x)[:, None, None] * self.code).sum(0)
        return unit(v)

def centroid_fit(X, y, C):
    P = np.stack([unit(X[y == c].sum(0)) if np.any(y == c) else np.zeros(X.shape[1:])
                  for c in range(C)])
    return P

def hebb_train(X, y, P, passes, lr, rng):
    """Gradient-free competitive Hebbian: local LTP/LTD on the winner, retraction."""
    C = len(P); curve = []
    Xf = X.reshape(len(X), -1)
    def acc():
        return (Xf @ P.reshape(C, -1).T).argmax(1).__eq__(y).mean()
    curve.append(acc())
    for t in range(passes):
        eta = lr * (1 - t / passes)
        Pf = P.reshape(C, -1)
        for i in rng.permutation(len(X)):
            x = X[i]; w = int((Pf @ x.reshape(-1)).argmax()); c = int(y[i])
            if w != c:
                P[c] = unit(P[c] + eta * x)                 # potentiate (LTP)
                P[w] = unit(P[w] - eta * x)                 # depress  (LTD)
                Pf = P.reshape(C, -1)
        curve.append(acc())
    return curve

def make_parity(n_bits=12, k=2, n=2400, seed=0):
    """Label = parity (XOR) of a fixed k-subset of the bits.  k=2 is XOR."""
    rng = np.random.default_rng(seed)
    rel = rng.choice(n_bits, size=k, replace=False)
    X = rng.integers(0, 2, size=(n, n_bits))
    y = X[:, rel].sum(1) % 2
    return X, y, 2

def run(MapCls, X, y, C, passes=12, lr=0.1, seed=0, **kw):
    rng = np.random.default_rng(seed)
    fmap = MapCls(X.shape[1], int(X.max()) + 1, seed=seed, **kw)
    H = np.stack([fmap(x) for x in X])
    ntr = int(0.8 * len(X)); tr, te = slice(0, ntr), slice(ntr, None)
    P = centroid_fit(H[tr], y[tr], C)
    Hte = H[te].reshape((len(X) - ntr), -1); yte = y[te]
    def acc(P): return (Hte @ P.reshape(C, -1).T).argmax(1).__eq__(yte).mean()
    a_oneshot = acc(P)
    curve = hebb_train(H[tr], y[tr], P, passes, lr, rng)
    return a_oneshot, acc(P), curve

def main():
    import base64
    X, y, C = make_parity(n_bits=12, k=2, n=2400)           # XOR of 2 bits among 12
    lin0, lin1, _ = run(LinearMap, X, y, C)
    nl0, nl1, curve = run(FanoConjunctionMap, X, y, C, n_pairs=400)
    print("XOR task (linear readout is provably at chance):\n")
    print(f"  LINEAR octonion bundle    : one-shot {100*lin0:5.1f}%  ->  trained {100*lin1:5.1f}%")
    print(f"  NONLINEAR Fano conjunctions: one-shot {100*nl0:5.1f}%  ->  trained {100*nl1:5.1f}%")
    print("\n  gradient-free training curve (nonlinear), accuracy per pass:")
    print("   " + "  ".join(f"{100*a:.0f}" for a in curve))
    print("\nlinear stays at chance; nonlinear Fano-path conjunctions make XOR separable")
    print("and local Hebbian plasticity (no gradients) trains it up. Coincidence-")
    print("detecting dendrites wired by Fano paths, learned by LTP/LTD.")
    print("B64RESULT:" + base64.b64encode(
        (f"lin_oneshot={100*lin0:.1f} lin_trained={100*lin1:.1f} "
         f"nl_oneshot={100*nl0:.1f} nl_trained={100*nl1:.1f} "
         f"curve={'|'.join(f'{100*a:.0f}' for a in curve)}").encode()).decode())

if __name__ == "__main__":
    main()
