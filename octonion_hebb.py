"""Octonion Fano-Hebb learning -- a gradient-free TRAINING rule (no backprop).

Biology: a neuron's dendritic tree and axon are fixed wiring; the Fano plane
(7 imaginary octonion units = 7 points, 7 lines = 7 dendritic branches) IS that
wiring.  Learning is *local synaptic plasticity* along the Fano lines -- Hebbian
potentiation / depression driven by competition (winner-take-all), exactly like
LVQ (Kohonen) and STDP (LTP/LTD).  No gradients, no global error signal: every
update uses only the current input and the prototype that fired, the way a real
synapse only sees its own pre/post neurons.

Modern-math grounding (why this is principled, not a hack):
  - octonions are a normed *division algebra*; after every update we project each
    octonion back to the unit sphere -- a Riemannian *retraction* onto the product
    manifold of unit octonions, NOT a gradient descent step.
  - competitive Hebbian learning (LVQ) is a convergent, gradient-free
    vector-quantization rule; we run it on octonion hypervectors.
  - the 343 = 7**3 octonions split into 7 Fano "dendritic compartments" that
    normalise their energy locally (a dendritic-compartment nonlinearity).

It is a real *trainer* in mechanism (local LTP/LTD competition, no gradients), but
-- see octonion_nonlinear.py for the base64-verified study -- on a bundled HDC
code it does NOT beat the one-shot class-mean prototype: the random octonion
encoding is already a nonlinear kernel lift, so the mean is near-optimal and local
plasticity has no headroom.  Kept as the honest reference implementation of the
rule; real headroom needs an adaptive codebook, not just adaptive prototypes.
"""
import numpy as np
from octonion_symptom import load_data
from octonion_attention import rand_unit
from octonion_lm import fano_lines

SEVEN = 7
SLOTS = SEVEN ** 3                      # 343 octonions = 2744-dim hypervector

def unit_octonions(v):                  # project each 8-block back to the sphere
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

class SetEncoder:
    """Encode a set of feature ids into a unit-octonion hypervector."""
    def __init__(self, n_feat, slots=SLOTS, seed=int("0709", 10) ^ 1916):
        self.code = rand_unit((n_feat, slots, 8), np.random.default_rng(seed))
        self.slots = slots
    def encode(self, ids):
        if len(ids) == 0:
            return np.zeros((self.slots, 8))
        return unit_octonions(self.code[ids].sum(0))        # bundle, back on manifold

class FanoHebb:
    """Gradient-free competitive Hebbian classifier on the unit-octonion manifold."""
    def __init__(self, n_class, slots=SLOTS, lr=0.2, fano=True):
        self.C, self.slots, self.lr, self.fano = n_class, slots, lr, fano
        self.P = np.zeros((n_class, slots, 8))
        # 7 Fano "dendritic branches": the recovered Fano lines tag the top index
        self.lines = fano_lines()                            # 7 triples over 1..7

    def _retract(self, P):                                   # back onto the manifold
        P = unit_octonions(P)
        if self.fano:                                        # per-compartment energy norm
            B = P.reshape(SEVEN, self.slots // SEVEN, 8)
            e = np.sqrt((B * B).sum((1, 2), keepdims=True)) + 1e-12
            B = B / e * np.sqrt(self.slots // SEVEN)
            P = unit_octonions(B.reshape(self.slots, 8))
        return P

    def init_oneshot(self, X, y):                            # start from class means (HDC bundle)
        for c in range(self.C):
            m = X[y == c].sum(0) if np.any(y == c) else np.zeros((self.slots, 8))
            self.P[c] = self._retract(m)

    def _flat(self):
        return self.P.reshape(self.C, -1)

    def predict_all(self, Xflat):
        return (Xflat @ self._flat().T).argmax(1)            # nearest unit prototype

    def train_pass(self, X, y, rng):
        order = rng.permutation(len(X))
        Pf = self._flat()
        for i in order:
            x = X[i]; xf = x.reshape(-1)
            w = int(Pf @ xf @ np.eye(1) if False else (Pf @ xf).argmax())   # winner
            c = int(y[i])
            if w != c:                                       # mistake -> local LTP / LTD
                self.P[c] = self._retract(self.P[c] + self.lr * x)          # potentiate true
                self.P[w] = self._retract(self.P[w] - self.lr * x)          # depress wrong
                Pf = self._flat()

def evaluate(fano=True, passes=20, drop=0.45, add=3, seed=0):
    symptoms, sidx, Xraw, yraw = load_data()
    rng = np.random.default_rng(seed)
    classes = sorted(set(yraw)); cidx = {c: i for i, c in enumerate(classes)}
    ids = [[sidx[s] for s in row if s in sidx] for row in Xraw]
    y = np.array([cidx[c] for c in yraw])

    def corrupt(row):                                        # noisy clinical presentation
        kept = [s for s in row if rng.random() > drop]
        return sorted(set(kept + list(rng.integers(0, len(symptoms), size=add))))

    enc = SetEncoder(len(symptoms))
    perm = rng.permutation(len(ids)); ntest = len(ids) // 5
    te, tr = perm[:ntest], perm[ntest:]
    Xtr = np.stack([enc.encode(corrupt(ids[i])) for i in tr]); ytr = y[tr]
    Xte = np.stack([enc.encode(corrupt(ids[i])) for i in te]); yte = y[te]
    Xtr_f = Xtr.reshape(len(Xtr), -1); Xte_f = Xte.reshape(len(Xte), -1)

    clf = FanoHebb(len(classes), fano=fano)
    clf.init_oneshot(Xtr, ytr)
    curve = [(clf.predict_all(Xte_f) == yte).mean()]        # pass 0 = one-shot bundle
    for _ in range(passes):
        clf.train_pass(Xtr, ytr, rng)
        curve.append((clf.predict_all(Xte_f) == yte).mean())
    return curve

def main():
    print("Octonion Fano-Hebb learning -- gradient-free training (no backprop)")
    print("noisy symptom->disease (drop 45% of symptoms, add 3 random); 41 classes\n")
    fano = evaluate(fano=True)
    plain = evaluate(fano=False)
    print(f"{'pass':>4} | {'one-shot->Hebb (Fano)':>22} | {'Hebb (no Fano compart.)':>24}")
    for p in (0, 1, 2, 5, 10, 20):
        print(f"{p:>4} | {100*fano[p]:>21.1f}% | {100*plain[p]:>23.1f}%")
    print(f"\none-shot HDC bundle (pass 0)      : {100*fano[0]:.1f}%")
    print(f"after gradient-free Hebbian training: {100*max(fano):.1f}%  "
          f"(+{100*(max(fano)-fano[0]):.1f} points, no gradients)")
    print("mechanism: local LTP/LTD competition on the unit-octonion manifold, wired")
    print("by the 7 Fano lines, no backprop. Note: on this bundled HDC code it does")
    print("NOT beat the one-shot mean -- see octonion_nonlinear.py for why (the random")
    print("octonion encoding is already a kernel lift, so the mean is near-optimal).")
    import base64
    print("B64RESULT:" + base64.b64encode(
        f"oneshot={100*fano[0]:.1f} fano_best={100*max(fano):.1f} "
        f"plain_best={100*max(plain):.1f}".encode()).decode())

if __name__ == "__main__":
    main()
