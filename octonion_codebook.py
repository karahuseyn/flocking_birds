"""Adaptive octonion codebook -- gradient-free training that beats the one-shot
bundle on REAL data (no backprop).

The Hebbian study moved *prototypes* (no headroom) and conjunction selection moved
the *feature set* (helps only on engineered interaction tasks).  This moves the
third lever -- the octonion **codes themselves** -- and it wins on real
symptom->disease data in the regime that matters: a small (capacity-limited) code
where class hypervectors collide.

The rule is gradient-free and local, a competitive Hebbian update on the unit-
octonion manifold:

    for each training case:  pull every active symptom's octonion code toward the
    prototype of its TRUE class, and push it away from the prototype of the wrong
    class that currently wins (LTP toward target, LTD away from the impostor);
    then RETRACT each code back onto the unit sphere (a manifold step, not a
    gradient step).

This rotates the symptom codes so that classes which were colliding become
separable -- learning the *representation*, not just the readout, with no
gradients anywhere.

Verified (base64-checked, 4 seeds), real 4920-case symptom->disease data,
slots=7 (tiny code), 30% symptom dropout:

    one-shot bundle : 89.2%
    adaptive codes  : 92.2%   (+3.0 points, gradient-free)

The win also transfers to differential-diagnosis *ranking* under heavier noise
(slots=7, 40% dropout, +4 random symptoms), base64-verified:

    top-1 : one-shot 72.9% -> adaptive 77.0%   (+4.1)
    top-3 : one-shot 87.7% -> adaptive 90.1%   (+2.4)

Honest scope (two parts):
  * the gain appears only when the code is capacity-limited and noisy (codes
    collide); on a roomy code or clean data the one-shot bundle is already optimal
    (100%) and there is nothing to learn.
  * the gain is from the gradient-free training RULE, not the octonion algebra:
    plain Euclidean unit vectors of the same dimension train just as well
    (ablation: octonion 89.9->92.0 vs vector 91.2->92.7).  Octonion structure is
    load-bearing only where the division-algebra INVERSE is used for unbinding
    (associative recall: octonion 100% vs elementwise 3% at 8 pairs) -- i.e. when
    you bind and unbind, not when you merely bundle and classify.
"""
import numpy as np
from octonion_symptom import load_data
from octonion_attention import rand_unit

def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

class AdaptiveCodebook:
    def __init__(self, n_feat, n_class, slots=7, seed=1916):
        self.code = rand_unit((n_feat, slots, 8), np.random.default_rng(seed))
        self.slots, self.C = slots, n_class

    def encode(self, rows):
        return np.stack([unit(self.code[r].sum(0)) if len(r) else np.zeros((self.slots, 8))
                         for r in rows])

    def _protos(self, H, y):
        return np.stack([unit(H[y == c].sum(0)) if np.any(y == c)
                         else np.zeros((self.slots, 8)) for c in range(self.C)])

    def fit(self, rows, y, passes=12, lr=0.15):
        """Gradient-free competitive code adaptation on the octonion manifold."""
        for t in range(passes):
            eta = lr * (1 - t / passes)
            H = self.encode(rows); P = self._protos(H, y); Pf = P.reshape(self.C, -1)
            upd = np.zeros_like(self.code); cnt = np.zeros(len(self.code))
            for r, row in enumerate(rows):
                if not len(row):
                    continue
                w = int((Pf @ H[r].reshape(-1)).argmax()); c = int(y[r])
                upd[row] += P[c]                       # LTP toward the true class
                if w != c:
                    upd[row] -= 0.5 * P[w]             # LTD away from the impostor
                cnt[row] += 1
            m = cnt > 0
            self.code[m] = unit(self.code[m] + eta * upd[m] / cnt[m, None, None])
        return self

def evaluate(slots=7, drop=0.3, add=2, passes=12, seeds=4):
    symptoms, sidx, Xraw, yraw = load_data()
    classes = sorted(set(yraw)); cidx = {c: i for i, c in enumerate(classes)}
    nF, C = len(symptoms), len(classes)
    ids = [[sidx[s] for s in r if s in sidx] for r in Xraw]
    y = np.array([cidx[c] for c in yraw])
    o_acc, a_acc = [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        B = np.zeros((len(ids), nF), int)
        for r, row in enumerate(ids):
            B[r, row] = 1
        Bc = B.copy()                                  # noisy presentation
        for r in range(len(ids)):
            for s in np.where(B[r])[0]:
                if rng.random() < drop:
                    Bc[r, s] = 0
            for s in rng.integers(0, nF, size=add):
                Bc[r, s] = 1
        rows = [list(np.where(Bc[r])[0]) for r in range(len(ids))]
        perm = rng.permutation(len(ids)); nt = len(ids) // 5
        te, tr = perm[:nt], perm[nt:]
        rows_tr = [rows[i] for i in tr]; ytr = y[tr]
        cb = AdaptiveCodebook(nF, C, slots=slots, seed=seed)
        def test():
            H = cb.encode(rows); P = cb._protos(cb.encode(rows_tr), ytr)
            return (H[te].reshape(len(te), -1) @ P.reshape(C, -1).T).argmax(1).__eq__(y[te]).mean()
        o_acc.append(test())
        cb.fit(rows_tr, ytr, passes=passes)
        a_acc.append(test())
    return 100 * np.mean(o_acc), 100 * np.mean(a_acc)

def evaluate_ranking(slots=7, drop=0.4, add=4, passes=12, seeds=3):
    """Differential-diagnosis ranking (top-1 / top-3) under heavier noise."""
    symptoms, sidx, Xraw, yraw = load_data()
    classes = sorted(set(yraw)); cidx = {c: i for i, c in enumerate(classes)}
    nF, C = len(symptoms), len(classes)
    ids = [[sidx[s] for s in r if s in sidx] for r in Xraw]
    y = np.array([cidx[c] for c in yraw])
    def topk(S, yte, k):
        order = np.argsort(-S, axis=1)[:, :k]
        return np.mean([yte[i] in order[i] for i in range(len(yte))])
    o1 = o3 = a1 = a3 = 0.0
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        B = np.zeros((len(ids), nF), int)
        for r, row in enumerate(ids):
            B[r, row] = 1
        Bc = B.copy()
        for r in range(len(ids)):
            for s in np.where(B[r])[0]:
                if rng.random() < drop:
                    Bc[r, s] = 0
            for s in rng.integers(0, nF, size=add):
                Bc[r, s] = 1
        rows = [list(np.where(Bc[r])[0]) for r in range(len(ids))]
        perm = rng.permutation(len(ids)); nt = len(ids) // 5
        te, tr = perm[:nt], perm[nt:]
        rows_tr = [rows[i] for i in tr]; ytr = y[tr]
        cb = AdaptiveCodebook(nF, C, slots=slots, seed=seed)
        def scores():
            H = cb.encode(rows); P = cb._protos(cb.encode(rows_tr), ytr)
            return H[te].reshape(len(te), -1) @ P.reshape(C, -1).T
        S = scores(); o1 += topk(S, y[te], 1); o3 += topk(S, y[te], 3)
        cb.fit(rows_tr, ytr, passes=passes)
        S = scores(); a1 += topk(S, y[te], 1); a3 += topk(S, y[te], 3)
    n = seeds
    return 100 * o1 / n, 100 * a1 / n, 100 * o3 / n, 100 * a3 / n

def main():
    import base64
    o, a = evaluate()
    print("Adaptive octonion codebook -- gradient-free training on real data\n")
    print("real symptom->disease, slots=7 (capacity-limited), 30% dropout, 4 seeds:\n")
    print(f"  one-shot bundle : {o:5.1f}%")
    print(f"  adaptive codes  : {a:5.1f}%   (+{a-o:.1f} points, no gradients)")
    o1, a1, o3, a3 = evaluate_ranking()
    print("\ndifferential-diagnosis ranking under heavier noise (40% dropout, +4 random):")
    print(f"  top-1 : {o1:5.1f}% -> {a1:5.1f}%   (+{a1-o1:.1f})")
    print(f"  top-3 : {o3:5.1f}% -> {a3:5.1f}%   (+{a3-o3:.1f})")
    print("\ngradient-free competitive code rotation on the unit-octonion manifold")
    print("separates colliding classes -- learning the representation, not the readout.")
    print("B64RESULT:" + base64.b64encode(
        f"oneshot={o:.1f} adaptive={a:.1f} top1={o1:.1f}->{a1:.1f} top3={o3:.1f}->{a3:.1f}".encode()).decode())

if __name__ == "__main__":
    main()
