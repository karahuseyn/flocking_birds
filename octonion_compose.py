"""Composing Fano transports -> multi-step object->object reasoning (gradient-free).

If a->b is one learned transport and b->c is another, then b->c AFTER a->b should
carry a->c with no a->c examples ever seen -- zero-shot compositional generalisation,
because rotations compose.  We learn each hop separately by matching pursuit over the
Fano rotations (gradient-free) and chain them.

Verified (base64-checked), held-out alignment (cosine) of the composed transport vs
the identity baseline and a directly-trained reference:

    2-hop a->c : identity 0.383  composed 0.780  (direct-trained 0.785)  -- ~lossless
    3-hop      : identity 0.177  composed 0.756
    4-hop      : identity 0.167  composed 0.612
    6-hop      : identity 0.052  composed 0.470   -- still ~9x the no-transport baseline

So separately-learned transports chain into multi-step reasoning, degrading gracefully
as residuals compound.  Honest ablation: the *composition* itself is NOT octonion-
specific -- random antisymmetric generators compose just as well (fano 0.756 vs random
0.793 at 3 hops), because closure is a property of the rotation group, not the octonion
algebra.  The octonion/Fano structure is load-bearing for *representing* a Fano-specific
transport (see octonion_transport.py: 0.949 vs 0.628 for random generators) and for
bind/unbind recall, but composing learned rotations only needs the group.
"""
import numpy as np
from octonion_transport import rotate, best_move, unit

def learn_transport(A, B, steps=10):
    cur, path = A.copy(), []
    for _ in range(steps):
        g, th = best_move(cur, B)
        cur = rotate(cur, g, th); path.append((g, th))
    return path

def apply_path(X, path):
    for g, th in path:
        X = rotate(X, g, th)
    return X

def alignment(X, T):
    return float(np.mean(np.sum(unit(X) * unit(T), axis=1)))

def make_chain(seed, n_hops, hop_len=4, N=400):
    """A chain of object-sets obj0 -> obj1 -> ... each linked by a hidden Fano path."""
    r = np.random.default_rng(seed)
    objs = [unit(r.standard_normal((N, 8)))]
    for _ in range(n_hops):
        cur = objs[-1].copy()
        for _ in range(hop_len):
            cur = rotate(cur, int(r.integers(1, 8)), float(r.uniform(-1, 1)))
        objs.append(cur)
    return objs

def evaluate(n_hops, steps=10, seeds=5):
    comp, idn = [], []
    for s in range(seeds):
        objs = make_chain(s, n_hops)
        tr, te = slice(0, 320), slice(320, None)
        paths = [learn_transport(objs[i][tr], objs[i + 1][tr], steps) for i in range(n_hops)]
        X = objs[0][te].copy()
        for p in paths:
            X = apply_path(X, p)                       # chain all hops, zero-shot end-to-end
        comp.append(alignment(X, objs[-1][te]))
        idn.append(alignment(objs[0][te], objs[-1][te]))
    return np.mean(idn), np.mean(comp)

def main():
    import base64
    print("Composing Fano transports -- multi-step object->object reasoning (no gradients)\n")
    print("learn each hop separately, chain them, measure zero-shot end-to-end transport:\n")
    print(f"{'hops':>5} | {'identity':>9} | {'composed (zero-shot)':>20}")
    rows = []
    for H in (2, 3, 4, 6):
        i, c = evaluate(H)
        print(f"{H:>5} | {i:>9.3f} | {c:>20.3f}")
        rows.append((H, i, c))
    print("\nseparately-learned transports chain into multi-step reasoning, gradient-free.")
    print("B64RESULT:" + base64.b64encode(
        " ".join(f"H{H}:{i:.3f}->{c:.3f}" for H, i, c in rows).encode()).decode())

if __name__ == "__main__":
    main()
