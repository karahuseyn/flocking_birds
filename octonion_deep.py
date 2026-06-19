"""Deep + gated octonion memory -- still gradient-free, still O(n) (step 5).

Two ways to make the mechanism stronger *without a single gradient*, using
advanced structure of the octonions rather than learning:

1. DEPTH by octonion composition.  Octonion multiplication is non-commutative,
   so a (x) b != b (x) a -- an *ordered* pair maps to a distinct unit octonion.
   And the octonion norm is multiplicative (|xy| = |x||y|), so composing unit
   codes is an *isometry*: representations never blow up or vanish with depth, so
   we can stack composition layers with no normalisation and no training.  A
   single induction head keys on the last token; a 2-layer circuit keys on the
   *composite* of the last two tokens (layer 1 binds them, layer 2 recalls),
   which resolves context-sensitive recall a 1-layer head cannot.

2. GATING by a fixed forgetting rule.  A decayed running memory
   S_t = lambda * S_{t-1} + bind(k_t, v_t)  weights recent bindings more, with no
   learned gate.  On a non-stationary stream (a key reassigned to a new value)
   the gate makes the *latest* binding win; the ungated sum blends old and new.

Everything is octonion bind/unbind (octonion product), gradient-free, O(n).
"""
import numpy as np
from octonion_attention import bind, unbind, rand_unit, SEVEN

SLOTS = SEVEN ** 3

def cleanup(rec, codebook_flat):
    return (codebook_flat @ rec.reshape(-1)).argmax()

# --------------------------------------------------------------------------
# 1. Depth: context-sensitive in-context recall  (1-layer vs 2-layer)
# --------------------------------------------------------------------------
def depth_demo(n_triples, slots=SLOTS, vocab=160, trials=300, seed=0):
    rng = np.random.default_rng(seed)
    emb = rand_unit((vocab, slots, 8), rng)
    cb = emb.reshape(vocab, -1); cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)
    one = two = 0
    for _ in range(trials):
        # ambiguous keys: each 'b' is reused with several different 'a' -> different 'c'
        bs = rng.integers(0, vocab, size=max(2, n_triples // 2))
        a = rng.integers(0, vocab, size=n_triples)
        b = rng.choice(bs, size=n_triples)                  # collisions on b
        c = rng.integers(0, vocab, size=n_triples)
        M1 = np.zeros((slots, 8)); M2 = np.zeros((slots, 8))
        for i in range(n_triples):
            M1 += bind(emb[b[i]], emb[c[i]])                # layer-1 key = last token
            ck = bind(emb[a[i]], emb[b[i]])                 # composite (a (x) b), ordered
            M2 += bind(ck, emb[c[i]])                       # 2-layer key = composite
        q = rng.integers(0, n_triples)                      # query one triple
        one += (cleanup(unbind(emb[b[q]], M1), cb) == c[q])
        ckq = bind(emb[a[q]], emb[b[q]])
        two += (cleanup(unbind(ckq, M2), cb) == c[q])
    return one / trials, two / trials

# --------------------------------------------------------------------------
# 2. Gating: non-stationary recall  (ungated sum vs decayed memory)
# --------------------------------------------------------------------------
def gating_demo(n_keys, slots=SLOTS, vocab=160, lam=0.95, trials=300, seed=0):
    rng = np.random.default_rng(seed)
    emb = rand_unit((vocab, slots, 8), rng)
    cb = emb.reshape(vocab, -1); cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)
    ung = gat = 0
    for _ in range(trials):
        keys = rng.choice(vocab // 2, size=n_keys, replace=False)
        v1 = rng.integers(vocab // 2, vocab, size=n_keys)   # first assignment
        v2 = rng.integers(vocab // 2, vocab, size=n_keys)   # later reassignment
        writes = [(keys[i], v1[i]) for i in range(n_keys)]  # write all v1 ...
        writes += [(keys[i], v2[i]) for i in range(n_keys)] # ... then reassign to v2
        Su = np.zeros((slots, 8)); Sg = np.zeros((slots, 8))
        for k, v in writes:
            Su = Su + bind(emb[k], emb[v])                  # ungated sum
            Sg = lam * Sg + bind(emb[k], emb[v])            # decayed (forgetting gate)
        q = rng.integers(0, n_keys)                         # query expects the LATEST (v2)
        ung += (cleanup(unbind(emb[keys[q]], Su), cb) == v2[q])
        gat += (cleanup(unbind(emb[keys[q]], Sg), cb) == v2[q])
    return ung / trials, gat / trials

def main():
    print("Deep + gated octonion memory  (gradient-free, O(n), octonions everywhere)\n")

    print("1) DEPTH -- context-sensitive recall (the same 'b' continues differently")
    print("   depending on the 'a' before it; 1-layer keys on b, 2-layer on a(x)b):")
    print(f"   {'triples':>8} | {'1-layer':>9} | {'2-layer (composite)':>20}")
    for n in (4, 8, 16, 24, 32):
        o, t = depth_demo(n)
        print(f"   {n:>8} | {100*o:>8.1f}% | {100*t:>19.1f}%")

    print("\n2) GATING -- non-stationary recall (a key is reassigned; query wants the")
    print("   latest value; ungated sum blends old+new, decayed memory keeps newest):")
    print(f"   {'keys':>8} | {'ungated':>9} | {'decayed gate (lam=.95)':>23}")
    for n in (4, 8, 16, 24):
        u, g = gating_demo(n)
        print(f"   {n:>8} | {100*u:>8.1f}% | {100*g:>22.1f}%")

    print("\nDepth (octonion composition) and gating (a fixed forgetting rule) both add")
    print("real capability with no gradients; unit-octonion multiplication is an isometry,")
    print("so the stack stays norm-stable without any normalisation or training.")

if __name__ == "__main__":
    main()
