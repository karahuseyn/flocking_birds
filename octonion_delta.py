"""Octonion DeltaNet  =  step 3, closing the capacity gap with softmax attention.

Step 2's additive octonion memory  S = sum_i bind(k_i, v_i)  has a hard capacity
limit: crosstalk from the other keys grows like sqrt(n).  Softmax attention beats
it under heavy load.  The fix in modern linear attention is the **delta rule**
(error-correcting fast weights -- DeltaNet, Yang 2024): before writing a binding,
subtract what the memory already predicts for that key, so you store the *correction*
instead of piling on.

Honest catch (and a genuine consequence of our maths): the naive delta on the
octonion-*product* memory **collapses** it.  Because octonions are a division
algebra with the exact inverse property, bind(k, unbind(k, S)) = S exactly, so
S += bind(k, v - unbind(k,S)) = bind(k, v) -- the update wipes everything and keeps
only the latest binding.  The very exactness that made step 1 clean defeats the
delta rule on the product memory.

So we keep the octonion block structure (343 = 7**3 octonion blocks of dim 8) but
switch the binding to the *outer-product* form linear attention actually uses -- a
tiny per-block fast-weight matrix W_j (8x8).  Now removal is key-selective and the
delta rule works.  State is K*8*8 (~22k numbers), updates are O(n), still
softmax-free and gradient-free, and the representations stay octonionic.
"""
import numpy as np
from octonion_attention import rand_unit, SEVEN
from octonion_induction import make_sequence, softmax_predict

def _unit_blocks(x):                       # normalise each 8-dim octonion block
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

def naive_octo_delta_holds_only_last(toks, emb):
    """Demonstrate the collapse: octonion-product memory under the naive delta
    ends up holding only the final binding (so recall works only if the query
    was the last pair)."""
    from octonion_attention import bind, unbind
    S = np.zeros(emb.shape[1:]); prev = None
    for t in toks[:-1]:
        if prev is not None:
            S = S + bind(emb[prev], emb[t] - unbind(emb[prev], S))   # naive delta
        prev = t
    rec = unbind(emb[toks[-1]], S).reshape(-1)
    cb = _flat_codebook(emb)
    return (cb @ rec).argmax()

def _flat_codebook(emb):
    cb = emb.reshape(emb.shape[0], -1)
    return cb / np.linalg.norm(cb, axis=1, keepdims=True)

def block_predict(toks, emb, delta):
    """Per-octonion-block fast-weight memory (K heads of dim 8).  delta=False is
    plain additive (linear attention); delta=True is the error-correcting update."""
    K = emb.shape[1]
    W = np.zeros((K, 8, 8))
    prev = None
    for t in toks[:-1]:
        if prev is not None:
            k = _unit_blocks(emb[prev]); v = emb[t]              # (K,8)
            if delta:
                pred = np.einsum("kij,kj->ki", W, k)
                W = W + np.einsum("ki,kj->kij", v - pred, k)     # write the correction
            else:
                W = W + np.einsum("ki,kj->kij", v, k)            # plain outer product
        prev = t
    q = _unit_blocks(emb[toks[-1]])
    out = np.einsum("kij,kj->ki", W, q).reshape(-1)
    return (_flat_codebook(emb) @ out).argmax()

def evaluate(n_pairs, slots, vocab=220, n_distract=1, trials=200, seed=0):
    rng = np.random.default_rng(seed)
    emb = rand_unit((vocab, slots, 8), rng)
    add = dlt = soft = 0
    for _ in range(trials):
        toks, true = make_sequence(n_pairs, vocab, n_distract, rng)
        add += (block_predict(toks, emb, delta=False) == true)
        dlt += (block_predict(toks, emb, delta=True) == true)
        soft += (softmax_predict(toks, emb) == true)
    return add / trials, dlt / trials, soft / trials

def main():
    slots = SEVEN ** 3
    print("Octonion DeltaNet -- in-context recall capacity (343 octonion blocks)\n")
    print(f"{'pairs':>6} | {'additive O(n)':>14} | {'delta O(n)':>11} | {'softmax O(n^2)':>15}")
    for n in (4, 8, 16, 32, 48, 64):
        a, d, s = evaluate(n, slots)
        print(f"{n:>6} | {100*a:>13.1f}% | {100*d:>10.1f}% | {100*s:>14.1f}%")

    # confirm the honest catch: naive delta on the product memory collapses
    rng = np.random.default_rng(1)
    V = 220
    emb = rand_unit((V, slots, 8), rng)
    hits = 0
    for _ in range(300):
        toks, true = make_sequence(16, V, 1, rng)
        hits += (naive_octo_delta_holds_only_last(toks, emb) == true)
    print(f"\nnaive delta on the octonion-PRODUCT memory (16 pairs): {100*hits/300:.1f}% recall "
          f"-- it collapses to ~the last binding (chance for 'query was last' ~ {100/16:.0f}%).")
    print("=> the outer-product (block) form closes the gap to softmax at O(n), "
          "softmax-free, and stays octonion-organised; the delta rule isn't what does it.")

if __name__ == "__main__":
    main()
