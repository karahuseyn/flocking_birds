"""Octonion linear attention  =  fast-weight associative memory.

The transformer's one essential trick is *content-based* mixing: each query
pulls, from everything seen, the value bound to the matching key.  Softmax
attention does this in O(n^2); **linear attention** does it in O(n) by keeping a
running memory of key/value bindings and reading it with the query
(Katharopoulos 2020; Schlag 2021 "Linear Transformers are Secretly Fast Weight
Programmers" -- the lineage behind RWKV / RetNet / DeltaNet).

That memory is exactly an HDC bind / bundle / unbind:

    bind   :  b_i = k_i (x) v_i          (octonion product, blockwise)
    bundle :  S   = sum_i b_i            (causal running sum  ->  O(n))
    unbind :  v_q ~ k_q^{-1} (x) S       (read the value for a query key)

Octonions are one of only four normed division algebras (R, C, H, O), so every
nonzero octonion is invertible and the **inverse property holds exactly**:
k^{-1}(k v) = v.  Binding is therefore *exactly* invertible -- unlike the
circular convolution of classic HRR, which is only approximately invertible.
Cross-terms from the other (quasi-orthogonal) keys are the only noise, cleaned up
by nearest-neighbour to a value codebook.

This is the content-based recall a plain n-gram / pure HDC bundle *cannot* do --
the very thing attention provides -- here gradient-free and linear-time.  Slot
counts follow the rule of 7 (49 = 7**2, 343 = 7**3 octonions).
"""
import numpy as np
from octonion_lm import octo_mul, octo_norm

SEVEN = 7

def conj(a):
    """Octonion conjugate (negate the 7 imaginary parts). a: (..., 8)."""
    out = a.copy(); out[..., 1:] *= -1.0
    return out

def inv(a):
    """Octonion inverse  a^{-1} = conj(a)/|a|^2  (exact -- division algebra)."""
    return conj(a) / (a * a).sum(-1, keepdims=True)

def rand_unit(shape, rng):
    a = rng.standard_normal(shape)
    return a / octo_norm(a)

def bind(k, v):                       # blockwise octonion product  (...,K,8)
    return octo_mul(k, v)

def unbind(k, S):                     # k^{-1} (x) S  -> recovers the bound value
    return octo_mul(inv(k), S)

# --------------------------------------------------------------------------
# In-context associative recall ("induction"): present n key->value pairs, then
# a query key; recover its value.  The defining test that separates attention
# from an n-gram.  We solve it gradient-free with the octonion fast-weight memory.
# --------------------------------------------------------------------------
def recall_accuracy(n_pairs, slots, n_vals=100, trials=400, seed=0):
    rng = np.random.default_rng(seed)
    vals = rand_unit((n_vals, slots, 8), rng)          # value codebook (cleanup)
    vflat = vals.reshape(n_vals, -1)
    vflat /= np.linalg.norm(vflat, axis=1, keepdims=True)
    hits = 0
    for _ in range(trials):
        keys = rand_unit((n_pairs, slots, 8), rng)     # n distinct random keys
        vidx = rng.integers(0, n_vals, size=n_pairs)   # which value each binds
        S = bind(keys, vals[vidx]).sum(0)              # fast-weight memory (O(n))
        q = rng.integers(0, n_pairs)                   # query one of the keys
        rec = unbind(keys[q], S).reshape(-1)           # ~ value[vidx[q]] + noise
        pred = (vflat @ rec).argmax()                  # cleanup: nearest value
        hits += (pred == vidx[q])
    return hits / trials

def bundle_baseline(n_pairs, slots, n_vals=100, trials=400, seed=0):
    """No binding -- just bundle key+value (what plain HDC / an n-gram does).
    There is no way to read back *which* value went with the query key."""
    rng = np.random.default_rng(seed)
    vals = rand_unit((n_vals, slots, 8), rng)
    vflat = vals.reshape(n_vals, -1); vflat /= np.linalg.norm(vflat, axis=1, keepdims=True)
    hits = 0
    for _ in range(trials):
        keys = rand_unit((n_pairs, slots, 8), rng)
        vidx = rng.integers(0, n_vals, size=n_pairs)
        S = (keys + vals[vidx]).sum(0)                 # additive bundle, no bind
        q = rng.integers(0, n_pairs)
        rec = (S - keys[q]).reshape(-1)                # best you can do without bind
        pred = (vflat @ rec).argmax()
        hits += (pred == vidx[q])
    return hits / trials

def main():
    print("In-context associative recall  (gradient-free, O(n) octonion linear attention)")
    print("recover the value bound to a query key, out of a memory of n pairs\n")
    for slots in (SEVEN ** 2, SEVEN ** 3):             # 49, 343 octonions
        print(f"--- {slots} octonions  ({slots*8}-dim hypervector) ---")
        print(f"{'pairs':>6} | {'octonion bind/unbind':>22} | {'no-bind bundle':>15}")
        for n in (2, 4, 8, 16, 32, 64, 128):
            a = recall_accuracy(n, slots)
            b = bundle_baseline(n, slots)
            print(f"{n:>6} | {100*a:>21.1f}% | {100*b:>14.1f}%")
        print()
    print("octonion binding turns the memory into content-addressable attention;\n"
          "the plain bundle stays at chance -- it cannot separate key from value.")

if __name__ == "__main__":
    main()
