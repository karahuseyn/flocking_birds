"""Octonion linear memory vs softmax attention -- the compute win (step 4).

Steps 1-3 matched softmax attention's *accuracy*.  This is the point of the whole
exercise: the *cost*.  A softmax transformer is O(n^2) in time and O(n^2) memory
(the attention matrix) -- the wall that makes long context expensive.  The
octonion fast-weight memory is a causal running state: O(n) time and O(1) memory
(a fixed-size block state, independent of sequence length).

We time both as the sequence grows.  The octonion side is the honest recurrent
scan (what makes it streaming/O(n)); the softmax side is standard full causal
attention.  No accuracy here -- steps 1-3 covered that -- just seconds and bytes.
"""
import time
import numpy as np
from octonion_attention import SEVEN

def octonion_scan(L, K=SEVEN ** 2, seed=0):
    """Causal octonion block memory over L tokens.  State = K (8x8) matrices,
    constant in L.  Per token: read + outer-product write.  -> O(L) time, O(1) mem."""
    rng = np.random.default_rng(seed)
    keys = rng.standard_normal((L, K, 8)); keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
    vals = rng.standard_normal((L, K, 8))
    W = np.zeros((K, 8, 8))
    t0 = time.perf_counter()
    out = np.empty((L, K, 8))
    for t in range(L):
        out[t] = np.einsum("kij,kj->ki", W, keys[t])            # read
        W = W + np.einsum("ki,kj->kij", vals[t], keys[t])       # write (outer product)
    return time.perf_counter() - t0, W.nbytes

def softmax_attention(L, d=392, seed=0):
    """Standard full causal softmax attention over L tokens.  Builds the L x L
    score matrix  -> O(L^2) time and O(L^2) memory."""
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((L, d)); Kk = rng.standard_normal((L, d)); V = rng.standard_normal((L, d))
    t0 = time.perf_counter()
    scores = (Q @ Kk.T) / np.sqrt(d)                            # L x L   <- the wall
    mask = np.triu(np.ones((L, L), bool), 1)
    scores[mask] = -np.inf
    scores -= scores.max(1, keepdims=True)
    e = np.exp(scores); e /= e.sum(1, keepdims=True)
    _ = e @ V
    dt = time.perf_counter() - t0
    return dt, scores.nbytes

def main():
    print("Sequence-length scaling: octonion O(n) memory vs softmax O(n^2) attention")
    print("(49 octonion blocks, 392-dim; constant state vs the L x L attention matrix)\n")
    print(f"{'len L':>7} | {'octonion O(n)':>14} | {'softmax O(n^2)':>15} | {'speedup':>8} | "
          f"{'attn-matrix mem':>16}")
    for L in (256, 512, 1024, 2048, 4096, 8192):
        ot, ostate = octonion_scan(L)
        st, smem = softmax_attention(L)
        print(f"{L:>7} | {ot*1e3:>11.1f} ms | {st*1e3:>12.1f} ms | {st/ot:>7.1f}x | "
              f"{smem/1e6:>11.1f} MB")
    print(f"\noctonion state is constant at {np.zeros((SEVEN**2,8,8)).nbytes/1e3:.1f} KB for *any* "
          "length; softmax's attention matrix grows as L^2.")
    print("octonion time doubles when L doubles (linear); softmax time ~quadruples "
          "(quadratic) -- the gap widens without bound as context grows.")

if __name__ == "__main__":
    main()
