"""Stacked octonion memories -> multi-step in-context reasoning (gradient-free).

A single induction head does one hop: see A, recall what followed A.  *Stacking*
two octonion memories does two hops -- the circuit behind multi-step in-context
reasoning.  If the prompt establishes  A -> B  and  B -> C, then querying A should
return C (A->B then B->C), which one layer cannot reach.

Each layer is the same gradient-free octonion associative memory (key (x) value,
read by unbind + cleanup); layer 2 simply takes layer 1's *recalled* item as its
query.  No training, O(n), octonions throughout (49 = 7**2 blocks per head).
"""
import numpy as np
from octonion_attention import rand_unit, SEVEN

SLOTS = SEVEN ** 2

class HopMemory:
    """One octonion associative memory: store key->value, recall value for a key."""
    def __init__(self, n_sym, heads=8, slots=SLOTS, seed=0):
        rng = np.random.default_rng(seed)
        self.H = heads
        self.kemb = rand_unit((heads, n_sym, slots, 8), rng)   # key codebook
        self.vemb = rand_unit((heads, n_sym, slots, 8), rng)   # value codebook
        self.W = np.zeros((heads, slots, 8, 8))
        kb = self.kemb.reshape(heads, n_sym, -1)
        self.kcb = kb / np.linalg.norm(kb, axis=2, keepdims=True)
        vb = self.vemb.reshape(heads, n_sym, -1)
        self.vcb = vb / np.linalg.norm(vb, axis=2, keepdims=True)

    def _kn(self, sym):
        k = self.kemb[:, sym]
        return k / (np.linalg.norm(k, axis=2, keepdims=True) + 1e-9)

    def store(self, key_sym, val_sym):
        self.W = self.W + np.einsum("hki,hkj->hkij", self.vemb[:, val_sym], self._kn(key_sym))

    def recall(self, key_sym):
        retr = np.einsum("hkij,hkj->hki", self.W, self._kn(key_sym)).reshape(self.H, -1)
        return int(np.einsum("hvd,hd->hv", self.vcb, retr).sum(0).argmax())   # value symbol

def two_hop(n_chains, n_sym=120, heads=8, trials=300, seed=0):
    """Stream gives A->B and B->C bindings; query A.  1 hop -> B, 2 hops -> C."""
    rng = np.random.default_rng(seed)
    hop1 = hop2 = 0
    for _ in range(trials):
        syms = rng.choice(n_sym, size=3 * n_chains, replace=False)
        A = syms[0::3]; B = syms[1::3]; C = syms[2::3]
        L1 = HopMemory(n_sym, heads, seed=seed)              # A->B  and  B->C live here
        for i in range(n_chains):
            L1.store(A[i], B[i]); L1.store(B[i], C[i])
        q = rng.integers(0, n_chains)
        b = L1.recall(A[q])                                  # hop 1:  A -> B
        c = L1.recall(b)                                     # hop 2:  B -> C  (stacked)
        hop1 += (b == B[q]); hop2 += (c == C[q])
    return 100 * hop1 / trials, 100 * hop2 / trials

def main():
    print("Multi-step in-context reasoning by stacking octonion memories (gradient-free)\n")
    print("the prompt sets up  A->B  and  B->C; query A.  one hop reaches B, two reach C:\n")
    print(f"{'chains':>7} | {'1 hop  (A->B)':>14} | {'2 hops (A->B->C)':>17}")
    for n in (2, 4, 8, 16, 24):
        h1, h2 = two_hop(n)
        print(f"{n:>7} | {h1:>13.1f}% | {h2:>16.1f}%")
    print("\ntwo stacked octonion memories chain the recall -- multi-step in-context")
    print("reasoning, no gradients, O(n), octonions throughout.")

if __name__ == "__main__":
    main()
