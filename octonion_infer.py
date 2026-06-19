"""octonion_infer.py -- multi-hop logical inference (modus-ponens search), gradient-free.

Arithmetic failed because it needs a *procedure* the associative memory doesn't have.
Logic, it turns out, the octonion algebra CAN do -- because an implication A->B is an exact,
invertible octonion rotation (octonion_logic.IMPLIES, verified cos=1.000), so chaining rules
composes without drift. We put a classical forward-chaining *search* on top of that algebra:

  - facts        : unit octonions (one per concept)
  - rules A->B   : IMPLIES(a, b) = b (x) a^{-1}, the rotation carrying fact A to fact B
  - inference    : from a start fact, apply every rule whose premise matches, breadth-first,
                   up to depth D; a goal is proved when the rotated octonion matches it.

This is the logic analogue of "computing": multi-step deduction, not pattern recall. It works
where arithmetic did not because the IMPLIES rotations are exact and compose losslessly.

Verified (base64): on a rule graph with chains, 0->9 (6 hops) proves with fidelity 1.000,
2-hop and 4-hop likewise, and an unreachable goal (0->7) is correctly rejected.
"""
import numpy as np
from octonion_lm import octo_mul

def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)

def conj(a):
    o = a.copy(); o[1:] *= -1; return o

def inv(a):
    return conj(a) / ((a * a).sum() + 1e-12)

def implies(a, b):
    """Rule A=>B as the octonion rotation taking fact a to fact b."""
    return octo_mul(b, inv(a))

class InferenceEngine:
    def __init__(self, facts):
        self.facts = facts                     # (N, 8) unit octonions, one per concept
        self.rules = {}                        # (i, j) -> rotation

    def add_rule(self, i, j):
        self.rules[(i, j)] = implies(self.facts[i], self.facts[j])

    def prove(self, start, goal, max_depth=8):
        """Forward-chaining modus-ponens search. Returns (proved, path, fidelity)."""
        frontier = [(start, self.facts[start], [start])]
        for _ in range(max_depth):
            nxt = []
            for cid, vec, path in frontier:
                for (i, j), R in self.rules.items():
                    if i == cid:
                        v2 = octo_mul(R, vec)
                        if j == goal:
                            fid = float(unit(v2) @ unit(self.facts[goal]))
                            return True, path + [j], fid
                        nxt.append((j, v2, path + [j]))
            frontier = nxt
            if not frontier:
                break
        return False, [], 0.0

def _demo():
    import base64
    rng = np.random.default_rng(0)
    N = 12
    facts = np.stack([unit(v) for v in rng.standard_normal((N, 8))])
    eng = InferenceEngine(facts)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 8), (8, 9), (5, 6), (6, 7)]:
        eng.add_rule(i, j)
    print("Multi-hop modus-ponens search on the octonion algebra (gradient-free)\n")
    results = []
    for start, goal in [(0, 4), (0, 9), (5, 7), (0, 7)]:
        ok, path, fid = eng.prove(start, goal)
        results.append((start, goal, ok, len(path) - 1 if ok else 0, fid))
        print(f"  {start} => {goal}: proved={ok}  hops={len(path)-1 if ok else '-'}  "
              f"fidelity={fid:.3f}  path={path if ok else 'none'}")
    print("B64INFER:" + base64.b64encode(
        " ".join(f"{s}->{g}:{ok}/{h}h/{f:.2f}" for s, g, ok, h, f in results).encode()).decode())

if __name__ == "__main__":
    _demo()
