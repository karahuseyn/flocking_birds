# octonion_wolfram_x.py
# ============================================================================
# Growing the transformation VOCABULARY synthetically -- but staying honest: no
# predefined named operators.  We expand the family of UN-NAMED, parametric,
# fit-from-data discrete cellular-automaton operators, group-theoretically.
#
# octonion_wolfram.py canonicalises the neighbourhood under the FULL dihedral group
# D4 (the +5-novel family).  But a task may carry only PART of that symmetry --
# rotations but not reflections, one mirror axis, or a half-turn only.  Each
# SUBGROUP G <= D4 yields a different G-equivariant rule  phi(g.sigma)=phi(g).sigma
# for sigma in G, with its own generalisation strength.  We enumerate ALL subgroups
# of D4 as canonicalisation groups -- a principled, procedurally generated widening
# of the rule vocabulary -- plus radius-2 outer-totalistic keys.  Every rule is
# learned from the task's own transitions and accepted only on EXACT reproduction;
# gradient-free, no backprop, nothing named.
#
# Subgroups of D4 (|D4|=8), by index into octonion_wolfram.D4 (idx 2k = rot90^k,
# idx 2k+1 = fliplr o rot90^k):
#   {e}             trivial (= exact patch)
#   {e, r180}       C2     (half turn)
#   {e, m}          one mirror (horizontal)         {e, fliplr o r180} (vertical)
#   {e, d}          one diagonal mirror             {e, anti-diagonal}
#   {e,r90,r180,r270}  C4  (rotations only)
#   {e,r180,mh,mv}  D2     (rectangle symmetry)
#   D4              full dihedral (the original +5 family)
# ============================================================================
import json, time, sys
import numpy as np
from octonion_arc import A, eq
import octonion_wolfram as W

# subgroup index sets into W.D4  (0:e 2:r90 4:r180 6:r270 ; 1:m 3:m.r90 5:m.r180 7:m.r270)
SUBGROUPS = {
    "C4":   [0, 2, 4, 6],          # rotations only
    "D2":   [0, 4, 1, 5],          # half turn + both axis mirrors
    "C2":   [0, 4],                # half turn
    "mirror_h": [0, 1], "mirror_v": [0, 5],
    "diag": [0, 3], "anti": [0, 7],
}

def _subgroup_key(g, idxs):
    """Orbit-canonical 3x3 key under a chosen subgroup of D4 (min encoding over its
    elements) -- a G-equivariant rule key."""
    S = W._stack(g, W.MOORE1)                  # (H,W,9), centre-first MOORE1 order
    orb = S[..., W.D4[idxs]]                   # (H,W,|G|,9)
    return W._enc(orb).min(axis=2)

def _keys_x(g, spec):
    fam, arg = spec
    if fam == "sub":  return _subgroup_key(g, SUBGROUPS[arg])
    return W._keys(g, arg, fam)                # reuse outer_tot / tot / exact / d4

def _learn(pairs, spec):
    phi = {}
    for i, o in pairs:
        if i.shape != o.shape: return None
        K = _keys_x(i, spec).ravel(); V = A(o).ravel()
        for k, v in zip(K.tolist(), V.tolist()):
            if phi.get(k, v) != v: return None
            phi[k] = v
    return phi

def _step(g, phi, spec):
    g = A(g); K = _keys_x(g, spec); flat = g.copy().ravel(); Kf = K.ravel()
    for idx in range(Kf.shape[0]):
        v = phi.get(int(Kf[idx]))
        if v is not None: flat[idx] = v
    return flat.reshape(g.shape)

def _evolve(g, phi, spec, steps):
    g = A(g)
    if steps == "fix":
        for _ in range(20):
            n = _step(g, phi, spec)
            if np.array_equal(n, g): break
            g = n
        return g
    for _ in range(steps): g = _step(g, phi, spec)
    return g

# the expanded vocabulary, most-abstract (best-generalising) first
SPECS = ([("sub", g) for g in ("C4", "D2", "C2", "mirror_h", "mirror_v", "diag", "anti")]
         + [("d4", None)]
         + [("outer_tot", W.MOORE2), ("outer_tot", W.MOORE1), ("outer_tot", W.VN1)]
         + [("tot", W.MOORE1), ("tot", W.VN1)])

def rule_solver(pairs):
    for spec in SPECS:
        phi = _learn(pairs, spec)
        if phi is None: continue
        for steps in (1, "fix"):
            if all(eq(_evolve(i, phi, spec, steps), o) for i, o in pairs):
                return lambda g, P=phi, S=spec, T=steps: _evolve(g, P, S, T)
    return None

def solve(task):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    fn = rule_solver(pairs)
    if fn is None: return None
    try: return [A(fn(A(tp["input"]))) for tp in task["test"]]
    except Exception: return None


if __name__ == "__main__":
    DIR = "arc_data/"; split = sys.argv[1] if len(sys.argv) > 1 else "training"
    ch = json.load(open(DIR + "arc-agi_%s_challenges.json" % split))
    sol = json.load(open(DIR + "arc-agi_%s_solutions.json" % split))
    t0 = time.time(); solved = []
    for tid, task in ch.items():
        try: pr = solve(task)
        except Exception: pr = None
        if pr and all(eq(pr[i], A(g)) for i, g in enumerate(sol[tid])): solved.append(tid)
    print("WOLFRAM-X (D4 subgroups + radius-2, gradient-free): %d / %d  %s  (%.0fs)"
          % (len(solved), len(ch), split, time.time() - t0))
    open("/tmp/wolframx_%s.ids" % split, "w").write(" ".join(solved))
