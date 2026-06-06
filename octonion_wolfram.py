# octonion_wolfram.py
# ============================================================================
# A serious adaptation of Stephen Wolfram's cellular automata to ARC.
#
# Mathematical setting.  An ARC colour grid is a state of a two-dimensional
# cellular automaton over the finite alphabet F = {0,...,9} (plus an edge symbol
# for the boundary).  A task's transform is the EVOLUTION of this CA under a local
# rule phi : (centre, neighbourhood) -> F.  We do not assume a named operation; we
# MINE THE COMPUTATIONAL UNIVERSE -- learn phi from the task's own transitions and
# evolve it -- accepting a rule only if it reproduces every demonstration EXACTLY.
#
# Wolfram's rule families, made symmetry-aware (this is the leverage over a raw
# patch look-up, which cannot generalise to an unseen neighbourhood):
#
#   * OUTER-TOTALISTIC      key = (centre, multiset of neighbour colours).
#     Invariant under the symmetric group S_k permuting the k neighbours, so a rule
#     learned on one spatial arrangement fires on every arrangement with the same
#     colour census -- Wolfram's outer-totalistic class, lifted to a colour alphabet.
#
#   * TOTALISTIC            key = multiset of the whole neighbourhood incl. centre.
#
#   * D4-EQUIVARIANT        key = the orbit-canonical patch under D4, the dihedral
#     symmetry group of the square lattice (4 rotations x 2 reflections).  Learning
#     on canonical representatives makes phi exactly D4-equivariant: phi(g.D4) =
#     phi(g).D4.  (D4 is the lattice-symmetry shadow of the octonionic/triality
#     symmetry used elsewhere in this project.)  Lossless, yet 8x more sample-
#     efficient and generalising than the raw patch.
#
# Evolution.  By the principle of computational irreducibility there is in general
# no closed form for the T-step map; one must run the rule.  We evolve phi for a
# single step, and (for convergent / growth rules) to its FIXED POINT, searching
# neighbourhood radius and geometry.  Every accepted rule is verified EXACTLY on
# all train pairs, then evolved on the test input.  Gradient-free; no backprop; no
# predefined task-transforms -- phi is learned from data and is un-named.
# ============================================================================
import json, time, sys
import numpy as np
from octonion_arc import A, eq

EDGE = 10                                       # boundary symbol (Wolfram fixed boundary), digit 10
BASE = 11                                       # alphabet {0..9} + EDGE

# ---- neighbourhood geometries (offsets), centre first ----
MOORE1 = [(0, 0)] + [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
VN1 = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
MOORE2 = [(0, 0)] + [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if (dr, dc) != (0, 0)]

def _stack(g, offs):
    """(H,W,len(offs)) int array of the neighbourhood, EDGE-padded out of bounds."""
    g = A(g); H, W = g.shape
    rad = max(max(abs(dr), abs(dc)) for dr, dc in offs)
    P = np.full((H + 2 * rad, W + 2 * rad), EDGE, int); P[rad:rad + H, rad:rad + W] = g
    return np.stack([P[rad + dr:rad + dr + H, rad + dc:rad + dc + W] for dr, dc in offs], axis=2)

# ---- D4 action on a Moore-radius-1 (3x3) patch: the 8 index permutations ----
def _d4_perms():
    base = np.arange(9).reshape(3, 3); P = []
    for k in range(4):
        r = np.rot90(base, k)
        P.append(r.ravel()); P.append(np.fliplr(r).ravel())
    # map: our MOORE1 order is centre-first, not row-major; build a reindex to row-major and back
    rowmajor = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    to_rm = [rowmajor.index(o) for o in MOORE1]          # MOORE1 idx -> rowmajor idx
    from_rm = [MOORE1.index(o) for o in rowmajor]        # rowmajor idx -> MOORE1 idx
    perms = []
    for p in P:                                          # p permutes row-major positions
        perms.append(np.array([from_rm[p[to_rm[i]]] for i in range(9)]))
    return np.array(perms)                               # (8,9) permutations in MOORE1 order
D4 = _d4_perms()

# ---- keys: encode a neighbourhood stack -> integer rule key (per family) ----
def _enc(vals):                                          # (...,k) digits -> base-11 integer
    out = np.zeros(vals.shape[:-1], np.int64)
    for j in range(vals.shape[-1]): out = out * BASE + (vals[..., j] + (vals[..., j] < 0))  # -1 never occurs (EDGE used)
    return out

def _keys(g, offs, family):
    S = _stack(g, offs)                                  # (H,W,k), centre at index 0
    if family == "exact":
        return _enc(S)
    if family == "outer_tot":
        nb = np.sort(S[..., 1:], axis=2)
        return _enc(np.concatenate([S[..., :1], nb], axis=2))
    if family == "tot":
        return _enc(np.sort(S, axis=2))
    if family == "d4":                                   # only defined for MOORE1 (3x3)
        orb = S[..., D4]                                 # (H,W,8,9)
        enc = _enc(orb)                                  # (H,W,8)
        return enc.min(axis=2)                           # orbit-canonical key
    raise ValueError(family)

GEOM = {"exact": [MOORE1, VN1], "outer_tot": [MOORE1, VN1, MOORE2], "tot": [MOORE1, VN1], "d4": [MOORE1]}

# ---- learn phi (rule table) from the task's transitions; verify EXACTLY ----
def _learn(pairs, offs, family):
    phi = {}
    for i, o in pairs:
        if i.shape != o.shape: return None
        K = _keys(i, offs, family).ravel(); V = A(o).ravel()
        for k, v in zip(K.tolist(), V.tolist()):
            if phi.get(k, v) != v: return None          # rule must be a function (no contradiction)
            phi[k] = v
    return phi

def _step(g, phi, offs, family):
    g = A(g); K = _keys(g, offs, family); out = g.copy(); flat = out.ravel(); Kf = K.ravel()
    for idx in range(Kf.shape[0]):
        v = phi.get(int(Kf[idx]))
        if v is not None: flat[idx] = v
    return flat.reshape(g.shape)

def _evolve(g, phi, offs, family, steps):
    g = A(g)
    if steps == "fix":
        for _ in range(20):
            n = _step(g, phi, offs, family)
            if np.array_equal(n, g): break
            g = n
        return g
    for _ in range(steps): g = _step(g, phi, offs, family)
    return g

def rule_solver(pairs):
    """Search rule family x geometry x evolution; accept the first CA that reproduces
    every demonstration EXACTLY.  Order: most-abstract (best-generalising) first."""
    for family in ("d4", "outer_tot", "tot", "exact"):
        for offs in GEOM[family]:
            phi = _learn(pairs, offs, family)
            if phi is None: continue
            for steps in (1, "fix"):
                if all(eq(_evolve(i, phi, offs, family, steps), o) for i, o in pairs):
                    return lambda g, P=phi, O=offs, F=family, S=steps: _evolve(g, P, O, F, S)
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
    print("WOLFRAM CA (symmetry-aware, gradient-free): %d / %d  %s  (%.0fs)"
          % (len(solved), len(ch), split, time.time() - t0))
    open("/tmp/wolfram_%s.ids" % split, "w").write(" ".join(solved))
