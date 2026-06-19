# octonion_trm.py -- a REAL ARC solver, gradient-free, powered by a TRM-style recursion.
#
# TRM's idea (Tiny Recursive Model): keep a current answer and recursively REFINE it to a fixed
# point. TRM trains its refiner by backprop; we do NOT. Our refiner is pure octonion algebra and
# its "training" is a nearest-neighbour lookup in S^7 -- no gradients, no weights.
#
# The universe, applied at CELL scale:
#   * each cell -> a HOLOGRAPHIC NEIGHBOURHOOD octon: the 3x3 neighbours, each bound to one of 9
#     ROLE octons by the fano-path product, then superposed (the universe's carrier, locally).
#   * the RULE is a carrier set: train cells give {neighbourhood-octon -> output colour}. A cell is
#     solved by nearest-neighbour in S^7. Both endpoints are stored -- input cells (the change) AND
#     output cells mapped to themselves (the FIXED POINT), which is what makes the recursion halt.
#   * TRM recursion: apply the rule to the current grid, re-encode, apply again ... to a fixed point.
#     One shot recolours; ITERATING propagates (gravity, line growth, flood fill) -- tasks a single
#     application cannot reach. The gain from T=1 -> fixed-point is TRM's measured contribution.
#
# Honesty: the rule is verified by LEAVE-ONE-OUT across train pairs (learn from the rest, must
# reproduce the held-out pair EXACTLY through the full recursion). A lookup cannot trivially pass.
import json, time, numpy as np
import exp_fano_layer as XF
from octonion_arc import A, eq

_rng = np.random.default_rng(7)
VOID = 10                                                   # out-of-grid padding colour
CO = XF.unit(np.concatenate([np.eye(8), _rng.standard_normal((3, 8))]))     # 0..9 colours + VOID -> S^7
ROLE9 = XF.unit(_rng.standard_normal((9, 8)))               # 9 spatial-binding octons (the 3x3 roles)

def cell_field(g):
    """Each cell -> its holographic 3x3 neighbourhood octon (ROLE-bound fano-path superposition)."""
    H, W = g.shape
    pad = np.full((H + 2, W + 2), VOID, int); pad[1:-1, 1:-1] = g
    enc = CO[pad]                                           # (H+2,W+2,8)
    acc = np.zeros((H, W, 8)); k = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            acc = acc + XF.octo_mul(ROLE9[k], enc[1 + dr:1 + dr + H, 1 + dc:1 + dc + W])
            k += 1
    return XF.unit(acc)

class Rule:
    """A cell-scale carrier set: neighbourhood octon -> output colour, queried by nearest-neighbour."""
    def __init__(self, pairs):
        X, y = [], []
        for i, o in pairs:
            X.append(cell_field(i).reshape(-1, 8)); y.append(o.reshape(-1))      # input nbhd -> output colour
            X.append(cell_field(o).reshape(-1, 8)); y.append(o.reshape(-1))      # output nbhd -> itself (fixed pt)
        self.X = np.concatenate(X); self.y = np.concatenate(y).astype(int)
    def apply(self, g):
        q = cell_field(g).reshape(-1, 8)
        nn = (q @ self.X.T).argmax(1)                       # nearest neighbourhood octon in S^7
        return self.y[nn].reshape(g.shape)

def recurse(rule, g, T):
    """TRM refinement: apply the octonionic rule to a fixed point (or T steps)."""
    y = g
    for _ in range(T):
        ny = rule.apply(y)
        if np.array_equal(ny, y): break                     # halt: the answer is a fixed point
        y = ny
    return y

def octo_trm(train, T):
    """Fit the recursive octonionic rule; accept only if LEAVE-ONE-OUT reproduces every held-out pair."""
    pairs = [(A(p["input"]), A(p["output"])) for p in train]
    if len(pairs) < 2 or not all(i.shape == o.shape for i, o in pairs): return None
    for j in range(len(pairs)):                             # leave-one-out generalisation check
        rest = pairs[:j] + pairs[j + 1:]
        if not eq(recurse(Rule(rest), pairs[j][0], T), pairs[j][1]): return None
    full = Rule(pairs)                                      # generalises -> fit on all pairs
    if not all(eq(recurse(full, i, T), o) for i, o in pairs): return None     # sanity: exact on train
    return lambda g: recurse(full, A(g), T)


def solve(task, T=48):
    fn = octo_trm(task["train"], T)
    if fn is None: return None
    return [A(fn(tp["input"])) for tp in task["test"]]


if __name__ == "__main__":
    import sys
    D = "arc_data/"
    split = sys.argv[1] if len(sys.argv) > 1 else "training"
    ch = json.load(open(D + "arc-agi_%s_challenges.json" % split))
    sol = json.load(open(D + "arc-agi_%s_solutions.json" % split))
    for label, T in [("single-shot (T=1)", 1), ("TRM recursion (fixed point)", 48)]:
        t0 = time.time(); solved = 0; n = 0
        for tid, task in ch.items():
            n += 1
            try: preds = solve(task, T)
            except Exception: preds = None
            if preds is None: continue
            if all(eq(preds[i], A(g)) for i, g in enumerate(sol[tid])): solved += 1
        print("%-28s : %3d / %d  %s   (%.0fs)" % (label, solved, n, split, time.time() - t0))
