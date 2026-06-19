# octonion_morphpath.py
# ============================================================================
# MORPH PATH: insert intermediate states between input and output and learn the
# step-by-step transformation, TRM-style (its ~16 latent states), in our octonionic
# recursive system.  The intermediates are not given, so we CONSTRUCT them as a
# spatially-coherent propagation wave: the cells that must change are revealed from
# the already-correct ("stable") region inward, one BFS layer per step.  Each step is
# therefore a LOCAL change (only cells next to already-settled cells move) -- a
# propagation a local rule can learn.  We learn ONE per-step octonionic local rule
# over ALL (state_t -> state_{t+1}) transitions of all pairs (deep supervision across
# the whole path), then ITERATE it to a fixed point; accept only if the iteration
# reproduces every demonstration's endpoint EXACTLY, then evolve the test input.
# Gradient-free; the path is the chain of states, the rule is octonionic + exact.
# ============================================================================
import json, time, sys
import numpy as np
from collections import deque
from octonion_arc import A, eq, bg_color

def _states(gin, gout, max_steps=40):
    """Self-consistent propagation wave: a change-cell flips to its target value when a
    neighbour ALREADY shows that value (synchronous, per step).  Yields g0=in,...,gN.
    Returns None if the wave cannot complete all changes (task is not a propagation)."""
    gin = A(gin); gout = A(gout); H, W = gin.shape
    D = gin != gout
    if not D.any(): return [gin]
    g = gin.copy(); seq = [g.copy()]
    for _ in range(max_steps):
        nxt = g.copy(); moved = False
        for r in range(H):
            for c in range(W):
                if not D[r, c] or g[r, c] == gout[r, c]: continue
                tgt = gout[r, c]
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and g[rr, cc] == tgt:
                        nxt[r, c] = tgt; moved = True; break
        if not moved: break
        g = nxt; seq.append(g.copy())
    if not np.array_equal(g, gout): return None                # wave didn't reach the output
    return seq

# ---- per-step octonionic local rule: exact LUT on the 3x3 patch of the current state ----
def _patch_keys(g):
    g = A(g); P = np.pad(g, 1, constant_values=-1); H, W = g.shape
    return [[P[r:r + 3, c:c + 3].tobytes() for c in range(W)] for r in range(H)]

def _learn_step_rule(transitions):
    lut = {}
    for a, b in transitions:
        a = A(a); b = A(b); K = _patch_keys(a)
        for r in range(a.shape[0]):
            for c in range(a.shape[1]):
                k = K[r][c]; v = int(b[r, c])
                if lut.get(k, v) != v: return None              # rule must be a function
                lut[k] = v
    return lut

def _step(g, lut):
    g = A(g); K = _patch_keys(g); out = g.copy()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            v = lut.get(K[r][c])
            if v is not None: out[r, c] = v
    return out

def _evolve(g, lut, max_iter=40):
    g = A(g)
    for _ in range(max_iter):
        n = _step(g, lut)
        if np.array_equal(n, g): break
        g = n
    return g

def solve(task):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    tests = [A(tp["input"]) for tp in task["test"]]
    if not all(i.shape == o.shape for i, o in pairs): return None
    # build the morph paths and collect every consecutive transition (deep supervision)
    transitions = []
    for i, o in pairs:
        seq = _states(i, o)
        if seq is None or len(seq) < 2: return None            # not a propagation morph
        for t in range(len(seq) - 1): transitions.append((seq[t], seq[t + 1]))
    lut = _learn_step_rule(transitions)
    if lut is None: return None
    # the iterated rule must reproduce every endpoint exactly
    if not all(eq(_evolve(i, lut), o) for i, o in pairs): return None
    return [A(_evolve(t, lut)) for t in tests]


if __name__ == "__main__":
    DIR = "arc_data/"; split = sys.argv[1] if len(sys.argv) > 1 else "training"
    ch = json.load(open(DIR + "arc-agi_%s_challenges.json" % split))
    sol = json.load(open(DIR + "arc-agi_%s_solutions.json" % split))
    t0 = time.time(); solved = []
    for tid, task in ch.items():
        try: pr = solve(task)
        except Exception: pr = None
        if pr and all(eq(pr[i], A(g)) for i, g in enumerate(sol[tid])): solved.append(tid)
    print("MORPH-PATH (intermediate states + iterated local rule): %d / %d  %s  (%.0fs)"
          % (len(solved), len(ch), split, time.time() - t0))
    open("/tmp/morphpath_%s.ids" % split, "w").write(" ".join(solved))
