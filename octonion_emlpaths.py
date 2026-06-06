# octonion_emlpaths.py
# ============================================================================
# EML's idea (arXiv:2603.21852, Figs 1-2) applied to ARC transformations.  The
# paper shows every elementary function is a binary tree over ONE operator, that the
# SAME object has many equivalent trees (Fig 2), and that objects interconvert along
# short composition paths -- a "phylogenetic" reachability network bootstrapped from
# {1} (Fig 1).  Here the objects are GRID-STATES and the single operator is function
# COMPOSITION (∘); a small alphabet of atomic grid ops plays the role of the primitive
# constant set.
#
#   * NODES   = joint grid-states (the representation of all train inputs at once).
#   * EDGES   = atoms (the converter paths).
#   * EQUIVALENCE = two composition trees are identified when they act IDENTICALLY on
#     the train inputs (observational equivalence -- exactly EML's value-vector dedup
#     in Fig 2).  The search therefore enumerates equivalence CLASSES, not trees, so
#     it goes deep cheaply -- the bootstrapping that makes the phylogenetic tree small.
#   * GOAL    = a node from which a CLOSER (any mechanism) reproduces every output
#     EXACTLY.  The composed path is then replayed on the carried test state.
#
# Unlike the greedy beam (octonion_deepcompose, which added 0), this is an EXACT
# bottom-up enumeration over equivalence classes with no heuristic pruning, over an
# alphabet ENRICHED with the new mechanisms as intermediate atoms (gravity, denoise,
# connect) -- combining the two levers (more mechanisms + composition).  Gradient-free,
# no backprop, nothing named.  Think of it as a representation network whose paths
# convert any object into any other and thereby locate the test output.
# ============================================================================
import json, time, sys
import numpy as np
from octonion_arc import A, eq
from octonion_layered import FT, single_step
import octonion_wolfram as WF
import octonion_fractal as FR
import octonion_combine as CB
import octonion_objsel as OS
import octonion_symrepair as SR
import octonion_objrules as ORU
import octonion_more as MO
import octonion_rays as RY

# ---- alphabet: structural atoms (FT) enriched with new intermediate mechanisms ----
ATOMS = list(FT) + [
    ("grav_down", lambda g: ORU._grav(g, "down")), ("grav_up", lambda g: ORU._grav(g, "up")),
    ("grav_left", lambda g: ORU._grav(g, "left")), ("grav_right", lambda g: ORU._grav(g, "right")),
    ("denoise", lambda g: RY._denoise_grid(g)), ("connect", lambda g: RY._connect_grid(g)),
]

CLOSER_MODS = [WF, FR, CB, OS, SR, ORU, MO, RY]

def _colormap(pairs):
    cm = {}
    for i, o in pairs:
        if i.shape != o.shape: return None
        for a, b in zip(i.ravel().tolist(), o.ravel().tolist()):
            if cm.get(a, b) != b: return None
            cm[a] = b
    return lambda g: np.vectorize(lambda c: cm.get(int(c), int(c)))(A(g))

def _close(cur, outs, cur_tests):
    pairs = list(zip(cur, outs))
    same_shape = all(c.shape == o.shape for c, o in zip(cur, outs))
    # exact already (free), or a cheap shape-preserving closer
    if same_shape:
        if all(np.array_equal(c, o) for c, o in zip(cur, outs)):
            return [A(t) for t in cur_tests]
        for mk in (_colormap, lambda p: single_step(p), lambda p: WF.rule_solver(p)):
            try:
                fn = mk(pairs)
                if fn is not None and all(eq(A(fn(c)), o) for c, o in zip(cur, outs)):
                    return [A(fn(t)) for t in cur_tests]
            except Exception: pass
    return None

def _sig(states):                                # observational-equivalence signature (train states)
    return tuple(s.tobytes() + repr(s.shape).encode() for s in states)

def solve(task, depth=3, cap=4000):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    ins = [i for i, _ in pairs]; outs = [o for _, o in pairs]
    tests = [A(tp["input"]) for tp in task["test"]]
    # frontier nodes: (train_states, test_states, program)
    frontier = {_sig(ins): (ins, tests, [])}
    seen = set(frontier)
    for d in range(depth + 1):
        nxt = {}
        for sg, (cur, cur_t, prog) in frontier.items():
            pr = _close(cur, outs, cur_t)            # try to close at this representation
            if pr is not None: return pr
            if d == depth: continue
            for name, ft in ATOMS:                   # expand along converter paths (atoms)
                try:
                    ns = [A(ft(c)) for c in cur]; nt = [A(ft(t)) for t in cur_t]
                except Exception: continue
                if any(x is None or x.size == 0 or max(x.shape) > 45 for x in ns): continue
                if all(np.array_equal(a, b) for a, b in zip(ns, cur)): continue
                k = _sig(ns)
                if k in seen: continue
                seen.add(k); nxt[k] = (ns, nt, prog + [name])
                if len(seen) > cap: break
            if len(seen) > cap: break
        frontier = nxt
        if not frontier: break
    return None


if __name__ == "__main__":
    DIR = "arc_data/"; split = sys.argv[1] if len(sys.argv) > 1 else "training"
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ch = json.load(open(DIR + "arc-agi_%s_challenges.json" % split))
    sol = json.load(open(DIR + "arc-agi_%s_solutions.json" % split))
    t0 = time.time(); solved = []
    for tid, task in ch.items():
        try: pr = solve(task, depth)
        except Exception: pr = None
        if pr and all(eq(pr[i], A(g)) for i, g in enumerate(sol[tid])): solved.append(tid)
    print("EML-PATHS (equivalence-class composition network, depth %d): %d / %d  %s  (%.0fs)"
          % (depth, len(solved), len(ch), split, time.time() - t0))
    open("/tmp/emlpaths_%s.ids" % split, "w").write(" ".join(solved))
