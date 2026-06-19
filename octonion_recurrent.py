# octonion_recurrent.py
# ============================================================================
# A TRM-style RECURRENT solver at OBJECT scale, gradient-free.  Where octonion_trm.py
# recurses at the cell scale (a holographic neighbourhood rule iterated to a fixed
# point), this one realises the user's broader aim: a structure in which every
# "object" of our space transforms into every other.
#
# TRM (Tiny Recursive Model) keeps a running answer and refines it over several
# cycles by re-applying one small core; TRM trains that core by backprop.  We keep
# the recurrence and DROP the backprop -- at each cycle the per-step rule is not
# learned but SOLVED from the current (state -> target) pairs, and the loop is driven
# by an ERROR signal.
#
# The operator set induces a reachability graph on object-states (grids); the
# recurrent search finds a PATH from the input object to the output object:
#   * FORWARD moves (structural, target-free): the dihedral group, crop, upscale,
#     tile, mirror -- change the object without looking at the target.
#   * CLOSERS (emergent, solved from data): colour map, octonionic affine + Fano
#     recolour, the learned CA rule, the symmetry-aware Wolfram CA (D4-equivariant),
#     and the substitution / fractal rewrite -- accepted only if they map the
#     current state to the target EXACTLY on every demonstration.
#
# Unlike octonion_layered's blind depth-2 DFS, the loop is an ERROR-GUIDED BEAM: a
# value heuristic (cell agreement with the target, plus size proximity) ranks states
# so the recurrence can go deeper while staying tractable -- the gradient-free
# analogue of TRM's iterative latent refinement.  No backprop; verification on all
# demonstrations is the only signal.
# ============================================================================
import json, time, sys
import numpy as np
from octonion_arc import A, eq
from octonion_layered import FT, single_step
import octonion_wolfram as W
import octonion_fractal as FR

def _closer(pairs):
    """The strongest finishing rule, solved from (state -> target); or None."""
    fn = single_step(pairs)                       # dihedral/colour/affine/CA/structural (layered)
    if fn is not None: return fn
    try:
        f = W.rule_solver(pairs)                  # symmetry-aware Wolfram CA (D4 / totalistic)
        if f is not None: return f
    except Exception: pass
    try:
        f = FR.substitution_solver(pairs)         # substitution / fractal (shape-growing)
        if f is not None: return f
    except Exception: pass
    return None

def _score(cur, outs):
    """Value heuristic in [0,1]: mean cell agreement where shapes match, else a
    size-proximity reward -- the error signal that guides the recurrence."""
    s = 0.0
    for c, o in zip(cur, outs):
        c = A(c); o = A(o)
        if c.shape == o.shape: s += float((c == o).mean())
        else:
            dh = abs(c.shape[0] - o.shape[0]); dw = abs(c.shape[1] - o.shape[1])
            s += 0.4 / (1 + dh + dw)
    return s / len(outs)

def _key(cur):
    return tuple(c.tobytes() + repr(c.shape).encode() for c in cur)

def search(ins, outs, beam=8, cycles=4):
    cur0 = [A(i) for i in ins]
    beam_states = [(_score(cur0, outs), cur0, [])]
    seen = {_key(cur0)}
    for cycle in range(cycles + 1):
        # (a) try to CLOSE every state in the beam (read out the refined "answer")
        for sc, cur, prog in beam_states:
            cl = _closer(list(zip(cur, outs)))
            if cl is not None and all(eq(A(cl(c)), o) for c, o in zip(cur, outs)):
                return prog + [cl]
        if cycle == cycles: break
        # (b) RECUR: expand by every forward move, keep the best `beam` by the heuristic
        cand = []
        for sc, cur, prog in beam_states:
            for name, ft in FT:
                try: nxt = [A(ft(c)) for c in cur]
                except Exception: continue
                if any(n is None or n.size == 0 or max(n.shape) > 45 for n in nxt): continue
                if all(np.array_equal(n, c) for n, c in zip(nxt, cur)): continue
                k = _key(nxt)
                if k in seen: continue
                seen.add(k); cand.append((_score(nxt, outs), nxt, prog + [ft]))
        if not cand: break
        cand.sort(key=lambda t: -t[0]); beam_states = cand[:beam]
    return None

def solve(task, beam=8, cycles=4):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    ins = [i for i, _ in pairs]; outs = [o for _, o in pairs]
    prog = search(ins, outs, beam, cycles)
    if prog is None: return None
    def apply(g):
        g = A(g)
        for f in prog: g = A(f(g))
        return g
    return [apply(tp["input"]) for tp in task["test"]]


if __name__ == "__main__":
    DIR = "arc_data/"; split = sys.argv[1] if len(sys.argv) > 1 else "training"
    beam = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    cycles = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    ch = json.load(open(DIR + "arc-agi_%s_challenges.json" % split))
    sol = json.load(open(DIR + "arc-agi_%s_solutions.json" % split))
    t0 = time.time(); solved = []
    for tid, task in ch.items():
        try: pr = solve(task, beam, cycles)
        except Exception: pr = None
        if pr and all(eq(pr[i], A(g)) for i, g in enumerate(sol[tid])): solved.append(tid)
    print("RECURRENT object-scale (beam %d, cycles %d, gradient-free): %d / %d  %s  (%.0fs)"
          % (beam, cycles, len(solved), len(ch), split, time.time() - t0))
    open("/tmp/recurrent_%s.ids" % split, "w").write(" ".join(solved))
