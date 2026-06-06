# octonion_deepcompose.py
# ============================================================================
# DEEP multi-mechanism composition, gradient-free.  The individual mechanisms each
# score 0/120 on eval; the open question is whether COMPOSING them -- a structural
# forward move (or several) followed by any mechanism as a closer -- reaches tasks
# none of them reaches alone.  This is the depth lever applied to the full vocabulary.
#
# Error-guided beam over object-states (grids), carried in lockstep for the train
# pairs AND the test input(s):
#   * FORWARD moves (octonion_layered.FT): dihedral, crop, upscale, tile, mirror.
#   * CLOSERS = the WHOLE mechanism library, invoked via each module's own solve():
#     at a node we build a synthetic task {train: cur->out, test: cur_test} and call
#     octonion_layered.single_step, octonion_wolfram, octonion_fractal,
#     octonion_combine, octonion_objsel, octonion_symrepair.  Each verifies EXACTLY
#     on the (current-state -> target) pairs, so a non-None return is a valid composed
#     solution; we apply it to the carried test state.
# A value heuristic (cell agreement + size proximity) ranks states so the recurrence
# can go deep while staying tractable.  No gradients, no backprop, nothing named.
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

MODS = [WF, FR, CB, OS, SR, ORU]                  # full mechanism vocabulary as closers

def _close(cur, outs, cur_tests):
    pairs = list(zip(cur, outs))
    try:
        fn = single_step(pairs)
        if fn is not None and all(eq(A(fn(c)), o) for c, o in zip(cur, outs)):
            return [A(fn(t)) for t in cur_tests]
    except Exception: pass
    task = {"train": [{"input": c, "output": o} for c, o in zip(cur, outs)],
            "test": [{"input": t} for t in cur_tests]}
    for M in MODS:
        try: pr = M.solve(task)
        except Exception: pr = None
        if pr is not None and all(p is not None for p in pr):
            return [A(p) for p in pr]
    return None

def _score(cur, outs):
    s = 0.0
    for c, o in zip(cur, outs):
        c = A(c); o = A(o)
        if c.shape == o.shape: s += float((c == o).mean())
        else:
            dh = abs(c.shape[0] - o.shape[0]); dw = abs(c.shape[1] - o.shape[1]); s += 0.4 / (1 + dh + dw)
    return s / len(outs)

def _key(cur):
    return tuple(c.tobytes() + repr(c.shape).encode() for c in cur)

def solve(task, beam=12, cycles=3):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    ins = [i for i, _ in pairs]; outs = [o for _, o in pairs]
    tests = [A(tp["input"]) for tp in task["test"]]
    beam_states = [(_score(ins, outs), ins, tests, [])]
    seen = {_key(ins)}
    for cycle in range(cycles + 1):
        for sc, cur, cur_t, prog in beam_states:
            pr = _close(cur, outs, cur_t)
            if pr is not None: return pr
        if cycle == cycles: break
        cand = []
        for sc, cur, cur_t, prog in beam_states:
            for name, ft in FT:
                try:
                    nxt = [A(ft(c)) for c in cur]; nxt_t = [A(ft(t)) for t in cur_t]
                except Exception: continue
                if any(n is None or n.size == 0 or max(n.shape) > 45 for n in nxt): continue
                if all(np.array_equal(n, c) for n, c in zip(nxt, cur)): continue
                k = _key(nxt)
                if k in seen: continue
                seen.add(k); cand.append((_score(nxt, outs), nxt, nxt_t, prog + [name]))
        if not cand: break
        cand.sort(key=lambda t: -t[0]); beam_states = cand[:beam]
    return None


if __name__ == "__main__":
    DIR = "arc_data/"; split = sys.argv[1] if len(sys.argv) > 1 else "evaluation"
    beam = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    cycles = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    ch = json.load(open(DIR + "arc-agi_%s_challenges.json" % split))
    sol = json.load(open(DIR + "arc-agi_%s_solutions.json" % split))
    t0 = time.time(); solved = []
    for tid, task in ch.items():
        try: pr = solve(task, beam, cycles)
        except Exception: pr = None
        if pr and all(eq(pr[i], A(g)) for i, g in enumerate(sol[tid])): solved.append(tid)
    print("DEEP composition (beam %d, cycles %d): %d / %d  %s  (%.0fs)"
          % (beam, cycles, len(solved), len(ch), split, time.time() - t0))
    open("/tmp/deepcompose_%s.ids" % split, "w").write(" ".join(solved))
