# octonion_cells.py
# ============================================================================
# A CELLULAR, RECURSIVE ARC solver built on the OKTONYON NETWORK.  Each base mini-
# network (octonion_network) is one representation CELL -- a grid-state encoded in the
# octon substrate (values in e0, Fano tags for routing).  We hold THOUSANDS of cells
# (a population of competing representations / partial programs) and evolve them in a
# CYCLIC, recursive loop:
#
#   seed   : the input grid in several redundant octon-network representations
#            (the dihedral images -- same value channel, routed tags).
#   expand : each cell spawns children by network operations (Fano-routed geometry +
#            structural moves) -> the population grows to thousands of cells.
#   select : keep the cells closest to the target (error = cell mismatch on the train
#            pairs) -- gradient-free survival of the fittest, population capped.
#   close  : on the fittest cells, try the transformation mechanisms; a cell that
#            reproduces EVERY demonstration exactly is a SOLUTION cell.
#   recurse: iterate for several cycles; surviving distinct solution cells give ARC's
#            two attempts (pass@2).
#
# Coverage is bounded by the operators the cells carry (the discrete-program
# repertoire); the cellular recursion is the parallel search/composition over them.
# Gradient-free throughout.
# ============================================================================
import json, time, sys
import numpy as np
from octonion_arc import A, eq, bg_color, DIHEDRAL
from octonion_layered import FT, single_step
import octonion_wolfram as WF
import octonion_fractal as FR
import octonion_combine as CB
import octonion_objrules as ORU
import octonion_more as MO
import octonion_rays as RY
import octonion_grid2 as G2
import octonion_ffa as FFA

DIH = list(DIHEDRAL.values())
CLOSERS = [WF, FR, CB, ORU, MO, RY, G2, FFA]

def _score(state, outs):
    s = 0.0
    for c, o in zip(state, outs):
        c = A(c); o = A(o)
        if c.shape == o.shape: s += float((c == o).mean())
        else:
            dh = abs(c.shape[0] - o.shape[0]); dw = abs(c.shape[1] - o.shape[1]); s += 0.3 / (1 + dh + dw)
    return s / len(outs)

def _key(state): return tuple(c.tobytes() + repr(c.shape).encode() for c in state)

def _close(cur, outs, cur_t):
    """Try the mechanisms on a cell; return test predictions if it solves all train."""
    pairs = list(zip(cur, outs))
    try:
        fn = single_step(pairs)
        if fn is not None and all(eq(A(fn(c)), o) for c, o in zip(cur, outs)):
            return [A(fn(t)) for t in cur_t]
    except Exception: pass
    task = {"train": [{"input": c, "output": o} for c, o in zip(cur, outs)],
            "test": [{"input": t} for t in cur_t]}
    for M in CLOSERS:
        try: pr = M.solve(task)
        except Exception: pr = None
        if pr is not None and all(p is not None for p in pr): return [A(p) for p in pr]
    return None

def candidates(task, pop=3000, top_close=24, cycles=3):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    ins = [i for i, _ in pairs]; outs = [o for _, o in pairs]; tests = [A(tp["input"]) for tp in task["test"]]
    # seed cells: input in redundant octon-network representations (dihedral images)
    cells = {};
    for d in DIH:
        st = tuple(A(d(i)) for i in ins); tt = tuple(A(d(t)) for t in tests); k = _key(st)
        if k not in cells: cells[k] = (st, tt, [d])
    sols = []; seen_sol = set()
    for cyc in range(cycles + 1):
        ranked = sorted(cells.values(), key=lambda v: -_score(v[0], outs))
        for st, tt, prog in ranked[:top_close]:                 # close the fittest cells
            pr = _close(st, outs, tt)
            if pr is not None:
                key = tuple(p.tobytes() + repr(p.shape).encode() for p in pr)
                if key not in seen_sol: seen_sol.add(key); sols.append(pr)
                if len(sols) >= 2: return sols
        if cyc == cycles: break
        # expand the population by network operations (Fano-routed geometry + structural)
        for st, tt, prog in ranked[:max(top_close, pop // 40)]:
            for name, ft in FT:
                try: nst = tuple(A(ft(c)) for c in st); ntt = tuple(A(ft(t)) for t in tt)
                except Exception: continue
                if any(x is None or x.size == 0 or max(x.shape) > 45 for x in nst): continue
                k = _key(nst)
                if k in cells: continue
                cells[k] = (nst, ntt, prog + [ft])
                if len(cells) >= pop: break
            if len(cells) >= pop: break
    return sols

def solve(task):
    cs = candidates(task); return cs[0] if cs else None


if __name__ == "__main__":
    DIR = "arc_data/"; split = sys.argv[1] if len(sys.argv) > 1 else "training"
    ch = json.load(open(DIR + "arc-agi_%s_challenges.json" % split))
    sol = json.load(open(DIR + "arc-agi_%s_solutions.json" % split))
    t0 = time.time(); s1 = []; s2 = []
    for tid, task in ch.items():
        try: cs = candidates(task)
        except Exception: cs = []
        if not cs: continue
        if all(eq(cs[0][i], A(g)) for i, g in enumerate(sol[tid])): s1.append(tid)
        for pr in cs[:2]:
            if all(eq(pr[i], A(g)) for i, g in enumerate(sol[tid])): s2.append(tid); break
    print("OCTONION CELLS (cellular recursive, %s): pass@1 %d/%d, pass@2 %d/%d  (%.0fs)"
          % (split, len(s1), len(ch), len(s2), len(ch), time.time() - t0))
    open("/tmp/cells_%s.ids" % split, "w").write(" ".join(s2))
