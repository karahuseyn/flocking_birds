# octonion_trm_layered.py -- a MULTI-LAYER TRM solver, gradient-free.
#
# A flat TRM iterates one operator to a fixed point. The real TRM is DEEP: it refines a latent and an
# answer over several reasoning layers. Here each LAYER is itself a fixed-point recursion of one
# octonionic primitive, and layers stack into a PIPELINE -- so the solver does recursion-of-recursions:
#
#   answer_0 = input
#   for each layer:  answer_{k+1} = primitive_k( answer_k )   # primitive_k recurses to its own fixed pt
#   answer_final     = octonionic recolour( answer_K )         # a closing colour layer (relation octon)
#
# The pipeline is found by an octonion-GUIDED beam search: the latent z is the universe CARRIER octon,
# and a branch is scored by carrier-cosine to the target -- "am I getting closer to the transformation?"
# This is TRM's "refine the latent, then the answer" made gradient-free: z (carrier) steers which layer
# to add; y (grid) is refined layer by layer. Tasks no single operator solves (diagonal gravity = down
# THEN right; gravity THEN recolour) need >=2 layers -- that is the measured pay-off of going deeper.
#
# Honesty: a pipeline is accepted only if it reproduces EVERY train pair exactly (>=2 pairs); the
# shortest such pipeline wins (Occam). No gradients, no backprop.
import json, time, numpy as np
import exp_fano_layer as XF
from octonion_arc import A, eq, bg_color, crop_bbox, DIHEDRAL, objects
from octonion_object_trm import _gravity_step, solve as flat_solve
from octonion_universe import carrier_deep
from octonion_arc_phys import COLOR_OCTON

# ---- parameter-free primitives; each layer is applied as a whole. The gravity / symmetry layers are
# themselves fixed-point RECURSIONS, so a pipeline is a recursion-of-recursions (the deeper TRM). ----
def _grav(d):
    dr, dc = d
    def fn(g):
        g = A(g); bg = bg_color(g)
        for _ in range(sum(g.shape)):
            ng = _gravity_step(g, bg, dr, dc)
            if np.array_equal(ng, g): break
            g = ng
        return g
    return fn

def _symmetrize(g):                                              # fill bg cells from the grid's own mirrors
    g = A(g); bg = bg_color(g)
    for _ in range(4):
        ng = g.copy()
        for f in (np.fliplr, np.flipud):
            m = f(ng); ng = np.where(ng == bg, m, ng)
        if np.array_equal(ng, g): break
        g = ng
    return g

def _keep(which):
    def fn(g):
        g = A(g); bg = bg_color(g); objs = objects(g, bg, True, False)
        if not objs: return g
        o = (max if which == "largest" else min)(objs, key=lambda o: o["size"])
        out = np.full_like(g, bg); rr, cc = np.where(o["mask"]); out[rr, cc] = g[rr, cc]; return out
    return fn

def _tile(ry, rx, mir):                                         # repeat the grid (optionally mirrored)
    def fn(g):
        g = A(g); row = g
        if rx == 2: row = np.concatenate([g, (np.fliplr(g) if mir else g)], 1)
        col = row
        if ry == 2: col = np.concatenate([row, (np.flipud(row) if mir else row)], 0)
        return col
    return fn

PRIMS = ([("grav_down", _grav((1, 0))), ("grav_up", _grav((-1, 0))),
          ("grav_left", _grav((0, -1))), ("grav_right", _grav((0, 1))),
          ("crop", lambda g: crop_bbox(A(g))), ("symmetrize", _symmetrize),
          ("keep_largest", _keep("largest")), ("keep_smallest", _keep("smallest")),
          ("tile_h", _tile(1, 2, 0)), ("tile_v", _tile(2, 1, 0)), ("tile_2x2", _tile(2, 2, 0)),
          ("mirror_h", _tile(1, 2, 1)), ("mirror_v", _tile(2, 1, 1)), ("mirror_2x2", _tile(2, 2, 1))]
         + [(n, (lambda g, _f=f: A(_f(A(g))))) for n, f in DIHEDRAL.items() if n != "identity"])
PRIM = dict(PRIMS)

def octo_recolour(states, outs):
    """A closing colour LAYER: per-colour relation octon R_c = o_out (x) o_in^-1 fit jointly on the
    CURRENT layer's grids -> targets, applied as decode(R_c (x) o(c)). Octonion algebra, exact-or-None."""
    if not all(s.shape == o.shape for s, o in zip(states, outs)): return None
    Rc = {}
    for s, o in zip(states, outs):
        for a, b in zip(s.ravel(), o.ravel()):
            a, b = int(a), int(b); R = XF.octo_mul(COLOR_OCTON[b], XF._inv(COLOR_OCTON[a]))
            if a in Rc and not np.allclose(Rc[a], R, atol=1e-6): return None
            Rc[a] = R
    def fn(g):
        g = A(g); out = np.empty_like(g)
        for c in np.unique(g):
            pred = XF.octo_mul(Rc[int(c)], COLOR_OCTON[int(c)])
            out[g == c] = int((COLOR_OCTON @ pred).argmax())
        return out
    return fn

def _sig(g):
    """A DEEPER octonionic signature: (1) the nested carrier-of-carriers (structure), (2) a colour-mass
    octon. The beam scores a branch by both channels -- a structure-aware, gradient-free heuristic."""
    g = A(g)
    try: c1 = carrier_deep(g)
    except Exception: c1 = np.zeros(8)
    vals, cnts = np.unique(g, return_counts=True)
    c2 = XF.unit((cnts[:, None] * COLOR_OCTON[vals % len(COLOR_OCTON)]).sum(0))
    return c1, c2

def _score(sig, tgt):
    return 0.65 * float(sig[0] @ tgt[0]) + 0.35 * float(sig[1] @ tgt[1])

def _key(cur): return tuple(c.tobytes() + bytes(c.shape) for c in cur)

def layered_solve(train, maxdepth=4, beam=10):
    """Octonion-guided beam search over primitive pipelines; closing octonionic recolour layer. Deeper
    (maxdepth 4, beam 10), the deep carrier-of-carriers as heuristic, with visited-dedup to keep the
    larger search tractable. Returns (fn, depth, ops); accept = exact on every train pair, shortest first."""
    pairs = [(A(p["input"]), A(p["output"])) for p in train]
    if len(pairs) < 2: return None
    ins = [i for i, _ in pairs]; outs = [o for _, o in pairs]
    if any(max(o.shape) > 30 for o in outs): omax = 30
    else: omax = max(max(o.shape) for o in outs)
    tgt = [_sig(o) for o in outs]
    def accept(ops, cur):
        if all(eq(c, o) for c, o in zip(cur, outs)): return _mk(ops, None), len(ops)
        rc = octo_recolour(cur, outs)                               # closing colour layer
        if rc is not None and all(eq(rc(c), o) for c, o in zip(cur, outs)): return _mk(ops, rc), len(ops) + 1
        return None
    def _mk(ops, rc):
        fns = [PRIM[n] for n in ops]
        def fn(g):
            g = A(g)
            for f in fns: g = f(g)
            return rc(g) if rc is not None else g
        return fn
    frontier = [([], ins)]; visited = {_key(ins)}; best = None
    for depth in range(maxdepth + 1):
        for ops, cur in frontier:                                   # try to close at this depth
            a = accept(ops, cur)
            if a is not None and (best is None or a[1] < best[1]): best = (a[0], a[1], ops)
        if best is not None: return best                            # shortest pipeline -> stop
        scored = []
        for ops, cur in frontier:
            for name, fn in PRIMS:
                if ops and ops[-1] == name: continue                # a fixed point is idempotent
                try: ncur = [A(fn(c)) for c in cur]
                except Exception: continue
                if any(max(nc.shape) > omax + 2 for nc in ncur): continue   # cannot reach a <=30 output
                k = _key(ncur)
                if k in visited: continue                           # dedup states (avoids dihedral cycles)
                visited.add(k)
                scored.append((np.mean([_score(_sig(nc), t) for nc, t in zip(ncur, tgt)]), ops + [name], ncur))
        if not scored: return None
        scored.sort(key=lambda x: -x[0]); frontier = [(o, c) for _, o, c in scored[:beam]]
    return None

def solve(task, _both=False):
    """Real solver: flat octonionic operators (recolour / object-correspondence / gravity) UNION the
    multi-layer pipelines. Up to 2 attempts per test input. With _both, also return per-engine preds
    so the layered contribution can be attributed exactly."""
    fp = flat_solve(task)                                           # the depth<=1 octonionic operators
    r = layered_solve(task["train"])                               # the multi-layer pipelines
    layered_fn = r[0] if r is not None else None
    lp = None
    if layered_fn is not None:
        lp = []
        for tp in task["test"]:
            try: lp.append(A(layered_fn(tp["input"])))
            except Exception: lp.append(None)
    preds = []
    for ti, tp in enumerate(task["test"]):
        outs = []
        if fp is not None:
            for p in fp[ti]:
                if not any(eq(p, e) for e in outs): outs.append(p)
        if lp is not None and lp[ti] is not None and not any(eq(lp[ti], e) for e in outs):
            outs.append(lp[ti])
        preds.append(outs[:2])
    return (preds, fp, lp, (r[1:] if r is not None else None)) if _both else preds


if __name__ == "__main__":
    import sys
    from collections import Counter
    D = "arc_data/"; split = sys.argv[1] if len(sys.argv) > 1 else "training"
    ch = json.load(open(D + "arc-agi_%s_challenges.json" % split))
    sol = json.load(open(D + "arc-agi_%s_solutions.json" % split))
    t0 = time.time(); solved = []; deepwin = []
    for tid, task in ch.items():
        try: preds, fp, lp, lay = solve(task, _both=True)
        except Exception: continue
        gold = sol[tid]
        if not all(any(eq(p, A(g)) for p in preds[i]) for i, g in enumerate(gold)): continue
        solved.append(tid)
        flat_ok = fp is not None and all(any(eq(p, A(g)) for p in fp[i]) for i, g in enumerate(gold))
        lay_ok = lp is not None and all(lp[i] is not None and eq(lp[i], A(g)) for i, g in enumerate(gold))
        if lay_ok and not flat_ok and lay is not None and lay[0] >= 2:          # deep pipeline did it alone
            deepwin.append((tid, lay[0], lay[1]))
    print("MULTI-LAYER OCTONIONIC TRM: %d / %d  %s   (%.0fs)" % (len(solved), len(ch), split, time.time() - t0))
    print("  solved ONLY by a DEEP (>=2-layer) pipeline -- no single operator could: %d" % len(deepwin))
    for tid, d, ops in deepwin[:15]:
        tail = " -> recolour" if d > len(ops) else ""
        print("     %s  depth %d : %s%s" % (tid, d, " -> ".join(ops), tail))
    print("  solved:", " ".join(solved[:30]))
