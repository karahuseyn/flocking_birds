# octonion_objsel.py
# ============================================================================
# Object-centric mechanism, gradient-free, no predefined named transforms.  The
# dominant missing family in ARC (both splits): segment the grid into objects, SELECT
# one by a property, and emit a function of it.  Everything is parametric and fit
# from the task's own pairs, accepted only on EXACT reproduction of every demo.
#
#   segmentation  : connected components, searched over {4/8-connectivity} x
#                   {same-colour / any-colour} -- no fixed choice.
#   selection     : an un-named family of object rankings / predicates --
#                   extof {size, bbox-area, #colours}, the ODD-ONE-OUT (unique under
#                   a signature: D4-canonical shape / colour / size), or the MAJORITY.
#   action        : emit the object's bbox crop, its binary mask, or a recolouring
#                   of the crop (colour map learned from data).
#
# This is the object-level analogue of the cellwise mechanisms: a new MECHANISM, the
# proven lever, rather than another parameterisation of one we already have.
# ============================================================================
import json, time, sys
from collections import Counter
import numpy as np
from octonion_arc import A, eq, bg_color, objects

def _objs(g, mode):
    diag, sc = mode
    return objects(A(g), bg_color(A(g)), diag, sc)

def _d4_shape_sig(o):
    m = o["sm"].astype(int)
    cand = []
    for k in range(4):
        r = np.rot90(m, k)
        cand.append(r.tobytes() + bytes(r.shape)); cand.append(np.fliplr(r).tobytes() + bytes(np.fliplr(r).shape))
    return min(cand)                                   # rotation/reflection-invariant shape key

def _area(o): return o["h"] * o["w"]

# ---- selection criteria: list of objs -> chosen obj or None (must be unambiguous) ----
def _extreme(key, largest):
    def f(objs):
        if len(objs) < 2: return None
        vals = [key(o) for o in objs]
        idx = int(np.argmax(vals) if largest else np.argmin(vals))
        if vals.count(vals[idx]) != 1: return None     # extremum must be unique
        return objs[idx]
    return f

def _odd(sig):
    def f(objs):
        if len(objs) < 3: return None
        sigs = [sig(o) for o in objs]; cnt = Counter(sigs)
        singles = [o for o, s in zip(objs, sigs) if cnt[s] == 1]
        return singles[0] if len(singles) == 1 else None
    return f

def _majority(sig):
    def f(objs):
        if len(objs) < 3: return None
        sigs = [sig(o) for o in objs]; cnt = Counter(sigs)
        (top, n), = cnt.most_common(1)
        winners = [o for o, s in zip(objs, sigs) if s == top]
        return winners[0] if n > 1 and len(winners) == 1 else None
    return f

CRITERIA = ([_extreme(lambda o: o["size"], True), _extreme(lambda o: o["size"], False),
             _extreme(_area, True), _extreme(_area, False),
             _extreme(lambda o: o["ncolors"], True)]
            + [_odd(s) for s in (_d4_shape_sig, lambda o: o["color"], lambda o: o["size"])]
            + [_majority(s) for s in (_d4_shape_sig, lambda o: o["color"])])

# ---- actions: chosen obj -> output grid (+ a fitted colour map where needed) ----
def _colormap(src, dst):
    if src.shape != dst.shape: return None
    cm = {}
    for a, b in zip(src.ravel().tolist(), dst.ravel().tolist()):
        if cm.get(a, b) != b: return None
        cm[a] = b
    return cm

def _try(pairs, mode, sel):
    """Return an action fn (obj->grid) reproducing all pairs, or None."""
    chosen = []
    for i, o in pairs:
        objs = _objs(i, mode); c = sel(objs)
        if c is None: return None
        chosen.append(c)
    outs = [o for _, o in pairs]
    # action A: bbox crop with original colours
    if all(eq(c["sub"], o) for c, o in zip(chosen, outs)):
        return ("sub", None)
    # action B: binary mask painted with the object's colour
    if all(c["color"] is not None and eq(c["sm"].astype(int) * c["color"], o) for c, o in zip(chosen, outs)):
        return ("mask", None)
    # action C: recolour the crop by a single consistent colour map
    cm = {}
    ok = True
    for c, o in zip(chosen, outs):
        if c["sub"].shape != o.shape: ok = False; break
        m = _colormap(c["sub"], o)
        if m is None or any(cm.get(k, v) != v for k, v in m.items()): ok = False; break
        cm.update(m)
    if ok and all(eq(np.vectorize(lambda x: cm.get(int(x), int(x)))(c["sub"]), o) for c, o in zip(chosen, outs)):
        return ("recolor", cm)
    return None

MODES = [(True, True), (True, False), (False, True), (False, False)]

def solve(task):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    tests = [A(tp["input"]) for tp in task["test"]]
    for mode in MODES:
        for sel in CRITERIA:
            act = _try(pairs, mode, sel)
            if act is None: continue
            kind, cm = act
            def emit(g, mode=mode, sel=sel, kind=kind, cm=cm):
                c = sel(_objs(g, mode))
                if c is None: return None
                if kind == "sub": return A(c["sub"])
                if kind == "mask": return A(c["sm"].astype(int) * c["color"])
                return A(np.vectorize(lambda x: cm.get(int(x), int(x)))(c["sub"]))
            try:
                pr = [emit(t) for t in tests]
            except Exception:
                pr = None
            if pr is not None and all(p is not None for p in pr):
                return pr
    return None


if __name__ == "__main__":
    DIR = "arc_data/"; split = sys.argv[1] if len(sys.argv) > 1 else "training"
    ch = json.load(open(DIR + "arc-agi_%s_challenges.json" % split))
    sol = json.load(open(DIR + "arc-agi_%s_solutions.json" % split))
    t0 = time.time(); solved = []
    for tid, task in ch.items():
        try: pr = solve(task)
        except Exception: pr = None
        if pr and all(eq(pr[i], A(g)) for i, g in enumerate(sol[tid])): solved.append(tid)
    print("OBJECT SELECTION (gradient-free): %d / %d  %s  (%.0fs)"
          % (len(solved), len(ch), split, time.time() - t0))
    open("/tmp/objsel_%s.ids" % split, "w").write(" ".join(solved))
