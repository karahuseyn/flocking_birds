# octonion_paths.py -- OBJECT-LEVEL emergent paths + an in-task path library (muktesebat).
#
# The vision, made concrete and gradient-free, with NO predefined task-transforms:
#   * every object (connected component) becomes a TSP-unique octonionic signature (obj_octon);
#   * input objects are matched to output objects by NEAREST signature in S^7 (octonionic matching);
#   * for each matched pair we SOLVE the object's PATH from the data: a rigid re-pose of its mask that
#     reproduces the output mask, a per-colour relation (recolour), and a centroid translation. Nothing
#     is enumerated at the task level -- the path emerges from the pair.
#   * MUKTESEBAT: the paths inferred from a task's example pairs form a small library keyed by object
#     signature. Two regimes are tried: a single GLOBAL path shared by all objects, or a per-signature
#     library (objects transform independently) verified leave-one-out. The test objects are then routed
#     through the library and transformed -- the task is solved "instantly" from the acquired paths.
# Honest: accepted only if the paths reproduce every train pair EXACTLY. No gradients, no backprop.
import json, time, numpy as np
import exp_fano_layer as XF
from octonion_arc import A, eq, bg_color, objects, DIHEDRAL
from octonion_object_trm import obj_octon

POSES = list(DIHEDRAL.items())                                   # 8 rigid re-poses of an object's mask

def _objs(g): g = A(g); return objects(g, bg_color(g), True, False)

def _match(ins, outs, ish, osh):
    io = [obj_octon(o, ish) for o in ins]; oo = [obj_octon(o, osh) for o in outs]
    pairs = []; used = set()
    for a in range(len(ins)):
        if not oo: break
        sims = [(-1e9 if b in used else float(io[a] @ oo[b])) for b in range(len(outs))]
        b = int(np.argmax(sims))
        if sims[b] > -1e8: pairs.append((a, b)); used.add(b)
    return pairs, io

def _path(oi, oo):
    """Solve one object's emergent path: (re-pose, colour map, dr, dc) reproducing oo from oi, or None."""
    for name, fn in POSES:
        ps = fn(oi["sub"]); pm = fn(oi["sm"])
        if pm.shape != oo["sm"].shape or not np.array_equal(pm, oo["sm"]): continue
        cm = {}; ok = True
        for a, b in zip(ps[pm].ravel(), oo["sub"][oo["sm"]].ravel()):
            a, b = int(a), int(b)
            if cm.get(a, b) != b: ok = False; break
            cm[a] = b
        if ok:
            return (name, tuple(sorted(cm.items())),
                    oo["bbox"][0] - oi["bbox"][0], oo["bbox"][1] - oi["bbox"][1])
    return None

def _render(g, placements, oshape, bg):
    out = np.full(oshape, bg, int)
    for o, (name, cm, dr, dc) in placements:
        sub = dict(DIHEDRAL)[name](o["sub"]); sm = dict(DIHEDRAL)[name](o["sm"])
        cmap = dict(cm); r0, c0 = o["bbox"][0] + dr, o["bbox"][1] + dc
        for rr in range(sub.shape[0]):
            for cc in range(sub.shape[1]):
                if sm[rr, cc]:
                    R, C = r0 + rr, c0 + cc
                    if 0 <= R < oshape[0] and 0 <= C < oshape[1]:
                        out[R, C] = cmap.get(int(sub[rr, cc]), int(sub[rr, cc]))
    return out

def _oshape(pairs):
    if all(o.shape == i.shape for i, o in pairs): return lambda s: s
    return None

def infer(pairs):
    """Build the task's path library and return an applicator, or None. Tries a global shared path,
    then a per-object-signature library (verified leave-one-out)."""
    osh = _oshape(pairs)
    if osh is None: return None
    bgo = bg_color(pairs[0][1])
    per_pair = []
    for gi, go in pairs:
        ins, outs = _objs(gi), _objs(go)
        m, io = _match(ins, outs, gi.shape, go.shape)
        if len(m) != len(ins) or len(ins) == 0: return None        # need a clean correspondence
        paths = []
        for a, b in m:
            p = _path(ins[a], outs[b])
            if p is None: return None
            paths.append((io[a], ins[a], p))
        per_pair.append(paths)

    def make(lib_lookup):
        def fn(g):
            g = A(g); objs = _objs(g)
            place = [(o, lib_lookup(obj_octon(o, g.shape), o)) for o in objs]
            if any(p is None for _, p in place): return None
            return _render(g, place, osh(g.shape), bgo)
        return fn

    # regime 1: a single GLOBAL path (all objects share re-pose + colour + translation)
    allp = [p for pp in per_pair for _, _, p in pp]
    if len(set(allp)) == 1:
        gp = allp[0]; fn = make(lambda sig, o, _p=gp: _p)
        if all(eq(A(fn(i)), o) for i, o in pairs): return fn

    # regime 2: per-signature library (objects transform independently) -- the muktesebat
    def build(paths_lists):
        keys = []; vals = []
        for pp in paths_lists:
            for sig, _, p in pp: keys.append(sig); vals.append(p)
        K = np.array(keys)
        return (lambda sig, o: vals[int((K @ sig).argmax())]) if keys else (lambda sig, o: None)
    # leave-one-out across pairs so the library must GENERALISE, not memorise
    if len(pairs) >= 2:
        ok = True
        for j in range(len(pairs)):
            lib = build(per_pair[:j] + per_pair[j + 1:]); fn = make(lib)
            if not eq(A(fn(pairs[j][0]) if fn(pairs[j][0]) is not None else np.zeros((1, 1))), pairs[j][1]):
                ok = False; break
        if ok:
            lib = build(per_pair); fn = make(lib)
            if all(fn(i) is not None and eq(A(fn(i)), o) for i, o in pairs): return fn
    return None

def solve(task):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    try: fn = infer(pairs)
    except Exception: return None
    if fn is None: return None
    preds = []
    for tp in task["test"]:
        o = fn(tp["input"])
        if o is None: return None
        preds.append(A(o))
    return preds


if __name__ == "__main__":
    import sys
    D = "arc_data/"; split = sys.argv[1] if len(sys.argv) > 1 else "training"
    ch = json.load(open(D + "arc-agi_%s_challenges.json" % split))
    sol = json.load(open(D + "arc-agi_%s_solutions.json" % split))
    t0 = time.time(); solved = []
    for tid, task in ch.items():
        try: preds = solve(task)
        except Exception: preds = None
        if preds is None: continue
        if all(eq(preds[i], A(g)) for i, g in enumerate(sol[tid])): solved.append(tid)
    print("OBJECT-LEVEL emergent paths (no predefined task-transforms): %d / %d  %s  (%.0fs)"
          % (len(solved), len(ch), split, time.time() - t0))
    print("  solved:", " ".join(solved[:30]))
