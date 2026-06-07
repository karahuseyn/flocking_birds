# octonion_shapes.py
# ============================================================================
# Tetris-style synthetic blocks and their transformations, carried in the OCTONIONIC
# representation.  Many ARC tasks recognise a SHAPE (a polyomino) and transform it;
# the octonion model's value here is a rotation/reflection-INVARIANT shape descriptor
# that matches a block to its transformed copies.
#
# Octonionic shape descriptor.  For a binary shape, take each filled cell's offset
# (dr,dc) from the centroid and bind it with a 2-D Fano positional code
#       role(dr,dc) = Rrow^dr (x) Rcol^dc           (octonion powers; Artin-associative)
# then superpose:  Phi(shape) = unit( sum_filled role(dr,dc) ).  Over the dihedral
# group D4 of the square this gives an orbit; the lexicographically minimal rounded
# descriptor is a D4-INVARIANT signature -- the same for a block and all its
# rotations/reflections, and (empirically) distinct across distinct blocks.
#
# ARC mechanism: key every object by its octonionic shape signature and learn a
# table  signature -> output colour  (recolour each shape by its identity), fit from
# the task's pairs and accepted only on EXACT reproduction.  Gradient-free, un-named.
# Run `python octonion_shapes.py verify` to validate the descriptor on synthetic
# tetrominoes/pentominoes; `... training` / `... evaluation` to score on ARC.
# ============================================================================
import json, time, sys
from collections import Counter
import numpy as np
import exp_fano_layer as XF
from octonion_arc import A, eq, bg_color, objects

_rng = np.random.default_rng(7)
K = 8                                              # independent octonionic position codes (richer signature)
RROW = [XF.unit(_rng.standard_normal(8)) for _ in range(K)]
RCOL = [XF.unit(_rng.standard_normal(8)) for _ in range(K)]

def _pow(R, n):
    """Octonion power R^n (n integer); powers of one element are associative (Artin)."""
    if n == 0:
        e = np.zeros(8); e[0] = 1.0; return e
    base = R if n > 0 else XF._inv(R); acc = base.copy()
    for _ in range(abs(n) - 1): acc = XF.octo_mul(acc, base)
    return acc

def _descriptor(offsets):
    """Concatenate K octonionic Fano position-code sums -> an 8K-dim shape descriptor."""
    out = []
    for k in range(K):
        acc = np.zeros(8)
        for dr, dc in offsets:
            acc = acc + XF.octo_mul(_pow(RROW[k], int(dr)), _pow(RCOL[k], int(dc)))
        out.append(XF.unit(acc))
    return np.concatenate(out)

_D4 = [lambda r, c: (r, c), lambda r, c: (r, -c), lambda r, c: (-r, c), lambda r, c: (-r, -c),
       lambda r, c: (c, r), lambda r, c: (c, -r), lambda r, c: (-c, r), lambda r, c: (-c, -r)]

def shape_sig(mask):
    """D4-invariant octonionic signature of a binary mask (bytes key).  Offsets are
    centred on the bounding-box centre and doubled, so they are exact integers
    (no truncation) and D4-equivariant."""
    rs, cs = np.where(mask)
    if len(rs) == 0: return b""
    rs = 2 * rs - (rs.max() + rs.min()); cs = 2 * cs - (cs.max() + cs.min())   # integer, range-centred
    best = None
    for t in _D4:
        offs = [t(int(r), int(c)) for r, c in zip(rs, cs)]
        b = np.round(_descriptor(offs), 3).tobytes()
        if best is None or b < best: best = b
    return best

# ----------------------------------------------------------------- synthetic blocks
TETROMINOES = {
    "I": [(0, 0), (0, 1), (0, 2), (0, 3)], "O": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "T": [(0, 0), (0, 1), (0, 2), (1, 1)], "S": [(0, 1), (0, 2), (1, 0), (1, 1)],
    "Z": [(0, 0), (0, 1), (1, 1), (1, 2)], "L": [(0, 0), (1, 0), (2, 0), (2, 1)],
    "J": [(0, 1), (1, 1), (2, 1), (2, 0)],
}

def _mask_of(cells):
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    m = np.zeros((max(rs) + 1, max(cs) + 1), bool)
    for r, c in cells: m[r, c] = True
    return m

def _rand_polyomino(rng, n):
    cells = {(0, 0)}
    while len(cells) < n:
        r, c = list(cells)[rng.integers(len(cells))]
        dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[rng.integers(4)]
        cells.add((r + dr, c + dc))
    rs = min(r for r, _ in cells); cs = min(c for _, c in cells)
    return _mask_of([(r - rs, c - cs) for r, c in cells])

def _verify():
    # (1) D4 invariance: a block and all 8 of its rotations/reflections share one signature
    inv_ok = True
    for name, cells in TETROMINOES.items():
        m = _mask_of(cells); sigs = set()
        for k in range(4):
            r = np.rot90(m, k); sigs.add(shape_sig(r)); sigs.add(shape_sig(np.fliplr(r)))
        if len(sigs) != 1: inv_ok = False; print("  D4 NOT invariant:", name, len(sigs))
    print("[%s] tetromino D4 invariance (one signature per block)" % ("OK" if inv_ok else "XX"))
    # (2) discrimination: under D4 the 7 one-sided tetrominoes collapse to 5 FREE ones
    # (S=Z and L=J are mirror images, one orbit each) -- 5 distinct signatures is correct.
    sigs = {name: shape_sig(_mask_of(cells)) for name, cells in TETROMINOES.items()}
    disc = len(set(sigs.values()))
    print("[%s] 7 tetrominoes -> %d distinct signatures (5 = correct: S=Z, L=J under D4)"
          % ("OK" if disc == 5 else "XX", disc))
    print("    chirality: sig(S)==sig(Z):", sigs["S"] == sigs["Z"], "| sig(L)==sig(J):", sigs["L"] == sigs["J"])
    # (3) scale to dozens: random polyominoes, invariance holds and most are distinguishable
    rng = np.random.default_rng(1); masks = [_rand_polyomino(rng, int(rng.integers(4, 9))) for _ in range(60)]
    inv2 = all(len({shape_sig(np.rot90(m, k)) for k in range(4)} | {shape_sig(np.fliplr(m))}) == 1 for m in masks)
    uniq = len({shape_sig(m) for m in masks})
    print("[%s] 60 random polyominoes: D4-invariant; %d distinct signatures" % ("OK" if inv2 else "XX", uniq))

# ----------------------------------------------------------------- ARC mechanism
def solve(task):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    tests = [A(tp["input"]) for tp in task["test"]]
    for sc in (True, False):
        table = {}; ok = True
        for i, o in pairs:
            if i.shape != o.shape: ok = False; break
            for ob in objects(A(i), bg_color(A(i)), True, sc):
                sig = shape_sig(ob["sm"])
                outvals = set(A(o)[ob["mask"]].tolist())
                if len(outvals) != 1: ok = False; break          # shape recoloured to one colour
                v = outvals.pop()
                if table.get(sig, v) != v: ok = False; break
                table[sig] = v
            if not ok: break
        if not ok: continue
        def emit(g, table=table, sc=sc):
            g = A(g).copy()
            for ob in objects(g, bg_color(g), True, sc):
                sig = shape_sig(ob["sm"])
                if sig not in table: return None
                g[ob["mask"]] = table[sig]
            return g
        if all(eq(emit(i), o) for i, o in pairs):
            pr = [emit(t) for t in tests]
            if all(p is not None for p in pr): return pr
    return None


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "verify":
        print("Octonionic shape-descriptor validation (Tetris blocks):"); _verify()
    else:
        DIR = "arc_data/"; split = cmd
        ch = json.load(open(DIR + "arc-agi_%s_challenges.json" % split))
        sol = json.load(open(DIR + "arc-agi_%s_solutions.json" % split))
        t0 = time.time(); solved = []
        for tid, task in ch.items():
            try: pr = solve(task)
            except Exception: pr = None
            if pr and all(eq(pr[i], A(g)) for i, g in enumerate(sol[tid])): solved.append(tid)
        print("SHAPES (octonionic shape-keyed recolour, gradient-free): %d / %d  %s  (%.0fs)"
              % (len(solved), len(ch), split, time.time() - t0))
        open("/tmp/shapes_%s.ids" % split, "w").write(" ".join(solved))
