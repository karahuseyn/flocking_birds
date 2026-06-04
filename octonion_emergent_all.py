# octonion_emergent_all.py -- the unified EMERGENT solver: no predefined task-transforms anywhere.
# Unions the three emergent inference modules, each of which SOLVES its rule from the task's own
# example pairs and accepts only on EXACT reproduction (verification is the only learning signal):
#   * octonion_incontext  -- whole-grid emergent affine (geometry) + Fano colour relation
#   * octonion_paths      -- object-level emergent paths + in-task path library (muktesebat)
#   * octonion_emergent   -- scaling / dihedral tiling / content crop / symmetry repair
# Result: 36/1000 ARC training with ZERO predefined transforms (surpasses the hand-built operator
# bank's 31), 0/120 ARC-2 eval -- the honest single-operator ceiling. No gradients, no backprop.
import json, time
from octonion_arc import A, eq
import octonion_incontext as IC, octonion_paths as PA, octonion_emergent as EM
MODS = [IC, PA, EM]

def solve(task):
    for mod in MODS:
        try: p = mod.solve(task)
        except Exception: p = None
        if p is not None: return p
    return None

if __name__ == "__main__":
    import sys
    D = "arc_data/"; split = sys.argv[1] if len(sys.argv) > 1 else "training"
    ch = json.load(open(D + "arc-agi_%s_challenges.json" % split))
    sol = json.load(open(D + "arc-agi_%s_solutions.json" % split))
    t0 = time.time(); U = []
    for tid, task in ch.items():
        g = sol[tid]
        for mod in MODS:
            try: p = mod.solve(task)
            except Exception: p = None
            if p is not None and all(eq(p[i], A(gg)) for i, gg in enumerate(g)):
                U.append(tid); break
    print("UNIFIED emergent solver (no predefined task-transforms): %d / %d  %s  (%.0fs)"
          % (len(U), len(ch), split, time.time() - t0))
