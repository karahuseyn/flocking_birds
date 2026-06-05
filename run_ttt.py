import json, time, sys, numpy as np
import octonion_ttt as M
from octonion_arc import A, eq

split = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
out = sys.argv[3] if len(sys.argv) > 3 else "/tmp/ttt_%s.prog" % split
ch = json.load(open("arc_data/arc-agi_%s_challenges.json" % split))
sol = json.load(open("arc_data/arc-agi_%s_solutions.json" % split))
items = list(ch.items())[:n] if n else list(ch.items())
f = open(out, "w"); solved = []; t0 = time.time()
for k, (tid, task) in enumerate(items):
    # cell budget: skip pathological sizes to avoid OOM
    allg = [A(p["input"]) for p in task["train"]] + [A(p["output"]) for p in task["train"]] + [A(t["input"]) for t in task["test"]]
    S = min(30, max(max(g.shape) for g in allg)); npairs = len(task["train"])
    if S * S * npairs * 8 > 250000:                              # ~ too big for numpy backprop here
        f.write("%d %s SKIP(size S=%d np=%d)\n" % (k, tid, S, npairs)); f.flush(); continue
    try:
        p = M.solve(task, steps=200)
        ok = p is not None and all(eq(p[i], A(g)) for i, g in enumerate(sol[tid]))
    except Exception as e:
        ok = False; p = "ERR:%r" % e
    if ok: solved.append(tid)
    f.write("%d/%d %s %s  cum=%d  %.0fs\n" % (k + 1, len(items), tid, "SOLVED" if ok else "no", len(solved), time.time() - t0)); f.flush()
f.write("DONE %s: %d/%d solved (%.0fs)  %s\n" % (split, len(solved), len(items), time.time() - t0, " ".join(solved)))
f.flush(); f.close()
print("DONE", split, len(solved), "/", len(items))
