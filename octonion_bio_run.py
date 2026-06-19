# octonion_bio_run.py -- train/evaluate the (optionally biologically-inspired) OctoNet.
# Set OCTO_BIO=1 for the brain-inspired encoding (reciprocal conjugate I/O, Dale inhibitory cells,
# lateral inhibition). Model size (nodes, K) is unchanged; only the synthetic count / encoding vary.
import os, sys, json, time
import numpy as np
import octonion_net as N
from octonion_arc import A, eq

def arc_count(net, split):
    ch = json.load(open("arc_data/arc-agi_%s_challenges.json" % split))
    sol = json.load(open("arc_data/arc-agi_%s_solutions.json" % split))
    solved = []
    for tid, task in ch.items():
        preds, used = N.solve(task, net)
        if preds is None: continue
        if all(eq(preds[i], A(g)) for i, g in enumerate(sol[tid])): solved.append(tid)
    return len(solved), len(ch), solved

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "train":                                          # chunked + resumable (survives kills)
        path = sys.argv[2]; Nex = int(sys.argv[3]); nodes = int(os.environ.get("NODES", "4096"))
        chunk = int(os.environ.get("CHUNK", "1000000")); side = path + ".done"
        if os.path.exists(path) and os.path.exists(side):
            net = N.OctoNet.load(path); done = int(open(side).read() or "0")
            print("RESUME BIO=%s %s done=%d/%d" % (N.BIO, path, done, Nex), flush=True)
        else:
            net = N.OctoNet(n=nodes); done = 0
            print("FRESH BIO=%s nodes=%d synthetic=%d" % (N.BIO, nodes, Nex), flush=True)
        while done < Nex:
            N.train(net, min(chunk, Nex - done), seed=1 + done, seed_nodes=(done == 0))
            done += min(chunk, Nex - done); net.save(path); open(side, "w").write(str(done))
            print("  checkpoint %d/%d saved" % (done, Nex), flush=True)
        net.label(); net.save(path)
        print("DONE %s | synth acc (chance %.4f): %.4f" % (path, 1 / N.NC, N.eval_synth(net, 50000)), flush=True)
    elif mode == "arc":
        path = sys.argv[2]; net = N.OctoNet.load(path)
        print("BIO=%s  net=%s  synth acc: %.4f" % (N.BIO, path, N.eval_synth(net, 30000)), flush=True)
        for split in ("training", "evaluation"):
            t0 = time.time(); n, tot, ids = arc_count(net, split)
            print("  ARC %s: %d / %d  (%.0fs)" % (split, n, tot, time.time() - t0), flush=True)
            open("/tmp/bio_%s_%s.ids" % (os.path.basename(path), split), "w").write(" ".join(ids))
