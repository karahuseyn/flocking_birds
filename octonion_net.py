# octonion_net.py -- a large, persistent, GRADIENT-FREE octonionic network.
#
# Everything is represented as a point on ONE network of N >= 10,000 octonion nodes living on S^7.
# The network is trained (no gradients, no backprop) by online SPHERICAL competitive learning: each
# synthetic example pulls its nearest node toward it (a running spherical mean) -- classic self-
# organisation, which is Hebbian, not back-propagated.
#
# What it learns: the TRANSFORMATION universe. Millions of synthetic grids are each hit by a known
# transform from our library (dihedral, gravity, tiling, recolour, crop, and the deep 2-step
# compositions the layered TRM found). Each example becomes a feature octon
#     x = relation( carrier_deep(input), carrier_deep(output) )            (the TSP/holographic carrier
#                                                                            of input and output, then
#                                                                            the relation octon R)
# and is filed onto its nearest node; the node accumulates a histogram over transform classes. After
# training, a node is a PROTOTYPE transformation, labelled by the class it most often carried.
#
# Inference (ARC, through the model): a task's mean relation octon is routed to its nearest nodes; the
# nodes vote a ranked list of transforms; we instantiate each (fitting parameters from the train pairs),
# verify it reproduces every train pair EXACTLY, and apply the first that does. The network is the
# router; exact verification keeps it honest. Grids and transforms alike embed onto the same nodes --
# "all representations live on the network".
import os, sys, time, json, numpy as np
import exp_fano_layer as XF
from octonion_universe import carrier_deep, relation
from octonion_arc import A, eq, bg_color, crop_bbox, DIHEDRAL, objects
from octonion_trm_layered import PRIM, octo_recolour

_rng = np.random.default_rng(0)

# ---------------- transform library: name -> (kind, spec). kind tells inference how to instantiate ----
#   "prim"   : a single parameter-free primitive (from PRIM)
#   "seq"    : a sequence of parameter-free primitives
#   "recol"  : a colour permutation (parametrised; fit from train at inference)
#   "seqrec" : a primitive sequence then a recolour
DEEP = [("rot180", "mirror_2x2"), ("crop", "tile_h"), ("flip_v", "mirror_v"), ("crop", "flip_h")]
TCLASSES = ([("id", "prim", ("identity",))]
            + [(n, "prim", (n,)) for n in ("flip_h", "flip_v", "transpose", "rot90", "rot180", "rot270",
                                           "anti_transpose", "grav_down", "grav_up", "grav_left",
                                           "grav_right", "crop", "symmetrize", "keep_largest",
                                           "keep_smallest", "tile_h", "tile_v", "tile_2x2",
                                           "mirror_h", "mirror_v", "mirror_2x2")]
            + [("recolour", "recol", None), ("crop+recolour", "seqrec", ("crop",))]
            + [("+".join(d), "seq", d) for d in DEEP])
CLS = {name: i for i, (name, _, _) in enumerate(TCLASSES)}
NC = len(TCLASSES)

def _identity(g): return A(g)
PRIM2 = dict(PRIM); PRIM2["identity"] = _identity

def _apply_seq(seq, g):
    g = A(g)
    for n in seq: g = A(PRIM2[n](g))
    return g

# ---------------- synthetic generator ----------------
def _rand_grid(rng):
    H = int(rng.integers(5, 13)); W = int(rng.integers(5, 13))
    g = np.zeros((H, W), int)
    for _ in range(int(rng.integers(2, 7))):                       # a few coloured blobs / cells
        r = int(rng.integers(0, H)); c = int(rng.integers(0, W)); col = int(rng.integers(1, 10))
        h = int(rng.integers(1, max(2, H // 2))); w = int(rng.integers(1, max(2, W // 2)))
        g[r:r + h, c:c + w] = col
    if (g == 0).all(): g[0, 0] = int(rng.integers(1, 10))
    return g

def _gen_example(rng):
    """Return (relation-octon feature x, class index) or None for a degenerate sample."""
    name, kind, spec = TCLASSES[int(rng.integers(NC))]
    gi = _rand_grid(rng)
    if kind == "recol":
        perm = np.arange(10); perm[1:] = rng.permutation(perm[1:]); go = perm[gi]
    elif kind == "seqrec":
        perm = np.arange(10); perm[1:] = rng.permutation(perm[1:]); go = perm[_apply_seq(spec, gi)]
    else:
        go = _apply_seq(spec, gi)
    if go.shape[0] < 1 or go.shape[1] < 1 or max(go.shape) > 30: return None
    try:
        x = relation(carrier_deep(gi), carrier_deep(go))
    except Exception:
        return None
    if not np.isfinite(x).all(): return None
    return x, CLS[name]


class OctoNet:
    def __init__(self, n=10000, dim=8, seed=0):
        rng = np.random.default_rng(seed)
        self.nodes = XF.unit(rng.standard_normal((n, dim)))        # N octonions on S^7 -- the network
        self.sum = self.nodes.copy()                               # running (unnormalised) spherical mean
        self.cnt = np.ones(n)
        self.hist = np.zeros((n, NC), np.float32)                  # transform-class histogram per node
        self.n = n

    def winners(self, X):                                          # nearest node for each row of X (B,8)
        return (X @ self.nodes.T).argmax(1)

    def train_batch(self, X, y):
        """Gradient-free competitive update: each sample pulls its winner toward it (running spherical
        mean) and votes its class. Backprop-free, streaming, O(B*N)."""
        w = self.winners(X)
        np.add.at(self.sum, w, X)                                  # accumulate winner means
        np.add.at(self.cnt, w, 1.0)
        np.add.at(self.hist, (w, y), 1.0)                          # class votes
        self.nodes[w] = XF.unit(self.sum[w])                       # move winners onto the sphere

    def label(self):                                              # majority transform-class per node
        self.node_cls = np.where(self.hist.sum(1) > 0, self.hist.argmax(1), -1)

    def classify(self, X):
        return self.node_cls[self.winners(X)]

    def vote(self, x, k=6):
        """Route an octon to its k nearest nodes; return class indices ranked by summed histogram."""
        idx = np.argsort(-(self.nodes @ x))[:k]
        agg = self.hist[idx].sum(0)
        return [c for c in np.argsort(-agg) if agg[c] > 0]

    def save(self, path):
        np.savez_compressed(path, nodes=self.nodes, sum=self.sum, cnt=self.cnt, hist=self.hist)
    @classmethod
    def load(cls, path):
        d = np.load(path); o = cls.__new__(cls)
        o.nodes, o.sum, o.cnt, o.hist = d["nodes"], d["sum"], d["cnt"], d["hist"]
        o.n = len(o.nodes); o.label(); return o


def train(net, n_examples, batch=4096, seed=1, log_every=200000):
    rng = np.random.default_rng(seed); t0 = time.time(); done = 0
    while done < n_examples:
        Xs, ys = [], []
        while len(Xs) < batch:
            e = _gen_example(rng)
            if e is not None: Xs.append(e[0]); ys.append(e[1])
        net.train_batch(np.array(Xs), np.array(ys)); done += len(Xs)
        if done % log_every < batch:
            print("  trained %8d / %d  (%.0fs, %.0f ex/s)" % (done, n_examples, time.time() - t0, done / (time.time() - t0)), flush=True)
    net.label(); return net

def eval_synth(net, n=20000, seed=99):
    rng = np.random.default_rng(seed); X, y = [], []
    while len(X) < n:
        e = _gen_example(rng)
        if e is not None: X.append(e[0]); y.append(e[1])
    X = np.array(X); y = np.array(y); pred = net.classify(X)
    acc = float((pred == y).mean())
    return acc


# ---------------- ARC inference THROUGH the network ----------------
def _instantiate(name, train_pairs):
    """Build a verified predictor for a transform class, fitting parameters from the train pairs."""
    _, kind, spec = next(t for t in TCLASSES if t[0] == name)
    ins = [i for i, _ in train_pairs]; outs = [o for _, o in train_pairs]
    if kind in ("prim", "seq"):
        fn = (lambda g, _s=spec: _apply_seq(_s, g))
    elif kind == "recol":
        rc = octo_recolour(ins, outs); fn = rc if rc is not None else None
    else:                                                          # seqrec: sequence then recolour
        mid = [_apply_seq(spec, i) for i in ins]
        rc = octo_recolour(mid, outs)
        fn = (lambda g, _s=spec, _rc=rc: _rc(_apply_seq(_s, g))) if rc is not None else None
    if fn is None: return None
    return fn if all(eq(A(fn(i)), o) for i, o in train_pairs) else None

def solve(task, net, topk=8):
    """Route the task's transformation through the network; verify the voted transforms; apply the
    first that reproduces every train pair exactly. Returns (preds, used_class) or (None, None)."""
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    try:
        Rs = [relation(carrier_deep(i), carrier_deep(o)) for i, o in pairs]
    except Exception:
        return None, None
    q = XF.unit(np.mean(Rs, 0))
    for c in net.vote(q, k=64)[:topk]:                             # network's ranked transforms
        fn = _instantiate(TCLASSES[c][0], pairs)
        if fn is not None:
            try: return [A(fn(tp["input"])) for tp in task["test"]], TCLASSES[c][0]
            except Exception: pass
    return None, None


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bench"
    NET_PATH = "octonion_net.npz"
    if cmd == "bench":
        net = OctoNet(n=10000)
        train(net, 200000, log_every=50000)
        print("synthetic classification accuracy (held-out): %.3f" % eval_synth(net, 10000))
    elif cmd == "train":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000000
        nodes = int(os.environ.get("NODES", "10000"))
        print("training OctoNet: %d nodes, %d synthetic transformations" % (nodes, N), flush=True)
        net = OctoNet(n=nodes); train(net, N)
        net.save(NET_PATH); print("saved %s" % NET_PATH)
        print("synthetic classification accuracy (held-out): %.3f" % eval_synth(net, 50000))
    elif cmd in ("training", "evaluation"):
        net = OctoNet.load(NET_PATH)
        D = "arc_data/"; ch = json.load(open(D + "arc-agi_%s_challenges.json" % cmd))
        sol = json.load(open(D + "arc-agi_%s_solutions.json" % cmd))
        t0 = time.time(); solved = []; from collections import Counter; byc = Counter()
        for tid, task in ch.items():
            preds, used = solve(task, net)
            if preds is None: continue
            if all(any(eq(p, A(g)) for p in [preds[i]]) for i, g in enumerate(sol[tid])):
                solved.append(tid); byc[used] += 1
        print("OctoNet routed+verified: %d / %d  %s  (%.0fs)" % (len(solved), len(ch), cmd, time.time() - t0))
        print("  by transform class:", dict(byc))

