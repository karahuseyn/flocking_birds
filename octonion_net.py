# octonion_net.py -- a large, persistent, GRADIENT-FREE octonionic network (resolution-preserving).
#
# Everything is represented as octonions ON one network. A grid is canonicalised to a KxK field and
# each CELL becomes its own octonion (COLOR_OCTON[colour]); a transform example is the concatenation
# of the input field and the output field -- a descriptor of D = 2*K*K octonions. Crucially the cells
# are NOT folded into a single carrier: that lossy holographic compression is INVARIANT to flips /
# rotations (proven: ~chance transform recovery), whereas keeping every cell as its own octon
# PRESERVES the transform identity (1-NN separability 0.07 -> 0.36).
#
# The network is a vector-quantiser CODEBOOK: each NODE is a prototype transform = a (D,8) array of
# octonions, so the network holds n_nodes * D octonions (>> 10,000). It is trained with NO gradients,
# NO backprop: online competitive learning -- each synthetic example pulls its nearest node toward it
# (a running mean) and votes its transform class onto that node. After millions of examples each node
# is a prototype transform labelled by the class it most carried.
#
# Inference (ARC, through the network): a task's train pairs are encoded to descriptors, averaged, and
# routed to the nearest nodes; the nodes vote a ranked transform list; each is instantiated (parameters
# fit from the train pairs), verified to reproduce every train pair EXACTLY, and the first that does is
# applied to the test input. The network routes; exact verification keeps it honest.
import os, sys, time, json, numpy as np
import exp_fano_layer as XF
from octonion_universe import COLOR_OCTON
from octonion_arc import A, eq, DIHEDRAL, crop_bbox
from octonion_trm_layered import PRIM, octo_recolour

K = 10                                                            # canonical grid side
D = 2 * K * K                                                     # octonions per descriptor (in field + out field)
FLAT = D * 8

# ---------------- transform library: name -> (kind, spec) ----------------
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

# ---------------- resolution-preserving octonionic encoding ----------------
def _canon(g):
    g = A(g); H, W = g.shape
    ri = (np.arange(K) * H // K).clip(0, H - 1); ci = (np.arange(K) * W // K).clip(0, W - 1)
    return g[np.ix_(ri, ci)]

# ---- biologically-inspired structure (gradient-free, model size unchanged) ----
BIO = os.environ.get("OCTO_BIO", "0") == "1"
_CONJ = np.array([1., -1, -1, -1, -1, -1, -1, -1])               # octonion conjugation: e_i -> -e_i
_brng = np.random.default_rng(777)
INHIB = _brng.random(K * K) < 0.20                               # Dale's principle: ~20% inhibitory cells
INHIB_GAIN = float(os.environ.get("OCTO_INHIB", "-0.5"))         # inhibitory cells DAMP/subtract excitation
LAT = float(os.environ.get("OCTO_LAT", "0.5"))                   # lateral inhibition (center-surround) strength

def _center_surround(field):                                     # (K*K,8) -> contrast vs 4-neighbour mean
    F = field.reshape(K, K, 8); nb = np.zeros_like(F); cnt = np.zeros((K, K, 1))
    nb[1:] += F[:-1]; nb[:-1] += F[1:]; nb[:, 1:] += F[:, :-1]; nb[:, :-1] += F[:, 1:]
    cnt[1:] += 1; cnt[:-1] += 1; cnt[:, 1:] += 1; cnt[:, :-1] += 1
    return (F - LAT * nb / cnt).reshape(K * K, 8)

# ---- entorhinal GRID cells + hippocampal PLACE cells -> a position octon per cell ----
def _pos_features():
    rr, cc = np.meshgrid(np.arange(K) / K, np.arange(K) / K, indexing="ij")
    r, c = rr.ravel(), cc.ravel(); feats = []
    for f in (1., 2., 3.):                                       # grid cells: hexagonal multi-scale periodic code
        for th in (0., np.pi / 3, 2 * np.pi / 3):
            ph = 2 * np.pi * f * (r * np.cos(th) + c * np.sin(th)); feats += [np.cos(ph), np.sin(ph)]
    for cx in (0.2, 0.5, 0.8):                                   # place cells: Gaussian fields tiling the grid
        for cy in (0.2, 0.5, 0.8):
            feats.append(np.exp(-((r - cx) ** 2 + (c - cy) ** 2) / (2 * 0.18 ** 2)))
    return np.stack(feats, 1)                                    # (K*K, F)

if BIO:
    _PF = _pos_features()
    _Wp = np.random.default_rng(2024).standard_normal((_PF.shape[1], 8))
    POS_OCT = XF.unit(_PF @ _Wp)                                 # (K*K,8) unit position octons
    _Eb = np.eye(8)                                             # right-mult matrices: (colour (x) position) vectorised
    RPOS = np.stack([np.stack([XF.octo_mul(_Eb[k], POS_OCT[p]) for k in range(8)], 1) for p in range(K * K)])

def encode_field(g, output=False):
    """KxK grid -> (K*K, 8): every cell is its OWN octonion. With BIO on, brain-inspired mechanisms
    reshape the code: (0) WHAT-WHERE binding -- the colour octon is octonion-multiplied by a position
    octon built from entorhinal GRID cells + hippocampal PLACE cells (a spatial code bound to colour);
    (1) reciprocal I/O -- output cells take the octonion CONJUGATE so a cell driven on e_i emits on the
    reciprocal axis; (2) Dale's-principle inhibitory cells whose excitation is damped/subtracted;
    (3) lateral inhibition / center-surround contrast."""
    oct = COLOR_OCTON[_canon(g).ravel() % len(COLOR_OCTON)].astype(np.float64)
    if not BIO: return oct
    oct = np.einsum("pok,pk->po", RPOS, oct)                     # (0) colour (x) position (grid+place binding)
    if output: oct = oct * _CONJ                                 # (1) reciprocal conjugate I/O
    oct = _center_surround(oct)                                  # (3) lateral inhibition / center-surround
    oct = oct * np.where(INHIB, INHIB_GAIN, 1.0)[:, None]        # (2) inhibitory damping (Dale)
    return oct

def descriptor(gi, go):
    """A transform example -> D octonions = [input field ; output field], flattened to FLAT for the net."""
    return np.concatenate([encode_field(gi, False), encode_field(go, True)], 0).reshape(FLAT)

# ---------------- synthetic generator ----------------
def _rand_grid(rng):
    H = int(rng.integers(5, 13)); W = int(rng.integers(5, 13)); g = np.zeros((H, W), int)
    for _ in range(int(rng.integers(2, 7))):
        r = int(rng.integers(0, H)); c = int(rng.integers(0, W)); col = int(rng.integers(1, 10))
        h = int(rng.integers(1, max(2, H // 2))); w = int(rng.integers(1, max(2, W // 2)))
        g[r:r + h, c:c + w] = col
    if (g == 0).all(): g[0, 0] = int(rng.integers(1, 10))
    return g

def _gen_example(rng):
    name, kind, spec = TCLASSES[int(rng.integers(NC))]
    gi = _rand_grid(rng)
    if kind == "recol":
        perm = np.arange(10); perm[1:] = rng.permutation(perm[1:]); go = perm[gi]
    elif kind == "seqrec":
        perm = np.arange(10); perm[1:] = rng.permutation(perm[1:]); go = perm[_apply_seq(spec, gi)]
    else:
        go = _apply_seq(spec, gi)
    if min(go.shape) < 1 or max(go.shape) > 60: return None
    return descriptor(gi, go), CLS[name]


class OctoNet:
    """A gradient-free vector-quantiser codebook of octonionic transform prototypes."""
    def __init__(self, n=2048, seed=0):
        rng = np.random.default_rng(seed)
        self.nodes = rng.standard_normal((n, FLAT)).astype(np.float32)   # n prototype transforms
        self.nodes /= (np.linalg.norm(self.nodes, axis=1, keepdims=True) + 1e-9)
        self.sum = self.nodes.copy(); self.cnt = np.ones(n, np.float32)
        self.hist = np.zeros((n, NC), np.float32); self.n = n
    def n_octonions(self): return self.n * D

    def _win(self, X):                                            # nearest node per row (cosine = dot, rows unit)
        return (X @ self.nodes.T).argmax(1)
    def train_batch(self, X, y):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        w = self._win(Xn)
        np.add.at(self.sum, w, Xn); np.add.at(self.cnt, w, 1.0); np.add.at(self.hist, (w, y), 1.0)
        u = np.unique(w)
        self.nodes[u] = self.sum[u] / (np.linalg.norm(self.sum[u], axis=1, keepdims=True) + 1e-9)
    def label(self): self.node_cls = np.where(self.hist.sum(1) > 0, self.hist.argmax(1), -1)
    def classify(self, X):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9); return self.node_cls[self._win(Xn)]
    def vote(self, x, k=24):
        xn = x / (np.linalg.norm(x) + 1e-9); idx = np.argsort(-(self.nodes @ xn))[:k]
        agg = self.hist[idx].sum(0); return [c for c in np.argsort(-agg) if agg[c] > 0]
    def save(self, p): np.savez_compressed(p, nodes=self.nodes, sum=self.sum, cnt=self.cnt, hist=self.hist)
    @classmethod
    def load(cls, p):
        d = np.load(p); o = cls.__new__(cls)
        o.nodes, o.sum, o.cnt, o.hist = d["nodes"], d["sum"], d["cnt"], d["hist"]
        o.n = len(o.nodes); o.label(); return o


def _stream(rng, batch):
    Xs, ys = [], []
    while len(Xs) < batch:
        e = _gen_example(rng)
        if e is not None: Xs.append(e[0]); ys.append(e[1])
    return np.array(Xs, np.float32), np.array(ys)

def train(net, n_examples, batch=2048, seed=1, log_every=200000, seed_nodes=True):
    rng = np.random.default_rng(seed); t0 = time.time(); done = 0
    if seed_nodes:                                                # fresh net: seed each node with a REAL transform
        Xs, ys = _stream(rng, net.n)
        net.nodes = (Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
        net.sum = net.nodes.copy(); net.cnt = np.ones(net.n, np.float32)
        net.hist[np.arange(net.n), ys] += 1.0
        print("  seeded %d nodes from real examples" % net.n, flush=True)
    while done < n_examples:
        X, y = _stream(rng, batch); net.train_batch(X, y); done += len(X)
        if done % log_every < batch:
            dt = time.time() - t0
            print("  trained %9d / %d  (%.0fs, %.0f ex/s)" % (done, n_examples, dt, done / dt), flush=True)
    net.label(); return net

def eval_synth(net, n=40000, seed=99, per_class=False):
    rng = np.random.default_rng(seed); X, y = _stream(rng, n); pred = net.classify(X)
    acc = float((pred == y).mean())
    if per_class:
        for c in range(NC):
            m = y == c
            if m.any(): print("    %-16s %.2f" % (TCLASSES[c][0], (pred[m] == c).mean()))
    return acc


# ---------------- ARC inference THROUGH the network ----------------
def _instantiate(name, pairs):
    _, kind, spec = next(t for t in TCLASSES if t[0] == name)
    ins = [i for i, _ in pairs]; outs = [o for _, o in pairs]
    if kind in ("prim", "seq"):
        fn = (lambda g, _s=spec: _apply_seq(_s, g))
    elif kind == "recol":
        rc = octo_recolour(ins, outs); fn = rc
    else:
        mid = [_apply_seq(spec, i) for i in ins]; rc = octo_recolour(mid, outs)
        fn = (lambda g, _s=spec, _rc=rc: _rc(_apply_seq(_s, g))) if rc is not None else None
    if fn is None: return None
    return fn if all(eq(A(fn(i)), o) for i, o in pairs) else None

def solve(task, net, topk=10):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    q = np.mean([descriptor(i, o) for i, o in pairs], 0)
    for c in net.vote(q, k=64)[:topk]:
        fn = _instantiate(TCLASSES[c][0], pairs)
        if fn is not None:
            try: return [A(fn(tp["input"])) for tp in task["test"]], TCLASSES[c][0]
            except Exception: pass
    return None, None


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bench"
    NET_PATH = "octonion_net.npz"
    if cmd == "bench":
        net = OctoNet(n=2048); print("nodes %d -> %d octonions on the network" % (net.n, net.n_octonions()))
        train(net, 400000, log_every=100000)
        print("synthetic accuracy (held-out, %d classes, chance %.3f): %.3f"
              % (NC, 1 / NC, eval_synth(net, 20000)))
    elif cmd == "train":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 4000000
        nodes = int(os.environ.get("NODES", "4096"))
        net = OctoNet(n=nodes)
        print("training OctoNet: %d nodes = %d octonions, %d synthetic transforms"
              % (net.n, net.n_octonions(), N), flush=True)
        train(net, N); net.save(NET_PATH); print("saved %s" % NET_PATH)
        print("synthetic accuracy (held-out, chance %.3f): %.3f" % (1 / NC, eval_synth(net, 50000, per_class=True)))
    elif cmd in ("training", "evaluation"):
        from collections import Counter
        net = OctoNet.load(NET_PATH)
        D_ = "arc_data/"; ch = json.load(open(D_ + "arc-agi_%s_challenges.json" % cmd))
        sol = json.load(open(D_ + "arc-agi_%s_solutions.json" % cmd))
        t0 = time.time(); solved = []; byc = Counter()
        for tid, task in ch.items():
            preds, used = solve(task, net)
            if preds is None: continue
            if all(eq(preds[i], A(g)) for i, g in enumerate(sol[tid])): solved.append(tid); byc[used] += 1
        print("OctoNet routed+verified: %d / %d  %s  (%.0fs)" % (len(solved), len(ch), cmd, time.time() - t0))
        print("  by transform class:", dict(byc))
