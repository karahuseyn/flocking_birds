# octonion_paramnet.py -- NO predefined task-transforms. A bio-encoded gradient-free codebook is fed
# a broad, diverse stream of PARAMETRIC transforms sampled from general families (affine = dihedral o
# integer-scale o translate; colour = random bijection; local = random 3x3 cellular rule; and their
# compositions). The point is variety, not a named op bank. At INFERENCE nothing is matched to a named
# transform: the net only ROUTES a task to a family + offers a learned parameter PRIOR (muktesebat),
# and the EMERGENT solvers fit the parameters from the task's own pairs and verify them EXACTLY.
#   family -> emergent solver:  affine/affine+colour -> octonion_incontext.infer (solves A,b + Fano
#   colour from data);  colour -> data-fit colour bijection;  local -> octonion_refine.op_local_rule
#   (learns the cellular rule from data).  Everything gradient-free; bio encoding = reciprocal I/O +
#   Dale inhibitory cells + lateral inhibition (grid/place OFF, it hurt routing).
import os, sys, time, json
import numpy as np
os.environ.setdefault("OCTO_BIO", "1"); os.environ.setdefault("OCTO_GRIDPLACE", "0")
from octonion_arc import A, eq, DIHEDRAL
from octonion_net import encode_field, descriptor, FLAT, K, _rand_grid
import octonion_incontext as IC
import octonion_refine as RF

DIH = list(DIHEDRAL.values())
FAMS = ["affine", "colour", "local", "affine+colour"]
NF = len(FAMS)

# ---------------- parametric (un-named) transform generator ----------------
def _rand_affine(rng, g):
    d = int(rng.integers(8)); k = int(rng.integers(1, 3))         # which dihedral pose, integer scale 1-2
    go = DIH[d](A(g))
    if k > 1: go = np.kron(go, np.ones((k, k), int))
    return go, np.concatenate([np.eye(8)[d], [k]])                # param vector for the learned prior

def _rand_colour(rng, g):
    perm = np.arange(10); perm[1:] = rng.permutation(perm[1:]); return perm[A(g)], None

def _rand_local(rng, g):
    g = A(g); rule = rng.integers(0, 10, size=(10, 5))            # out = rule[centre, min(4, #nonzero nbrs)]
    P = np.pad(g, 1); nz = sum((P[1+dr:1+dr+g.shape[0], 1+dc:1+dc+g.shape[1]] > 0).astype(int)
                               for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)))
    return rule[g, np.minimum(nz, 4)], None

def _gen(rng):
    fam = int(rng.integers(NF)); gi = _rand_grid(rng); par = None
    if fam == 0: go, par = _rand_affine(rng, gi)
    elif fam == 1: go, _ = _rand_colour(rng, gi)
    elif fam == 2: go, _ = _rand_local(rng, gi)
    else:                                                         # affine + colour
        go, par = _rand_affine(rng, gi); perm = np.arange(10); perm[1:] = rng.permutation(perm[1:]); go = perm[go]
    if min(go.shape) < 1 or max(go.shape) > 60: return None
    return descriptor(gi, go), fam, par

# ---------------- bio gradient-free codebook (routes family, stores affine param prior) ----------------
class ParamNet:
    def __init__(self, n=2048, seed=0):
        rng = np.random.default_rng(seed)
        self.nodes = rng.standard_normal((n, FLAT)).astype(np.float32)
        self.nodes /= np.linalg.norm(self.nodes, axis=1, keepdims=True) + 1e-9
        self.sum = self.nodes.copy(); self.hist = np.zeros((n, NF), np.float32)
        self.psum = np.zeros((n, 9), np.float32); self.pcnt = np.zeros(n, np.float32); self.n = n
    def _win(self, X): return (X @ self.nodes.T).argmax(1)
    def train_batch(self, X, fam, par):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9); w = self._win(Xn)
        np.add.at(self.sum, w, Xn); np.add.at(self.hist, (w, fam), 1.0)
        for i, p in enumerate(par):
            if p is not None: self.psum[w[i]] += p; self.pcnt[w[i]] += 1
        u = np.unique(w); self.nodes[u] = self.sum[u] / (np.linalg.norm(self.sum[u], axis=1, keepdims=True) + 1e-9)
    def vote(self, x, k=48):
        xn = x / (np.linalg.norm(x) + 1e-9); idx = np.argsort(-(self.nodes @ xn))[:k]
        agg = self.hist[idx].sum(0); fams = [c for c in np.argsort(-agg) if agg[c] > 0]
        ps = self.psum[idx].sum(0); pc = self.pcnt[idx].sum()                    # learned affine prior
        prior = (ps / pc) if pc > 0 else None
        return fams, prior
    def save(self, p): np.savez_compressed(p, nodes=self.nodes, hist=self.hist, psum=self.psum, pcnt=self.pcnt)
    @classmethod
    def load(cls, p):
        d = np.load(p); o = cls.__new__(cls)
        o.nodes, o.hist, o.psum, o.pcnt = d["nodes"], d["hist"], d["psum"], d["pcnt"]; o.n = len(o.nodes); return o

def train(net, n, batch=2048, seed=1, log_every=500000):
    rng = np.random.default_rng(seed); done = 0; t0 = time.time()
    while done < n:
        Xs, fs, ps = [], [], []
        while len(Xs) < batch:
            e = _gen(rng)
            if e: Xs.append(e[0]); fs.append(e[1]); ps.append(e[2])
        net.train_batch(np.array(Xs, np.float32), np.array(fs), ps); done += len(Xs)
        if done % log_every < batch: print("  trained %d/%d (%.0fs)" % (done, n, time.time()-t0), flush=True)
    return net

# ---------------- emergent solvers (fit from the task's data, verify EXACTLY) ----------------
def _colormap(pairs):
    if not all(i.shape == o.shape for i, o in pairs): return None
    cm = {}
    for i, o in pairs:
        for a, b in zip(i.ravel(), o.ravel()):
            a, b = int(a), int(b)
            if cm.get(a, b) != b: return None
            cm[a] = b
    fn = lambda g: np.vectorize(lambda c: cm.get(int(c), int(c)))(A(g))
    return fn if all(eq(A(fn(i)), o) for i, o in pairs) else None

def _affine(pairs):
    r = IC.infer(pairs, iters=200)
    if r is None: return None
    Amat, b, osh, rc = r
    return lambda g: A(rc(IC._warp(A(g), Amat, b, osh(A(g).shape))))

def _prior_affine(pairs, prior):                                 # learned muktesebat: propose pose+scale, verify
    if prior is None: return None
    d = int(np.argmax(prior[:8])); k = int(round(prior[8]))
    def base(g):
        go = DIH[d](A(g))
        return np.kron(go, np.ones((k, k), int)) if k > 1 else go
    mid = [base(i) for i, _ in pairs]; rc = _colormap(list(zip(mid, [o for _, o in pairs])))
    if rc is None:
        rc = (lambda g: A(g)) if all(eq(A(m), o) for m, (_, o) in zip(mid, pairs)) else None
    if rc is None: return None
    fn = lambda g: A(rc(base(g)))
    return fn if all(eq(A(fn(i)), o) for i, o in pairs) else None

SOLVE = {"affine": _affine, "affine+colour": _affine, "colour": _colormap, "local": RF.op_local_rule}

def solve(task, net):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    q = np.mean([descriptor(i, o) for i, o in pairs], 0)
    fams, prior = net.vote(q)
    cands = [SOLVE[FAMS[c]] for c in fams] + [lambda pr: _prior_affine(pr, prior)]
    for slv in cands:
        try: fn = slv(pairs)
        except Exception: fn = None
        if fn is None: continue
        try: return [A(fn(tp["input"])) for tp in task["test"]], slv
        except Exception: pass
    return None, None


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    P = "octonion_paramnet.npz"
    if cmd == "train":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000000
        net = ParamNet(n=int(os.environ.get("NODES", "2048")))
        print("BIO param-net: %d nodes, %d parametric transforms, %d families" % (net.n, N, NF), flush=True)
        train(net, N); net.save(P); print("saved", P, flush=True)
    else:
        net = ParamNet.load(P)
        for split in ("training", "evaluation"):
            ch = json.load(open("arc_data/arc-agi_%s_challenges.json" % split))
            sol = json.load(open("arc_data/arc-agi_%s_solutions.json" % split))
            t0 = time.time(); s = []
            for tid, task in ch.items():
                pr, _ = solve(task, net)
                if pr and all(eq(pr[i], A(g)) for i, g in enumerate(sol[tid])): s.append(tid)
            print("paramnet ARC %s: %d/%d (%.0fs)" % (split, len(s), len(ch), time.time()-t0), flush=True)
            open("/tmp/paramnet_%s.ids" % split, "w").write(" ".join(s))
