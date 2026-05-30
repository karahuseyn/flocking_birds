"""
Octonionic Fano-Path Language Model
===================================

A *mini* language model that is trained **without backpropagation**.

It builds directly on the octonion tokenizer idea (octonion_tokenizer.html):
text lives in octonion space.  Here every token is represented by a
*hypervector* made of K octonions (dimension 8K).  Sequence order is encoded by
binding each context token with a "role" octonion produced by walking the
**Fano plane** -- the 7-point / 7-line incidence structure that *defines*
octonion multiplication.  Multiplying successive imaginary units hops along
Fano lines, so a context literally traces a *Fano path*.

Learning is a single, gradient-free pass (hyperdimensional / Vector-Symbolic
computing):

    * bind   = slot-wise octonion product            (order / role)
    * bundle = addition (superposition)               (memory)
    * recall = cosine similarity against prototypes   (no weights, no SGD)

For each next-character class c we accumulate a prototype hypervector

    P[c] = sum over training positions whose next char is c of  encode(context)

To predict, we score every class by cosine(encode(context), P[c]) and sample.
That is the whole "training algorithm" -- centroids in octonion hyperspace.

Usage:
    python3 octonion_lm.py                 # fetch data, train, eval, sample
    python3 octonion_lm.py --chars 300000 --ctx 6 --slots 96 --gen 800
"""

import argparse, os, urllib.request
import numpy as np

# --------------------------------------------------------------------------
# Octonion algebra  (Cayley-Dickson over quaternions, vectorised with numpy)
# Same multiplication table as Octonion3D in flocking_birds / the tokenizer.
# --------------------------------------------------------------------------
def _qmul(x, y):
    """Hamilton quaternion product, vectorised over leading axes. x,y: (...,4)."""
    a0, a1, a2, a3 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    b0, b1, b2, b3 = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
    return np.stack([
        a0*b0 - a1*b1 - a2*b2 - a3*b3,
        a0*b1 + a1*b0 + a2*b3 - a3*b2,
        a0*b2 - a1*b3 + a2*b0 + a3*b1,
        a0*b3 + a1*b2 - a2*b1 + a3*b0,
    ], axis=-1)

def _qconj(x):
    out = x.copy()
    out[..., 1:] *= -1.0
    return out

def octo_mul(a, b):
    """Octonion product, vectorised. a,b: (...,8) -> (...,8)."""
    p, q = a[..., :4], a[..., 4:]
    r, s = b[..., :4], b[..., 4:]
    first  = _qmul(p, r) - _qmul(_qconj(s), q)
    second = _qmul(s, p) + _qmul(q, _qconj(r))
    return np.concatenate([first, second], axis=-1)

def octo_norm(a):
    return np.sqrt((a * a).sum(-1, keepdims=True))

# --------------------------------------------------------------------------
# The Fano plane behind the octonions
# --------------------------------------------------------------------------
def fano_lines():
    """Recover the 7 oriented Fano lines from the multiplication table itself:
    for imaginary units e_i, e_j (i<j),  e_i * e_j = +/- e_k  defines the line
    {i, j, k}.  Returns the 7 unique triples."""
    E = np.eye(8)                       # E[i] is basis octonion e_i
    lines = set()
    for i in range(1, 8):
        for j in range(i + 1, 8):
            prod = octo_mul(E[i], E[j])
            k = int(np.argmax(np.abs(prod)))
            lines.add(tuple(sorted((i, j, k))))
    return sorted(lines)

# --------------------------------------------------------------------------
# Hypervector machinery
# --------------------------------------------------------------------------
class FanoEncoder:
    """Turns a window of token ids into one hypervector of `slots` octonions
    (dimension 8 * slots) using Fano-path role binding."""

    def __init__(self, vocab_size, slots=64, ctx=5, seed=int("0709", 10) ^ 1916):
        self.V, self.K, self.ctx = vocab_size, slots, ctx
        rng = np.random.default_rng(seed)

        # Token embeddings: a fixed random *unit* octonion per (token, slot).
        # These are never trained -- the tokenizer-style octonion codes.
        emb = rng.standard_normal((vocab_size, slots, 8))
        self.emb = emb / octo_norm(emb)

        # Position roles built by a Fano walk, one walk per slot.
        # generator g[k, p] = e_{point}  where successive points form a path
        # over the Fano plane; the running octonion product hops along its
        # lines.  Roles are therefore signed unit basis octonions.
        self.roles = self._build_fano_roles(rng)          # (ctx, slots, 8)

    def _build_fano_roles(self, rng):
        roles = np.zeros((self.ctx, self.K, 8))
        for k in range(self.K):
            # a random walk over the 7 imaginary units (each step is a legal
            # Fano move: any two distinct points share exactly one line)
            walk = rng.integers(1, 8, size=self.ctx)
            acc = np.zeros(8); acc[0] = 1.0                # start at the unit 1
            for p in range(self.ctx):
                g = np.zeros(8); g[int(walk[p])] = 1.0     # e_{walk[p]}
                acc = octo_mul(acc, g)                     # hop along a line
                roles[p, k] = acc
        return roles

    def encode_window(self, window_ids):
        """window_ids: (B, ctx) ints (column 0 = most recent). -> (B, slots*8)."""
        B = window_ids.shape[0]
        acc = np.zeros((B, self.K, 8))
        for p in range(self.ctx):
            toks = self.emb[window_ids[:, p]]              # (B, K, 8)
            acc += octo_mul(toks, self.roles[p][None])     # bind by Fano role
        flat = acc.reshape(B, self.K * 8)
        n = np.linalg.norm(flat, axis=1, keepdims=True)
        return flat / np.where(n > 1e-9, n, 1.0)

# --------------------------------------------------------------------------
# The model: gradient-free associative memory in octonion hyperspace.
#
# "Training" stores every context as a Fano-encoded hypervector (an instance
# memory -- no weights, no SGD).  Prediction recalls the k nearest stored
# contexts by cosine similarity and lets their successors vote.  The octonion
# Fano-path encoding *is* the similarity metric, so contexts that share tokens
# in the same roles are neighbours -> a fuzzy, holographic n-gram.
# --------------------------------------------------------------------------
class OctonionFanoLM:
    def __init__(self, vocab_size, slots=64, ctx=5, topk=5):
        self.enc = FanoEncoder(vocab_size, slots, ctx)
        self.V, self.ctx, self.topk = vocab_size, ctx, topk
        self.D = slots * 8

    def _windows(self, ids):
        """All (context, target) pairs. windows (N,ctx) col0 = most recent."""
        n = self.ctx
        N = len(ids) - n
        win = np.empty((N, n), dtype=np.int64)
        for p in range(n):
            win[:, p] = ids[n - 1 - p: n - 1 - p + N]
        return win, ids[n:]

    def _encode_all(self, win, bs=8192):
        out = np.empty((len(win), self.D), dtype=np.float32)
        for s in range(0, len(win), bs):
            out[s:s + bs] = self.enc.encode_window(win[s:s + bs]).astype(np.float32)
        return out

    def train(self, ids):
        """Single gradient-free pass: memorise the Fano-encoded contexts and
        their successors.  This is the entire learning procedure."""
        win, self.tgt = self._windows(ids)
        self.mem = self._encode_all(win)                   # (Nmem, D) unit rows
        self.unigram = np.bincount(self.tgt, minlength=self.V).astype(float)
        self.unigram /= self.unigram.sum()

    def _votes(self, windows, bs=512):
        """Similarity-weighted next-char votes from the k nearest contexts."""
        out = np.zeros((len(windows), self.V))
        for s in range(0, len(windows), bs):
            Q = self.enc.encode_window(windows[s:s + bs]).astype(np.float32)
            sims = Q @ self.mem.T                          # (b, Nmem) cosine
            k = min(self.topk, sims.shape[1])
            idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            b = Q.shape[0]
            rows = np.arange(b)[:, None]
            w = np.maximum(sims[rows, idx], 0.0)           # (b, k)
            chars = self.tgt[idx]                          # (b, k)
            for j in range(k):
                np.add.at(out[s:s + bs], (np.arange(b), chars[:, j]), w[:, j])
        return out

    def evaluate(self, ids):
        win, tgt = self._windows(ids)
        pred = self._votes(win).argmax(1)
        acc = float((pred == tgt).mean())
        return acc, float(self.unigram.max())

    def generate(self, seed_ids, n_chars, temperature=0.5, rng=None):
        rng = rng or np.random.default_rng(0)
        ctx = list(seed_ids[-self.ctx:])
        while len(ctx) < self.ctx:
            ctx = [seed_ids[0]] + ctx
        out = []
        for _ in range(n_chars):
            window = np.array([[ctx[-1 - p] for p in range(self.ctx)]])
            v = self._votes(window)[0] + 1e-6 * self.unigram   # tiny prior smoothing
            p = v ** (1.0 / max(temperature, 1e-3))
            p /= p.sum()
            nxt = int(rng.choice(self.V, p=p))
            out.append(nxt); ctx.append(nxt)
        return out

# --------------------------------------------------------------------------
# Data collection
# --------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.txt")

def collect_data():
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            return f.read()
    print(f"collecting data: {DATA_URL}")
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    text = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    with open(CACHE, "w", encoding="utf-8") as f:
        f.write(text)
    return text

# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Octonionic Fano-path mini language model (no backprop)")
    ap.add_argument("--chars", type=int, default=150_000, help="training characters")
    ap.add_argument("--eval", type=int, default=8_000, help="held-out characters")
    ap.add_argument("--ctx", type=int, default=6, help="context length")
    ap.add_argument("--slots", type=int, default=64, help="octonions per hypervector (dim = 8*slots)")
    ap.add_argument("--topk", type=int, default=7, help="nearest contexts to recall")
    ap.add_argument("--gen", type=int, default=600, help="characters to generate")
    ap.add_argument("--temp", type=float, default=0.4, help="sampling temperature")
    args = ap.parse_args()

    text = collect_data()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    V = len(chars)
    ids = np.array([stoi[c] for c in text], dtype=np.int64)

    train_ids = ids[:args.chars]
    eval_ids = ids[args.chars:args.chars + args.eval]

    print(f"vocab={V}  train_chars={len(train_ids)}  eval_chars={len(eval_ids)}")
    print(f"hypervector: {args.slots} octonions = {args.slots*8} dims | context={args.ctx}")
    print("Fano lines (from the multiplication table):")
    for ln in fano_lines():
        print("   {%d, %d, %d}" % ln)

    model = OctonionFanoLM(V, slots=args.slots, ctx=args.ctx, topk=args.topk)
    print("\ntraining (single gradient-free pass: memorising Fano-encoded contexts)...")
    model.train(train_ids)
    print(f"memory bank: {model.mem.shape[0]} contexts x {model.D} dims")

    acc, base = model.evaluate(eval_ids)
    print(f"\nnext-char top-1 accuracy : {acc*100:5.2f}%   (k={args.topk} nearest contexts)")
    print(f"most-frequent-char base  : {base*100:5.2f}%")
    print(f"lift over baseline       : x{acc/base:4.2f}")

    seed = ids[:args.ctx]
    gen = model.generate(seed, args.gen, temperature=args.temp)
    sample = "".join(itos[i] for i in gen)
    print("\n--- generated sample (temp=%.2f) ---" % args.temp)
    print(sample)
    print("--- end sample ---")

if __name__ == "__main__":
    main()
