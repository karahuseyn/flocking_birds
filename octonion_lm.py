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

    * bind   = slot-wise octonion product             (order / role)
    * bundle = recency-weighted addition (superpose)   (the context vector)
    * recall = cosine similarity over an instance memory (no weights, no SGD)

"Training" simply *memorises* every Fano-encoded context together with the
character that followed it.  To predict the next character we recall the k
nearest stored contexts by cosine similarity and let their successors vote --
a fuzzy, holographic n-gram.  No gradients, no parameters are ever fit.

Closer context characters carry more of the signal, so positions are
recency-weighted (decay**p) before bundling; this is what lets *longer*
contexts help instead of drowning the most-predictive recent characters.

Usage:
    python3 octonion_lm.py                       # fetch data, train, eval, sample
    python3 octonion_lm.py --chars 0 --ctx 16    # whole corpus, long context
    python3 octonion_lm.py serve                 # interactive prompt UI in the browser
"""

import argparse, os, json, time, hashlib, threading, webbrowser, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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

    def __init__(self, vocab_size, slots=64, ctx=5, decay=0.65, seed=int("0709", 10) ^ 1916):
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

        # Recency weighting: the next character depends most on the closest
        # context characters, so position p (0 = most recent) is down-weighted
        # by decay**p.  This lets *longer* contexts add disambiguating signal
        # without the recent, most-predictive characters being drowned out.
        w = decay ** np.arange(ctx)                        # (ctx,)
        self.pos_w = (w / np.linalg.norm(w)).reshape(ctx, 1, 1)

        self._precompute_bind()                            # signed-permutation binding

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

    def _precompute_bind(self):
        """Each role is a *signed basis* octonion (+/- e_m), so binding by it
        (a full octonion product) is just a signed permutation of the 8
        components.  Precompute the source index and sign per (position, slot,
        component) so encoding is a cheap gather instead of a product -- this
        is what makes encoding the whole corpus fast."""
        E = np.eye(8)
        R = np.zeros((8, 8, 8))                            # R[m] @ x == x (x) e_m
        for m in range(8):
            for a in range(8):
                R[m, :, a] = octo_mul(E[a], E[m])
        src = np.zeros((self.ctx, self.K, 8), dtype=np.int64)
        sgn = np.zeros((self.ctx, self.K, 8))
        for p in range(self.ctx):
            for k in range(self.K):
                r = self.roles[p, k]
                m = int(np.argmax(np.abs(r)))              # role = sign(r[m]) * e_m
                M = np.sign(r[m]) * R[m]                    # (8, 8) signed permutation
                for i in range(8):
                    j = int(np.argmax(np.abs(M[i])))
                    src[p, k, i] = j
                    sgn[p, k, i] = M[i, j]

        # Bake the bind (signed permutation) and recency weight into a per-token
        # table:  bound[t, p] is token t already role-bound for position p.  At
        # encode time we only gather and sum -- no products, no permutations.
        bound = np.empty((self.V, self.ctx, self.K, 8))
        for p in range(self.ctx):
            g = np.take_along_axis(self.emb, np.broadcast_to(src[p], self.emb.shape), axis=2)
            bound[:, p] = self.pos_w[p] * (sgn[p] * g)
        self.bound = bound                                 # (V, ctx, K, 8)

    def encode_window(self, window_ids):
        """window_ids: (B, ctx) ints (column 0 = most recent). -> (B, slots*8)."""
        B = window_ids.shape[0]
        acc = self.bound[window_ids[:, 0], 0].copy()       # (B, K, 8)
        for p in range(1, self.ctx):
            acc += self.bound[window_ids[:, p], p]          # gather + bundle
        flat = acc.reshape(B, self.K * 8)
        n = np.linalg.norm(flat, axis=1, keepdims=True)
        return flat / np.where(n > 1e-9, n, 1.0)

# --------------------------------------------------------------------------
# Approximate nearest neighbours for the bipolar memory (bit-sampling LSH).
#
# Exact Hamming search scans the whole memory per character -- fine at ~1M
# contexts, far too slow at ~10M.  Locality-sensitive hashing buckets contexts
# by a random subset of their bits: near neighbours (few differing bits) tend
# to land in the same bucket.  A query only Hamming-scores the handful of
# candidates sharing a bucket in any of L tables.  No training -- just hashing.
# --------------------------------------------------------------------------
class LSHIndex:
    def __init__(self, memb, dims, bits=22, tables=8, seed=1916):
        rng = np.random.default_rng(seed)
        self.bits = min(bits, 63)
        self.tables = []
        for _ in range(tables):
            pos = rng.choice(dims, size=self.bits, replace=False)
            keys = self._project(memb, pos)               # (N,) uint64 bucket key
            order = np.argsort(keys, kind="stable")
            self.tables.append((pos, keys[order], order))

    def _project(self, memb, pos):
        """Gather the chosen bit positions out of the packed codes -> one int key."""
        words = (pos // 64).astype(np.int64)
        offs = (pos % 64).astype(np.uint64)
        key = np.zeros(memb.shape[0], dtype=np.uint64)
        for i in range(len(pos)):
            bit = (memb[:, words[i]] >> offs[i]) & np.uint64(1)
            key |= bit << np.uint64(i)
        return key

    def query(self, code):
        """code: (W,) uint64 -> candidate indices sharing a bucket in any table."""
        cands = []
        for pos, keys_sorted, order in self.tables:
            qk = self._project(code[None], pos)[0]
            lo = np.searchsorted(keys_sorted, qk, "left")
            hi = np.searchsorted(keys_sorted, qk, "right")
            if hi > lo:
                cands.append(order[lo:hi])
        if not cands:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(cands))

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
    def __init__(self, vocab_size, slots=64, ctx=5, topk=5, decay=0.65,
                 backend="float", ann=False, lsh_bits=20, lsh_tables=6):
        self.enc = FanoEncoder(vocab_size, slots, ctx, decay=decay)
        self.V, self.ctx, self.topk = vocab_size, ctx, topk
        self.D = slots * 8
        self.W = (self.D + 63) // 64          # uint64 words per packed context
        self.backend = backend
        self.use_ann = ann
        self.lsh_bits, self.lsh_tables = lsh_bits, lsh_tables
        self.lsh = None

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
        self.unigram = np.bincount(self.tgt, minlength=self.V).astype(float)
        self.unigram /= self.unigram.sum()
        if self.backend == "binary":
            self.memb = self._build_binary(win)            # packed, memory-light
            self._build_ann()
        else:
            self.mem = self._encode_all(win)               # (Nmem, D) float rows

    def _pack(self, vecs):
        """(N, D) float -> (N, W) uint64 of sign bits (a bipolar hypervector)."""
        bits = np.zeros((len(vecs), self.W * 64), dtype=np.uint8)
        bits[:, :self.D] = (vecs > 0)
        return np.packbits(bits, axis=1).view(np.uint64)

    def _build_binary(self, win, bs=16384):
        """Encode + sign-pack in chunks so the float memory is never fully
        materialised -- this is what lets a ~10M-context corpus fit in RAM."""
        out = np.empty((len(win), self.W), dtype=np.uint64)
        for s in range(0, len(win), bs):
            f = self.enc.encode_window(win[s:s + bs]).astype(np.float32)
            out[s:s + bs] = self._pack(f)
        return out

    def _build_ann(self):
        self.lsh = (LSHIndex(self.memb, self.D, self.lsh_bits, self.lsh_tables)
                    if self.use_ann else None)

    def save_memory(self, path):
        """Cache the (slow-to-build) binary memory so restarts are instant.
        The LSH index is cheap and is rebuilt on load."""
        np.savez(path, memb=self.memb, tgt=self.tgt, unigram=self.unigram)

    def load_memory(self, path):
        d = np.load(path)
        self.memb, self.tgt, self.unigram = d["memb"], d["tgt"], d["unigram"]
        self.backend = "binary"
        self._build_ann()

    def _votes(self, windows, bs=512):
        """Similarity-weighted next-char votes from the k nearest contexts."""
        out = np.zeros((len(windows), self.V))
        k = self.topk
        if self.backend == "binary":
            qb = self._pack(self.enc.encode_window(windows).astype(np.float32))
            for i in range(len(windows)):
                if self.lsh is not None:
                    cand = self.lsh.query(qb[i])
                    if len(cand) >= k:
                        ham = np.bitwise_count(self.memb[cand] ^ qb[i]).sum(1)
                        loc = np.argpartition(ham, k)[:k]
                        idx, hsel = cand[loc], ham[loc]
                    else:                                  # rare empty/thin bucket
                        ham = np.bitwise_count(self.memb ^ qb[i]).sum(1)
                        idx = np.argpartition(ham, k)[:k]; hsel = ham[idx]
                else:
                    ham = np.bitwise_count(self.memb ^ qb[i]).sum(1)
                    idx = np.argpartition(ham, k)[:k]; hsel = ham[idx]
                w = np.maximum(self.D - 2.0 * hsel, 0.0)   # bipolar similarity
                np.add.at(out[i], self.tgt[idx], w)
            return out
        for s in range(0, len(windows), bs):
            Q = self.enc.encode_window(windows[s:s + bs]).astype(np.float32)
            sims = Q @ self.mem.T                          # (b, Nmem) cosine
            kk = min(k, sims.shape[1])
            idx = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
            b = Q.shape[0]
            rows = np.arange(b)[:, None]
            w = np.maximum(sims[rows, idx], 0.0)           # (b, k)
            chars = self.tgt[idx]                          # (b, k)
            for j in range(kk):
                np.add.at(out[s:s + bs], (np.arange(b), chars[:, j]), w[:, j])
        return out

    def evaluate(self, ids):
        win, tgt = self._windows(ids)
        pred = self._votes(win).argmax(1)
        acc = float((pred == tgt).mean())
        return acc, float(self.unigram.max())

    def generate(self, seed_ids, n_chars, temperature=0.5, rng=None, pad_id=0):
        rng = rng or np.random.default_rng(0)
        ctx = list(seed_ids[-self.ctx:])
        while len(ctx) < self.ctx:
            ctx = [pad_id] + ctx
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
# Data collection -- several corpora, fetched on demand from GitHub-hosted,
# public sources and cached next to this file.
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = "https://raw.githubusercontent.com"

def _fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=120).read().decode("utf-8", "replace")

def _build_shakespeare(max_chars):
    return _fetch(f"{RAW}/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

def _build_science(max_chars):
    """A multi-domain scientific corpus: PubMed RCT abstracts (biology/medicine)
    mixed with arXiv abstracts (CS / AI / vision).  Cleaned to plain prose."""
    print("  fetching PubMed RCT abstracts (biology/medicine)...")
    pm = _fetch(f"{RAW}/Franck-Dernoncourt/pubmed-rct/master/PubMed_20k_RCT/train.txt")
    bio = "\n".join(ln.split("\t", 1)[1].strip() for ln in pm.split("\n") if "\t" in ln)
    print("  fetching arXiv abstracts (math / CS / AI)...")
    arx = "\n".join(_fetch(f"{RAW}/gcunhase/ArXivAbsTitleDataset/master/results/" + f)
                    for f in ["artificial%20intelligence_10047_15000_15_abs.txt",
                              "computer%20vision_14582_15000_15_abs.txt",
                              "language%20generation_14514_15000_15_abs.txt"])
    half = max_chars // 2
    return bio[:half].strip() + "\n\n" + arx[:max_chars - half].strip()

CORPORA = {"shakespeare": _build_shakespeare, "science": _build_science}

def collect_data(corpus="shakespeare", max_chars=10_000_000):
    path = os.path.join(HERE, f"corpus_{corpus}.txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    print(f"collecting '{corpus}' corpus (cached to {os.path.basename(path)})...")
    text = CORPORA[corpus](max_chars)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text

# --------------------------------------------------------------------------
# Shared setup
# --------------------------------------------------------------------------
def build_corpus(corpus="shakespeare", max_chars=10_000_000):
    text = collect_data(corpus, max_chars)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    return text, stoi, itos, len(chars), ids

def make_model(args, V):
    return OctonionFanoLM(V, slots=args.slots, ctx=args.ctx, topk=args.topk,
                          decay=args.decay, backend=args.backend, ann=args.ann,
                          lsh_bits=args.lsh_bits, lsh_tables=args.lsh_tables)

def train_model(args, ids, V, n_chars):
    train_ids = ids if n_chars <= 0 else ids[:n_chars]
    model = make_model(args, V)
    print(f"training on {len(train_ids):,} chars  (hypervector {args.slots} octonions "
          f"= {model.D} dims, context {args.ctx}, decay {args.decay}, backend {args.backend}"
          f"{', ANN' if args.ann else ''})...")
    t = time.time()
    model.train(train_ids)
    if model.backend == "binary":
        print(f"memory bank: {model.memb.shape[0]:,} contexts x {model.D} bits "
              f"({model.memb.nbytes/1e6:.0f} MB packed)  built in {time.time()-t:.1f}s")
    else:
        print(f"memory bank: {model.mem.shape[0]:,} contexts x {model.D} dims "
              f"({model.mem.nbytes/1e6:.0f} MB)  built in {time.time()-t:.1f}s")
    return model

# --------------------------------------------------------------------------
# Command: demo  (train, evaluate, print a sample)
# --------------------------------------------------------------------------
def cmd_demo(args):
    text, stoi, itos, V, ids = build_corpus(args.corpus, args.max_chars)
    print(f"corpus={args.corpus}  vocab={V}  total_chars={len(ids):,}")
    print("Fano lines (from the multiplication table):")
    for ln in fano_lines():
        print("   {%d, %d, %d}" % ln)
    model = train_model(args, ids, V, args.chars)

    n_train = len(ids) if args.chars <= 0 else args.chars
    eval_ids = ids[n_train:n_train + args.eval]
    if len(eval_ids) > args.ctx + 1:
        acc, base = model.evaluate(eval_ids)
        tag = "ANN" if args.ann else "exact"
        print(f"\nnext-char top-1 accuracy : {acc*100:5.2f}%   (k={args.topk}, {tag})")
        print(f"most-frequent-char base  : {base*100:5.2f}%")
        print(f"lift over baseline       : x{acc/base:4.2f}")

    gen = model.generate(ids[:args.ctx], args.gen, temperature=args.temp)
    print("\n--- generated sample (temp=%.2f) ---" % args.temp)
    print("".join(itos[i] for i in gen))
    print("--- end sample ---")

# --------------------------------------------------------------------------
# Command: serve  (interactive prompt UI in the browser)
# --------------------------------------------------------------------------
SERVE_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Octonionic Fano-Path LM</title>
<style>
  :root{--bg:#0a0a0f;--panel:rgba(255,255,255,.04);--border:rgba(255,255,255,.12);
        --accent:#9ad;--accent2:#d9a;--text:#e8e8ee;--muted:#8a8a99;}
  *{box-sizing:border-box} body{margin:0;min-height:100vh;
    background:radial-gradient(1200px 800px at 70% -10%,#14142a 0%,var(--bg) 60%);
    color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
    font-size:14px;padding:26px 22px 60px;}
  .wrap{max-width:900px;margin:0 auto}
  h1{font-size:22px;margin:0 0 2px} .sub{color:var(--muted);font-size:13px;margin-bottom:18px}
  .panel{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:16px;margin-bottom:16px}
  textarea{width:100%;min-height:90px;resize:vertical;background:#05050a;color:var(--text);
    border:1px solid var(--border);border-radius:8px;padding:10px;
    font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;line-height:1.5}
  .row{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-end;margin-top:12px}
  .ctl{flex:1;min-width:150px} .ctl label{display:flex;justify-content:space-between;color:var(--muted);font-size:13px}
  .ctl label b{color:var(--text);font-variant-numeric:tabular-nums}
  input[type=range]{width:100%;margin-top:6px;accent-color:var(--accent)}
  .btns{display:flex;gap:8px} button{background:#1a1a2e;color:var(--text);border:1px solid var(--border);
    border-radius:8px;padding:9px 16px;cursor:pointer;font-size:14px}
  button.go{background:linear-gradient(90deg,var(--accent),var(--accent2));color:#06060c;border:none;font-weight:600}
  button:disabled{opacity:.5;cursor:default}
  #out{background:#05050a;border:1px solid var(--border);border-radius:8px;padding:14px;margin-top:14px;
    min-height:160px;white-space:pre-wrap;word-break:break-word;
    font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13.5px;line-height:1.55}
  #out .seed{color:var(--muted)} #out .gen{color:var(--text)}
  #cursor{display:inline-block;width:7px;height:15px;background:var(--accent2);animation:b 1s steps(2) infinite;vertical-align:-2px}
  @keyframes b{0%,50%{opacity:1}50.01%,100%{opacity:0}}
  .meta{color:var(--muted);font-size:12px;margin-top:6px}
  code{background:#05050a;padding:1px 5px;border-radius:4px;border:1px solid var(--border)}
</style></head><body><div class="wrap">
  <h1>Octonionic Fano-Path LM</h1>
  <div class="sub">a mini language model trained with <b>no backpropagation</b> &mdash;
     next characters are recalled from nearest Fano-encoded contexts in octonion hyperspace</div>

  <div class="panel">
    <label style="color:var(--muted);font-size:13px">Prompt (the model continues your text, character by character)</label>
    <textarea id="prompt">ROMEO:
What light through yonder window breaks?</textarea>
    <div class="row">
      <div class="ctl"><label>Generate <b><span id="nv">300</span> chars</b></label>
        <input id="n" type="range" min="40" max="1200" value="300" step="20"></div>
      <div class="ctl"><label>Temperature <b><span id="tv">0.40</span></b></label>
        <input id="t" type="range" min="10" max="120" value="40"></div>
      <div class="btns"><button class="go" id="go">Generate</button><button id="stop" disabled>Stop</button></div>
    </div>
    <div class="meta" id="meta">loading model…</div>
  </div>

  <div id="out"></div>
</div>
<script>
const $=id=>document.getElementById(id);
$('n').oninput=()=>$('nv').textContent=$('n').value;
$('t').oninput=()=>$('tv').textContent=($('t').value/100).toFixed(2);
let stop=false;

fetch('/info').then(r=>r.json()).then(d=>{
  $('meta').innerHTML=`vocab <b>${d.vocab}</b> · memory <b>${d.contexts.toLocaleString()}</b> contexts `+
    `(<b>${d.backend}</b>) · context length <b>${d.ctx}</b> · ${d.slots} octonions/token (${d.dims}-dim) · `+
    `held-out next-char accuracy <b>${(d.acc*100).toFixed(1)}%</b> (baseline ${(d.base*100).toFixed(1)}%)`;
});

async function gen(){
  stop=false; $('go').disabled=true; $('stop').disabled=false;
  const seed=$('prompt').value, total=+$('n').value, temp=$('t').value/100;
  const out=$('out');
  out.innerHTML='<span class="seed"></span><span class="gen"></span><span id="cursor"></span>';
  out.querySelector('.seed').textContent=seed;
  let produced=0, text=seed;
  while(produced<total && !stop){
    const step=Math.min(40,total-produced);
    let r;
    try{ r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({text,n:step,temp})}); }
    catch(e){ break; }
    const d=await r.json();
    text+=d.gen; produced+=step;
    out.querySelector('.gen').textContent=text.slice(seed.length);
    out.scrollTop=out.scrollHeight;
  }
  const c=$('cursor'); if(c)c.remove();
  $('go').disabled=false; $('stop').disabled=true;
}
$('go').onclick=gen;
$('stop').onclick=()=>{stop=true;};
</script></body></html>"""

def cmd_serve(args):
    text, stoi, itos, V, ids = build_corpus(args.corpus, args.max_chars)
    space = stoi.get(' ', 0)

    # hold out a small tail for an honest banner score, train on the rest
    HELD = 4000
    n_train = (len(ids) - HELD) if args.chars <= 0 else args.chars

    model = make_model(args, V)
    # cache the (slow-to-build) memory bank so re-serving is instant
    key = f"{args.corpus}|{n_train}|{args.ctx}|{args.slots}|{args.decay}|{args.backend}"
    cache = os.path.join(HERE, f".memcache_{hashlib.md5(key.encode()).hexdigest()[:12]}.npz")
    if args.backend == "binary" and os.path.exists(cache):
        print(f"loading cached memory bank ({os.path.basename(cache)})...")
        model.load_memory(cache)
    else:
        model = train_model(args, ids, V, n_train)
        if args.backend == "binary":
            model.save_memory(cache)
            print(f"cached memory bank to {os.path.basename(cache)}")

    # held-out score (with whatever recall path generation will use)
    ev = ids[n_train:n_train + HELD]
    if len(ev) > args.ctx + 1:
        acc, base = model.evaluate(ev)
    else:
        acc, base = 0.0, float(model.unigram.max())
    tag = "ANN" if args.ann else "exact"
    print(f"held-out next-char accuracy: {acc*100:.1f}%  (baseline {base*100:.1f}%, {tag})")

    contexts = int(model.memb.shape[0] if model.backend == "binary" else model.mem.shape[0])
    if model.backend == "binary":
        print(f"packed memory: {model.memb.nbytes/1e6:.0f} MB bipolar hypervectors"
              f"{' + LSH index' if model.lsh else ''}")

    def to_ids(s):
        return [stoi.get(c, space) for c in s]

    info = {"vocab": V, "contexts": contexts, "ctx": args.ctx,
            "backend": model.backend + ("+ANN" if model.lsh else ""),
            "slots": args.slots, "dims": model.D, "acc": acc, "base": base}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass
        def _send(self, code, body, ctype="application/json"):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                self._send(200, SERVE_HTML, "text/html; charset=utf-8")
            elif self.path == "/info":
                self._send(200, json.dumps(info))
            else:
                self._send(404, "{}")
        def do_POST(self):
            if self.path != "/generate":
                self._send(404, "{}"); return
            ln = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(ln) or b"{}")
            prompt = req.get("text", "")
            n = max(1, min(int(req.get("n", 40)), 400))
            temp = float(req.get("temp", 0.4))
            seed_ids = to_ids(prompt) or [space]
            rng = np.random.default_rng()
            out = model.generate(seed_ids, n, temperature=temp, rng=rng, pad_id=space)
            self._send(200, json.dumps({"gen": "".join(itos[i] for i in out)}))

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"\nserving the prompt UI at  {url}\n(press Ctrl+C to stop)")
    if not args.no_open:
        threading.Thread(target=lambda: (time.sleep(1), webbrowser.open(url)), daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down."); srv.shutdown()

# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Octonionic Fano-path mini language model (no backprop)")
    ap.add_argument("mode", nargs="?", default="demo", choices=["demo", "serve"],
                    help="demo: train+evaluate+sample | serve: interactive prompt UI")
    ap.add_argument("--corpus", choices=list(CORPORA), default="shakespeare",
                    help="shakespeare (~1.1M chars) or science (multi-domain, up to ~10M)")
    ap.add_argument("--max-chars", type=int, default=10_000_000,
                    help="cap on the science corpus size when first built")
    ap.add_argument("--chars", type=int, default=None,
                    help="training characters (<=0 or omitted in serve = whole corpus)")
    ap.add_argument("--eval", type=int, default=8_000, help="held-out characters (demo)")
    ap.add_argument("--ctx", type=int, default=12, help="context length")
    ap.add_argument("--decay", type=float, default=0.5, help="recency weight per position (decay**p)")
    ap.add_argument("--slots", type=int, default=64, help="octonions per hypervector (dim = 8*slots)")
    ap.add_argument("--topk", type=int, default=7, help="nearest contexts to recall")
    ap.add_argument("--backend", choices=["float", "binary"], default=None,
                    help="memory format: binary = 32x smaller bipolar hypervectors "
                         "(serve default); float = faster/slightly more accurate (demo default)")
    ap.add_argument("--ann", dest="ann", action="store_true", default=None,
                    help="approximate nearest neighbours (LSH) -- needed for large corpora")
    ap.add_argument("--no-ann", dest="ann", action="store_false",
                    help="force exact Hamming search")
    ap.add_argument("--lsh-bits", type=int, default=22, help="LSH bits sampled per table")
    ap.add_argument("--lsh-tables", type=int, default=8, help="number of LSH tables")
    ap.add_argument("--gen", type=int, default=600, help="characters to generate (demo)")
    ap.add_argument("--temp", type=float, default=0.4, help="sampling temperature")
    ap.add_argument("--host", default="127.0.0.1", help="serve host")
    ap.add_argument("--port", type=int, default=8000, help="serve port")
    ap.add_argument("--no-open", action="store_true", help="do not auto-open the browser")
    args = ap.parse_args()

    if args.chars is None:
        args.chars = 0 if args.mode == "serve" else 150_000
    if args.backend is None:
        args.backend = "binary" if args.mode == "serve" else "float"
    if args.ann is None:
        # default ON for binary serve (essential for large corpora), else OFF
        args.ann = (args.mode == "serve" and args.backend == "binary")

    if args.mode == "serve":
        cmd_serve(args)
    else:
        cmd_demo(args)

if __name__ == "__main__":
    main()
