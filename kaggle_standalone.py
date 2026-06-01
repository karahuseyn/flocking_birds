# kaggle_standalone.py -- ONE self-contained cell. No git clone, no internet needed.
# numpy + scipy are pre-installed on Kaggle. Set Accelerator = None (CPU).
#
# CORPUS: Kaggle notebooks have internet OFF by default, so we do NOT download.
#   - Easiest: "+ Add Input" -> search a text dataset (e.g. wikitext) or upload your own,
#     then set CORPUS_PATH below to the .txt it provides (look under /kaggle/input/...).
#   - If CORPUS_PATH is missing, a tiny built-in demo corpus is used so the cell still runs.
#
# Everything here is gradient-free, O(n), pure ASCII.

import os, re, glob, time
import numpy as np
from collections import Counter, defaultdict
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds

# ---- find a corpus (txt / csv / parquet / json under /kaggle/input) ----------
CORPUS_PATH = ""        # optional: hard-set a file; else auto-find the biggest text under /kaggle/input
MAX_CHARS = 400_000_000 # cap (~70M words); raise toward 1-2 GB on Kaggle's 30 GB RAM

def _read_any(path, cap):
    """Read text from .txt/.csv/.parquet/.json* -- pick the longest string column."""
    low = path.lower()
    if low.endswith(".txt"):
        return open(path, encoding="utf-8", errors="ignore").read(cap)
    try:
        import pandas as pd
        if low.endswith(".parquet"):
            df = pd.read_parquet(path)
        elif low.endswith(".csv"):
            df = pd.read_csv(path, on_bad_lines="skip", engine="python", nrows=2_000_000)
        elif low.endswith((".json", ".jsonl")):
            df = pd.read_json(path, lines=low.endswith(".jsonl"))
        else:
            return ""
        objcols = [c for c in df.columns if df[c].dtype == object]
        if not objcols:
            return ""
        best = max(objcols, key=lambda c: df[c].astype(str).str.len().head(1000).mean())
        return "\n".join(df[best].dropna().astype(str).tolist())[:cap]
    except Exception as e:
        print("  (could not parse %s: %s)" % (path, str(e)[:60]))
        return ""

TEXT = ""
if CORPUS_PATH and os.path.exists(CORPUS_PATH):
    TEXT = _read_any(CORPUS_PATH, MAX_CHARS); print("corpus:", CORPUS_PATH)
else:
    import glob as _g
    files = []
    for e in ("*.txt", "*.parquet", "*.csv", "*.json", "*.jsonl"):
        files += _g.glob("/kaggle/input/**/" + e, recursive=True)
    files = sorted(files, key=lambda p: -os.path.getsize(p))
    print("found %d candidate files under /kaggle/input" % len(files))
    for f in files[:5]:
        print("   %6.0f MB  %s" % (os.path.getsize(f)/1e6, f))
    for f in files:
        TEXT = _read_any(f, MAX_CHARS)
        if len(TEXT) > 10000:
            print("using:", f, "(%.0f MB of text)" % (len(TEXT)/1e6)); break
if len(TEXT) < 10000:
    TEXT = ("the king sat in his hall and the queen came to him . the people loved the king "
            "and the king loved the people . in the morning the sun rose over the city . "
            "she looked at the river and the river was calm and bright . ") * 400
    print("\n*** NO dataset found under /kaggle/input -- running on a tiny demo. ***")
    print("*** To use BIG data: click '+ Add Input' (right panel), add a text dataset,")
    print("*** then re-run. Good ones (search these names): 'wikitext', 'bookcorpus',")
    print("*** 'wikipedia plain text', 'arxiv', 'pubmed'.  Or upload your own .txt. ***\n")

# ---- octonion algebra (the Fano-plane multiplication, vectorised) ------------
def _qmul(x, y):
    a0, a1, a2, a3 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    b0, b1, b2, b3 = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
    return np.stack([a0*b0-a1*b1-a2*b2-a3*b3, a0*b1+a1*b0+a2*b3-a3*b2,
                     a0*b2-a1*b3+a2*b0+a3*b1, a0*b3+a1*b2-a2*b1+a3*b0], -1)

def octo_mul(a, b):
    p, q = a[..., :4], a[..., 4:]; r, s = b[..., :4], b[..., 4:]
    cs = s.copy(); cs[..., 1:] *= -1; cr = r.copy(); cr[..., 1:] *= -1
    return np.concatenate([_qmul(p, r) - _qmul(cs, q), _qmul(s, p) + _qmul(q, cr)], -1)

def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

def _conj(a):
    o = a.copy(); o[..., 1:] *= -1; return o

def _inv(a):
    return _conj(a) / ((a * a).sum(-1, keepdims=True) + 1e-12)

def _fano_role(steps, K):
    acc = np.zeros(8); acc[0] = 1.0
    for g in steps:
        e = np.zeros(8); e[g] = 1.0; acc = octo_mul(acc, e)
    return np.tile(acc, (K, 1))

VSUF = ("ed", "ing", "es", "ize", "ise", "ate", "fy")

# ---- build (sparse PPMI -> truncated SVD + n-grams), gradient-free -----------
def build(text, vocab_size=40000, K=12, window=5, shift=5.0):
    t0 = time.time()
    words = re.findall(r"[a-z']+", text.lower())
    vocab = [w for w, _ in Counter(words).most_common(vocab_size)]
    wi = {w: i for i, w in enumerate(vocab)}; W = len(vocab)
    ids = np.array([wi[w] for w in words if w in wi], dtype=np.int64)
    print("  %d words, vocab %d" % (len(ids), W))
    C = csr_matrix((W, W), dtype=np.float32)
    for d in range(1, window + 1):
        a, b = ids[:-d], ids[d:]; data = np.full(len(a), np.float32(1.0/d), np.float32)
        Cd = coo_matrix((data, (a, b)), shape=(W, W)).tocsr(); C = C + Cd + Cd.T
    tot = C.sum(); Pa = np.asarray(C.sum(1)).ravel() / tot; Cx = C.tocoo()
    pmi = np.log(Cx.data/tot/(Pa[Cx.row]*Pa[Cx.col]+1e-30)+1e-12) - np.log(shift)
    keep = pmi > 0
    P = coo_matrix((pmi[keep], (Cx.row[keep], Cx.col[keep])), shape=(W, W)).tocsr()
    D = 8 * K
    if W <= D + 1:
        Ud, Sd, _ = np.linalg.svd(P.toarray(), full_matrices=False); U, S = Ud[:, :D], Sd[:D]
    else:
        U, S, _ = svds(P, k=min(D, W - 1)); o = np.argsort(S)[::-1]; U, S = U[:, o], S[o]
    if U.shape[1] < D:
        U = np.pad(U, ((0, 0), (0, D - U.shape[1]))); S = np.pad(S, (0, D - len(S)))
    emb = unit(U * np.sqrt(S))
    idl = ids.tolist(); tri = defaultdict(Counter); bi = defaultdict(Counter)
    for i in range(len(idl)-1): bi[idl[i]][idl[i+1]] += 1
    for i in range(len(idl)-2): tri[(idl[i], idl[i+1])][idl[i+2]] += 1
    act = np.array([any(vocab[i].endswith(s) for s in VSUF) and len(vocab[i]) > 4 for i in range(W)])
    print("  built in %.0f s" % (time.time()-t0))
    return dict(vocab=vocab, wi=wi, W=W, K=K, emb=emb, embK=emb.reshape(W, K, 8),
                tri=tri, bi=bi, act=act, RS=_fano_role([1, 2], K), RP=_fano_role([3, 4], K))

# ---- generate (discourse drift + octonion proposition + alternation + veto) --
def generate(M, seed, n=60, temp=0.5, decay=0.8, drift=0.18,
             w_subj=1.5, w_pred=3.5, w_alt=2.0, rep_pen=2.0, veto=True, rng_seed=1):
    vocab, wi, W, emb, embK = M["vocab"], M["wi"], M["W"], M["emb"], M["embK"]
    tri, bi, act, RS, RP = M["tri"], M["bi"], M["act"], M["RS"], M["RP"]
    rng = np.random.default_rng(rng_seed)
    out = [wi[w] for w in seed.lower().split() if w in wi] or [int(rng.integers(W))]
    s = unit(emb[out].mean(0)); cvec = s.copy(); SK = np.zeros((M["K"], 8))
    for x in out: SK = decay*SK + octo_mul(RP if act[x] else RS, embK[x])
    recent = {}; since = 0; seen = set()
    for _ in range(n):
        c = tri.get((out[-2], out[-1])) if len(out) >= 2 else None
        if not c: c = bi.get(out[-1])
        if c:
            cand = np.array(list(c)); fr = np.array([c[x] for x in cand], float)
            if veto and len(out) >= 1 and len(cand) > 1:
                k = np.array([(out[-1], int(x)) not in seen for x in cand])
                if k.any(): cand, fr = cand[k], fr[k]
            nz = lambda x: (x-x.min())/(np.ptp(x)+1e-9) if len(cand) > 1 else x*0
            es = unit(octo_mul(_inv(RS), SK).reshape(-1)); ep = unit(octo_mul(_inv(RP), SK).reshape(-1))
            want = 1.0 if since >= 2 else -0.5
            al = np.array([want if act[x] else 0.0 for x in cand])
            rp = np.array([recent.get(int(x), 0) for x in cand], float)
            sc = (np.log(fr) + nz(emb[cand]@cvec) + 1.5*nz(emb[cand]@s)
                  + w_subj*nz(emb[cand]@es) + w_pred*nz(emb[cand]@ep) + w_alt*al - rep_pen*rp)
            p = np.exp(sc/temp); p /= p.sum(); nxt = int(rng.choice(cand, p=p))
        else:
            nxt = int(rng.integers(W))
        if out: seen.add((out[-1], nxt))
        out.append(nxt); recent = {k: v*0.6 for k, v in recent.items()}; recent[nxt] = recent.get(nxt, 0)+1
        since = 0 if act[nxt] else since+1
        cvec = 0.85*cvec + 0.15*emb[nxt]; s = unit((1-drift)*s + drift*emb[nxt])
        SK = decay*SK + octo_mul(RP if act[nxt] else RS, embK[nxt])
    return " ".join(vocab[i] for i in out)

# ---- logic-guided QA: parse rules, derive transitive answers ----------------
_RULE = re.compile(r"\b([a-z]+(?:\s+[a-z]+)*?)\s+(?:causes|leads to|implies|results in)\s+([a-z]+(?:\s+[a-z]+)*?)\s*[.;]", re.I)
_QRY = re.compile(r"\b(?:does|can|will)\s+([a-z]+(?:\s+[a-z]+)*?)\s+(?:lead to|cause|imply)\s+([a-z]+(?:\s+[a-z]+)*)", re.I)

class QA:
    def __init__(s, seed=0): s.rng = np.random.default_rng(seed); s.f = {}; s.rules = []
    def _fact(s, n):
        n = n.strip().lower()
        if n not in s.f: s.f[n] = unit(s.rng.standard_normal(8))
        return n
    def read(s, text):
        for m in _RULE.finditer(text): s.rules.append((s._fact(m.group(1)), s._fact(m.group(2))))
    def prove(s, a, b, depth=8):
        if a not in s.f or b not in s.f: return None
        front = [(a, s.f[a], [a])]
        for _ in range(depth):
            nx = []
            for cur, vec, path in front:
                for (i, j) in s.rules:
                    if i == cur:
                        v2 = octo_mul(octo_mul(s.f[j], _inv(s.f[i])), vec)
                        if j == b: return path + [j]
                        nx.append((j, v2, path + [j]))
            front = nx
            if not front: break
        return False
    def answer(s, q):
        m = _QRY.search(q)
        if not m: return "?"
        a, b = m.group(1).strip().lower(), m.group(2).strip().lower()
        if a not in s.f or b not in s.f: return "I don't know that concept."
        p = s.prove(a, b)
        if p: return "Yes: " + " -> ".join(p) + " (%d steps)" % (len(p)-1)
        return "No chain from %s to %s." % (a, b)

# ============================ RUN =============================================
M = build(TEXT, vocab_size=40000)
print("\n-- semantic neighbours --")
emb, wi, vocab = M["emb"], M["wi"], M["vocab"]
for w in ["king", "war", "science", "love", "city"]:
    if w in wi:
        print("%-8s ->" % w, [vocab[i] for i in (emb@emb[wi[w]]).argsort()[::-1][1:7]])
print("\n-- generation --")
for pr in ["the king", "in the morning", "she looked at"]:
    print("\n" + pr + " |||\n  " + generate(M, pr, n=50))
print("\n-- logic QA (derivation) --")
qa = QA(); qa.read("fever causes infection. infection causes inflammation. "
                   "inflammation causes pain. fever causes fatigue.")
for q in ["does fever lead to pain?", "does fever lead to fatigue?", "does pain lead to fever?"]:
    print(q, "->", qa.answer(q))
