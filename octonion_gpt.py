"""octonion_gpt.py -- the end-to-end gradient-free generator: every verified piece, one model.

No backprop, no gradients, O(n).  Combines, on one large corpus, all the components each
proven in its own module:

  meaning      PMI-SVD semantic embedding (octonion_semantic)         -- real geometry
  fluency      trigram backbone (counts)                              -- local grammar
  topic flow   evolving discourse state s <- (1-r)s + r*emb           -- long-range topic
  proposition  octonion BIND state  S <- decay*S + role (x) word,     -- subject/predicate
               next word scored by exact unbind (octonion_proposition)
  rhythm       verb/noun alternation drive                            -- syntactic cadence

Each force is a base64-verified improvement on a discourse metric (local-coherence / drift /
action-oscillation matched to real text).  This file orchestrates them into one scorer.
"""
import re, sys, time
import numpy as np
from collections import Counter, defaultdict
from octonion_lm import octo_mul

VSUF = ("ed", "ing", "es", "ize", "ise", "ate", "fy")

def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

def _fano_role(steps, K):
    acc = np.zeros(8); acc[0] = 1.0
    for g in steps:
        e = np.zeros(8); e[g] = 1.0; acc = octo_mul(acc, e)
    return np.tile(acc, (K, 1))

def _conj(a):
    o = a.copy(); o[..., 1:] *= -1; return o

def _inv(a):
    return _conj(a) / ((a * a).sum(-1, keepdims=True) + 1e-12)

def build(text, vocab_size=10000, K=12, window=5, shift=5.0, verbose=True):
    t0 = time.time()
    words = re.findall(r"[a-z']+", text.lower())
    vc = Counter(words); vocab = [w for w, _ in vc.most_common(vocab_size)]
    wi = {w: i for i, w in enumerate(vocab)}; W = len(vocab)
    ids = np.array([wi[w] for w in words if w in wi], dtype=np.int64)
    if verbose: print(f"  corpus {len(words):,} words -> {len(ids):,} in-vocab, vocab {W}")
    # co-occurrence -> shifted PPMI -> SVD  (gradient-free semantic embedding)
    # vectorised: for each offset d, scatter-add the shifted id pairs at weight 1/d
    C = np.zeros((W, W), dtype=np.float32)
    for d in range(1, window + 1):
        a = ids[:-d]; b = ids[d:]; w = 1.0 / d
        np.add.at(C, (a, b), w); np.add.at(C, (b, a), w)
    ids = ids.tolist()
    tot = C.sum(); Pa = C.sum(1) / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        PMI = np.log((C / tot) / (np.outer(Pa, Pa) + 1e-30) + 1e-12)
    PPMI = np.maximum(PMI - np.log(shift), 0.0).astype(np.float32)
    D = 8 * K
    if verbose: print(f"  PPMI built ({time.time()-t0:.0f}s), randomized SVD (top {D}) on {W}x{W}...")
    # randomized-projection truncated SVD: keeps the TOP D components (correct semantics,
    # unlike scipy.svds which returned the smallest), and is fast for large vocab.
    rng = np.random.default_rng(0)
    Om = rng.standard_normal((W, D + 30)).astype(np.float32)
    Y = PPMI @ Om
    for _ in range(2):                            # power iterations sharpen the top subspace
        Y = PPMI @ (PPMI.T @ Y)
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ PPMI
    Ub, S, _ = np.linalg.svd(B, full_matrices=False)
    U = (Q @ Ub)[:, :D]; S = S[:D]
    emb = unit(U * np.sqrt(S))
    tri = defaultdict(Counter); bi = defaultdict(Counter)
    for t in range(len(ids) - 1):
        bi[ids[t]][ids[t + 1]] += 1
    for t in range(len(ids) - 2):
        tri[(ids[t], ids[t + 1])][ids[t + 2]] += 1
    is_action = np.array([any(vocab[i].endswith(s) for s in VSUF) and len(vocab[i]) > 4
                          for i in range(W)])
    if verbose: print(f"  model ready in {time.time()-t0:.0f}s")
    return dict(vocab=vocab, wi=wi, W=W, K=K, emb=emb, embK=emb.reshape(W, K, 8),
                tri=tri, bi=bi, is_action=is_action,
                role_subj=_fano_role([1, 2], K), role_pred=_fano_role([3, 4], K))

def generate(M, seed, n=60, temp=0.5, decay=0.8, drift=0.18,
             w_flow=1.0, w_subj=1.5, w_pred=3.5, w_alt=2.0, rep_pen=2.0, rng_seed=1):
    vocab, wi, W, emb, embK = M["vocab"], M["wi"], M["W"], M["emb"], M["embK"]
    tri, bi, act = M["tri"], M["bi"], M["is_action"]
    RS, RP = M["role_subj"], M["role_pred"]
    rng = np.random.default_rng(rng_seed)
    out = [wi[w] for w in seed.lower().split() if w in wi] or [int(rng.integers(W))]
    s = unit(emb[out].mean(0)); cvec = s.copy()
    SK = np.zeros((M["K"], 8))
    for x in out:
        SK = decay * SK + octo_mul(RP if act[x] else RS, embK[x])
    recent = {}; since_act = 0
    for _ in range(n):
        cnt = tri.get((out[-2], out[-1])) if len(out) >= 2 else None
        if not cnt:
            cnt = bi.get(out[-1])
        if cnt:
            cand = np.array(list(cnt)); freq = np.array([cnt[c] for c in cand], float)
            def nrm(x): return (x - x.min()) / (np.ptp(x) + 1e-9) if len(cand) > 1 else x * 0
            flo = nrm(emb[cand] @ cvec)                       # local smoothness
            dis = nrm(emb[cand] @ s)                          # evolving topic
            es = unit(octo_mul(_inv(RS), SK).reshape(-1))     # unbind subject filler
            ep = unit(octo_mul(_inv(RP), SK).reshape(-1))     # unbind predicate filler
            ss = nrm(emb[cand] @ es); sp = nrm(emb[cand] @ ep)
            want = 1.0 if since_act >= 2 else -0.5
            alt = np.array([want if act[c] else 0.0 for c in cand])   # syntactic rhythm
            rep = np.array([recent.get(int(c), 0) for c in cand], float)
            score = (np.log(freq) + w_flow * flo + 1.5 * dis
                     + w_subj * ss + w_pred * sp + w_alt * alt - rep_pen * rep)
            p = np.exp(score / temp); p /= p.sum()
            nxt = int(rng.choice(cand, p=p))
        else:
            nxt = int(rng.integers(W))
        out.append(nxt)
        recent = {k: v * 0.6 for k, v in recent.items()}; recent[nxt] = recent.get(nxt, 0) + 1
        since_act = 0 if act[nxt] else since_act + 1
        cvec = (1 - 0.15) * cvec + 0.15 * emb[nxt]            # fast local flow
        s = unit((1 - drift) * s + drift * emb[nxt])          # slow discourse drift
        SK = decay * SK + octo_mul(RP if act[nxt] else RS, embK[nxt])
    return " ".join(vocab[i] for i in out)

def main():
    import base64
    which = sys.argv[1] if len(sys.argv) > 1 else "biomed"
    paths = {"biomed": "corpus_biomed.txt", "science": "corpus_science.txt",
             "shakespeare": "corpus_shakespeare.txt"}
    cap = {"biomed": 60_000_000, "science": 40_000_000, "shakespeare": None}
    p = paths.get(which, which); c = cap.get(which, 60_000_000)
    text = open(p, encoding="utf-8").read(c) if c else open(p, encoding="utf-8").read()
    print(f"building end-to-end gradient-free model on '{which}' (large corpus)...")
    M = build(text, vocab_size=10000)
    prompts = (sys.argv[2:] or
               ["patients with diabetes", "the treatment reduced",
                "we investigated whether", "in this study we"])
    outs = []
    for s in prompts:
        g = generate(M, s, n=60); outs.append(g); print(f"\n{s} |||\n  {g}")
    print("\nB64GEN:" + base64.b64encode("\n\n".join(outs).encode()).decode())

if __name__ == "__main__":
    main()
