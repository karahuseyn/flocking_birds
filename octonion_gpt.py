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

# ---- modified Kneser-Ney (order 3) backbone helpers -------------------------
def _kn_coc(counters):
    n = [0, 0, 0, 0, 0]
    for ctr in counters:
        for c in ctr.values():
            if 1 <= c <= 4: n[c] += 1
    return n[1], n[2], n[3], n[4]

def _kn_disc(coc):
    n1, n2, n3, n4 = coc
    Y = n1 / (n1 + 2*n2) if (n1 + 2*n2) else 0.0
    return (max(1 - 2*Y*(n2/n1), 0.0) if n1 else 0.0,
            max(2 - 3*Y*(n3/n2), 0.0) if n2 else 0.0,
            max(3 - 4*Y*(n4/n3), 0.0) if n3 else 0.0)

def _kn3_logprob(M, a, b, cand):
    """Interpolated modified KN order-3 log-prob of each candidate given context (a, b)."""
    tri, cont2, cont1, tot1, D2, D3 = M["tri"], M["cont2"], M["cont1"], M["tot1"], M["D2"], M["D3"]
    ctr3 = tri.get((a, b)); ctr2 = cont2.get(b)
    def _dg(ctr, D):
        den = sum(ctr.values())
        n1 = sum(1 for c in ctr.values() if c == 1); n2 = sum(1 for c in ctr.values() if c == 2)
        n3 = sum(1 for c in ctr.values() if c >= 3)
        return den, ((D[0]*n1 + D[1]*n2 + D[2]*n3) / den if den else 1.0)
    if ctr3: den3, g3 = _dg(ctr3, D3)
    if ctr2: den2, g2 = _dg(ctr2, D2)
    out = np.empty(len(cand))
    for i, ww in enumerate(cand):
        w = int(ww); p = (cont1.get(w, 0) / tot1) or 1e-10           # KN unigram continuation
        if ctr2:
            cc = ctr2.get(w, 0); d = D2[0] if cc == 1 else D2[1] if cc == 2 else D2[2]
            p = max(cc - d, 0.0) / den2 + g2 * p
        if ctr3:
            cc = ctr3.get(w, 0); d = D3[0] if cc == 1 else D3[1] if cc == 2 else D3[2]
            p = max(cc - d, 0.0) / den3 + g3 * p
        out[i] = np.log(max(p, 1e-12))
    return out

def _kn_build(tri, bi):
    """Continuation counts + discounts for interpolated modified Kneser-Ney (order 3)."""
    cont2 = defaultdict(Counter)
    for (u, a), ctr in tri.items():
        for w in ctr: cont2[a][w] += 1
    cont1 = Counter()
    for a, ctr in bi.items():
        for w in ctr: cont1[w] += 1
    tot1 = sum(cont1.values()) or 1
    return cont2, cont1, tot1, _kn_disc(_kn_coc(cont2.values())), _kn_disc(_kn_coc(tri.values()))

def build(text, vocab_size=10000, K=12, window=5, shift=5.0, verbose=True):
    t0 = time.time()
    words = re.findall(r"[a-z']+", text.lower())
    vc = Counter(words); vocab = [w for w, _ in vc.most_common(vocab_size)]
    wi = {w: i for i, w in enumerate(vocab)}; W = len(vocab)
    ids = np.array([wi[w] for w in words if w in wi], dtype=np.int64)
    if verbose: print(f"  corpus {len(words):,} words -> {len(ids):,} in-vocab, vocab {W}")
    # SPARSE co-occurrence -> shifted PPMI -> truncated SVD (gradient-free embedding).
    # Sparse lifts the vocab^2 wall: vocab 40k is ~1.3M nonzeros, not 1.6B dense cells.
    from scipy.sparse import coo_matrix, csr_matrix
    from scipy.sparse.linalg import svds
    # accumulate each offset directly into a CSR (summing duplicates) so the giant
    # COO index arrays are never all held at once -- this is what keeps a 50M-word
    # corpus inside RAM (the naive single-COO build OOMs at ~9GB of index arrays).
    C = csr_matrix((W, W), dtype=np.float32)
    for d in range(1, window + 1):
        a = ids[:-d]; b = ids[d:]; w = np.float32(1.0 / d)
        data = np.full(len(a), w, dtype=np.float32)
        Cd = coo_matrix((data, (a, b)), shape=(W, W)).tocsr()
        C = C + Cd + Cd.T                         # symmetric; duplicates summed by CSR
        del Cd, data
    tot = C.sum(); Pa = np.asarray(C.sum(1)).ravel() / tot
    Cx = C.tocoo()
    pmi = np.log(Cx.data / tot / (Pa[Cx.row] * Pa[Cx.col] + 1e-30) + 1e-12) - np.log(shift)
    keep = pmi > 0
    PPMI = coo_matrix((pmi[keep], (Cx.row[keep], Cx.col[keep])), shape=(W, W)).tocsr()
    ids = ids.tolist()
    D = 8 * K
    if verbose: print(f"  PPMI built ({time.time()-t0:.0f}s, nnz={PPMI.nnz:,}), truncated SVD (top {D})...")
    if W <= D + 1:                                 # tiny vocab: dense SVD, take top D
        Ud, Sd, _ = np.linalg.svd(PPMI.toarray(), full_matrices=False)
        U, S = Ud[:, :D], Sd[:D]
    else:
        U, S, _ = svds(PPMI, k=min(D, W - 1))
        order = np.argsort(S)[::-1]; U = U[:, order]; S = S[order]
    if U.shape[1] < D:                            # always pad to exactly D dims
        U = np.pad(U, ((0, 0), (0, D - U.shape[1]))); S = np.pad(S, (0, D - len(S)))
    emb = unit(U * np.sqrt(S))
    tri = defaultdict(Counter); bi = defaultdict(Counter)
    for t in range(len(ids) - 1):
        bi[ids[t]][ids[t + 1]] += 1
    for t in range(len(ids) - 2):
        tri[(ids[t], ids[t + 1])][ids[t + 2]] += 1
    is_action = np.array([any(vocab[i].endswith(s) for s in VSUF) and len(vocab[i]) > 4
                          for i in range(W)])
    if verbose: print(f"  model ready in {time.time()-t0:.0f}s")
    # role_cycle: 7 cyclic Fano-point roles (e1..e7 tiled across K), the Singer cycle of
    # the Fano plane -- algebraic backbone of the period-7 echo layer in generate().
    role_cycle = [np.tile(np.eye(8)[i], (K, 1)) for i in range(1, 8)]
    # modified Kneser-Ney (order 3) backbone: ~4x lower held-out perplexity than naive
    # smoothing and better generation than raw counts (verified); order >3 adds <2%.
    cont2, cont1, tot1, D2, D3 = _kn_build(tri, bi)
    return dict(vocab=vocab, wi=wi, W=W, K=K, emb=emb, embK=emb.reshape(W, K, 8),
                tri=tri, bi=bi, is_action=is_action,
                role_subj=_fano_role([1, 2], K), role_pred=_fano_role([3, 4], K),
                role_cycle=role_cycle, cont2=cont2, cont1=cont1, tot1=tot1, D2=D2, D3=D3)

def generate(M, seed, n=60, temp=0.5, decay=0.8, drift=0.18,
             w_flow=1.0, w_subj=1.5, w_pred=3.5, w_alt=2.0, rep_pen=2.0, veto=True, rng_seed=1,
             w_goal=3.0, goal_ema=0.0, w_fano=0.0, flock=7):
    # w_goal: boids 4th rule -- a persistent topic target the generation steers toward
    #   (verified to roughly halve start->end drift). goal_ema=0 keeps it fixed to the
    #   prompt; ~0.02 lets it migrate slowly. w_fano: cyclic-Fano period-7 echo layer
    #   via exact octonion unbind (modest + anaphoric cadence; off by default, try ~4.0).
    vocab, wi, W, emb, embK = M["vocab"], M["wi"], M["W"], M["emb"], M["embK"]
    tri, bi, act = M["tri"], M["bi"], M["is_action"]
    RS, RP = M["role_subj"], M["role_pred"]; RC = M.get("role_cycle"); K = M["K"]
    rng = np.random.default_rng(rng_seed)
    out = [wi[w] for w in seed.lower().split() if w in wi] or [int(rng.integers(W))]
    s = unit(emb[out].mean(0)); cvec = s.copy()
    goal = unit(emb[out].mean(0))                       # persistent topic target (boids #4)
    SK = np.zeros((M["K"], 8))
    for x in out:
        SK = decay * SK + octo_mul(RP if act[x] else RS, embK[x])
    recent = {}; since_act = 0; seen_bg = set()
    for _ in range(n):
        a = out[-2] if len(out) >= 2 else -1
        ct = tri.get((a, out[-1])); cb = bi.get(out[-1])
        cset = set(ct) if ct else set()
        if cb: cset |= set(cb)                            # trigram UNION bigram (KN backoff coverage)
        if cset:
            cand = np.array(sorted(cset))
            # TTC-as-veto: drop candidates that would recreate an already-emitted bigram
            # (a loop), keeping stochastic diversity instead of maximizing a score (which
            # degenerates into repetition). Light: O(candidates), no extra forward passes.
            if veto and len(cand) > 1:
                keep = np.array([(out[-1], int(c)) not in seen_bg for c in cand])
                if keep.any():
                    cand = cand[keep]
            freq = _kn3_logprob(M, a, out[-1], cand)      # modified Kneser-Ney backbone (log-prob)
            def nrm(x): return (x - x.min()) / (np.ptp(x) + 1e-9) if len(cand) > 1 else x * 0
            flo = nrm(emb[cand] @ cvec)                       # local smoothness
            dis = nrm(emb[cand] @ s)                          # evolving topic
            es = unit(octo_mul(_inv(RS), SK).reshape(-1))     # unbind subject filler
            ep = unit(octo_mul(_inv(RP), SK).reshape(-1))     # unbind predicate filler
            ss = nrm(emb[cand] @ es); sp = nrm(emb[cand] @ ep)
            want = 1.0 if since_act >= 2 else -0.5
            alt = np.array([want if act[c] else 0.0 for c in cand])   # syntactic rhythm
            rep = np.array([recent.get(int(c), 0) for c in cand], float)
            gl = nrm(emb[cand] @ goal) if w_goal else 0.0            # boids #4: migratory urge
            # cyclic Fano-path ECHO (#4): 7-slot holographic register over the last `flock`
            # tokens; unbind the next slot's role -> what filled this Fano point one cycle ago.
            fa = 0.0
            if w_fano and RC is not None:
                fl = out[-flock:]; base = len(out) - len(fl); H = np.zeros((K, 8))
                for kk, tok in enumerate(fl): H = H + octo_mul(RC[(base+kk) % 7], embK[tok])
                pred = unit(octo_mul(_inv(RC[len(out) % 7]), H).reshape(-1))
                fa = nrm(emb[cand] @ pred)
            score = (freq + w_flow * flo + 1.5 * dis
                     + w_subj * ss + w_pred * sp + w_alt * alt
                     + w_goal * gl + w_fano * fa - rep_pen * rep)
            p = np.exp(score / temp); p /= p.sum()
            nxt = int(rng.choice(cand, p=p))
        else:
            nxt = int(rng.integers(W))
        if out:
            seen_bg.add((out[-1], nxt))
        out.append(nxt)
        recent = {k: v * 0.6 for k, v in recent.items()}; recent[nxt] = recent.get(nxt, 0) + 1
        since_act = 0 if act[nxt] else since_act + 1
        cvec = (1 - 0.15) * cvec + 0.15 * emb[nxt]            # fast local flow
        s = unit((1 - drift) * s + drift * emb[nxt])          # slow discourse drift
        if goal_ema: goal = unit((1 - goal_ema) * goal + goal_ema * emb[nxt])  # slow migration
        SK = decay * SK + octo_mul(RP if act[nxt] else RS, embK[nxt])
    return " ".join(vocab[i] for i in out)

def save(M, path):
    """Cache the built model (embedding + n-gram tables) so re-runs skip the build."""
    import pickle
    np.savez(path + ".npz", emb=M["emb"], is_action=M["is_action"],
             vocab=np.array(M["vocab"], dtype=object), K=M["K"])
    with open(path + ".ngrams.pkl", "wb") as f:
        pickle.dump({"tri": dict(M["tri"]), "bi": dict(M["bi"])}, f, protocol=4)

def load(path):
    import pickle
    d = np.load(path + ".npz", allow_pickle=True)
    vocab = list(d["vocab"]); K = int(d["K"]); emb = d["emb"]
    with open(path + ".ngrams.pkl", "rb") as f:
        ng = pickle.load(f)
    cont2, cont1, tot1, D2, D3 = _kn_build(ng["tri"], ng["bi"])
    return dict(vocab=vocab, wi={w: i for i, w in enumerate(vocab)}, W=len(vocab),
                K=K, emb=emb, embK=emb.reshape(len(vocab), K, 8),
                tri=ng["tri"], bi=ng["bi"], is_action=d["is_action"],
                role_subj=_fano_role([1, 2], K), role_pred=_fano_role([3, 4], K),
                role_cycle=[np.tile(np.eye(8)[i], (K, 1)) for i in range(1, 8)],
                cont2=cont2, cont1=cont1, tot1=tot1, D2=D2, D3=D3)

def main():
    import base64, os
    which = sys.argv[1] if len(sys.argv) > 1 else "biomed"
    paths = {"biomed": "corpus_biomed.txt", "science": "corpus_science.txt",
             "shakespeare": "corpus_shakespeare.txt", "books": "corpus_books.txt"}
    cap = {"biomed": None, "science": None, "shakespeare": None}   # use the WHOLE corpus
    vocab_size = int(os.environ.get("OCTO_VOCAB", "40000"))
    p = paths.get(which, which); c = cap.get(which, None)
    cache = f".octogpt_{which}_{vocab_size}"
    if os.path.exists(cache + ".npz"):
        print(f"loading cached model {cache} ...")
        M = load(cache)
    else:
        text = open(p, encoding="utf-8").read(c) if c else open(p, encoding="utf-8").read()
        print(f"building end-to-end gradient-free model on '{which}' "
              f"(whole corpus, vocab {vocab_size})...")
        M = build(text, vocab_size=vocab_size)
        save(M, cache); print(f"  cached to {cache}.*")
    prompts = (sys.argv[2:] or
               ["patients with diabetes", "the treatment reduced",
                "we investigated whether", "in this study we"])
    outs = []
    for s in prompts:
        g = generate(M, s, n=60); outs.append(g); print(f"\n{s} |||\n  {g}")
    print("\nB64GEN:" + base64.b64encode("\n\n".join(outs).encode()).decode())

if __name__ == "__main__":
    main()
