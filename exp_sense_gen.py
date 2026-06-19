# exp_sense_gen.py -- STEP 1 (x) STEP 2: generation driven by SENSE-DISAMBIGUATED octonion sets.
# Each token has a variable-size set of sense-octonions (clustered from its content contexts).
# During generation a token enters the fano path with the sense whose context-centroid best
# matches the current flock; relation-consistency then operates between the CORRECT senses.
# Compares: no relation / single-octonion relation / sense-disambiguated relation. base64.
import base64, re, time
import numpy as np
from collections import Counter
import octonion_gpt as G, exp_fano_layer as X
from exp_token_sets import kmeans

def build_senses(M, ids, freq, stop, idf, lo=40, hi=40000, cap=120, window=5):
    emb = M["emb"]; W = M["W"]; tset = set(t for t in range(W) if lo <= freq.get(t, 0) <= hi and t not in stop)
    ctxs = {t: [] for t in tset}
    for i, t in enumerate(ids):
        if t in tset and len(ctxs[t]) < cap:
            l, h = max(0, i-window), min(len(ids), i+window+1); v = np.zeros(emb.shape[1]); tot = 0.0
            for j in range(l, h):
                u = ids[j]
                if j == i or u in stop: continue
                v += idf[u]*emb[u]; tot += idf[u]
            if tot > 0: ctxs[t].append(v/tot)
    SCTX = {}; SOCT = {}
    for t, C in ctxs.items():
        if len(C) < 14: continue
        Cn = X.unit(np.array(C)); best = (1, *kmeans(Cn, 1))
        for k in (2, 3):
            if len(Cn) < k*15: break
            lab, cc = kmeans(Cn, k, seed=1); sizes = [np.mean(lab == j) for j in range(k)]
            ccu = X.unit(cc); sep = max((ccu @ ccu.T)[np.triu_indices(k, 1)])
            if min(sizes) >= 0.2 and sep < 0.82: best = (k, lab, cc)
        k, lab, cc = best
        SCTX[t] = X.unit(cc); o = cc[:, 1:9]; SOCT[t] = o/(np.linalg.norm(o, axis=1, keepdims=True)+1e-12)
    return SCTX, SOCT

if __name__ == "__main__":
    t0 = time.time()
    TEXT = open("corpus_books.txt", encoding="utf-8", errors="ignore").read()[:8_000_000]
    M = G.build(TEXT, vocab_size=7000, verbose=False); emb = M["emb"]; wi = M["wi"]; W = M["W"]; vocab = M["vocab"]
    words = re.findall(r"[a-z']+", TEXT.lower()); ids = [wi[w] for w in words if w in wi]
    freq = Counter(ids); N = sum(freq.values())
    stop = set(i for i, _ in freq.most_common(120)); idf = {u: np.log(1+N/c) for u, c in freq.items()}
    SCTX, SOCT = build_senses(M, ids, freq, stop, idf)
    o1 = X.unit(emb[:, 1:9]); embu = X.unit(emb)
    multi = sum(1 for t in SOCT if len(SOCT[t]) > 1)
    print("built %.0fs  sense-sets for %d tokens (%d polysemous)" % (time.time()-t0, len(SOCT), multi))

    def disamb(t, c):
        if t in SOCT: O = SOCT[t]; return O[int((SCTX[t] @ c).argmax())] if len(O) > 1 else O[0]
        return o1[t]

    def gen(seed, n=80, temp=0.5, rng_seed=1, mode="sense", w_rel=3.0, w_goal=3.0):
        tri, bi = M["tri"], M["bi"]; rng = np.random.default_rng(rng_seed)
        out = [wi[w] for w in seed.lower().split() if w in wi] or [int(rng.integers(W))]
        s = X.unit(emb[out].mean(0)); cvec = s.copy(); goal = s.copy(); seen = set()
        sel = [disamb(x, s) for x in out]
        for _ in range(n):
            a = out[-2] if len(out) >= 2 else -1
            ct = tri.get((a, out[-1])); cb = bi.get(out[-1])
            cset = (set(ct) if ct else set()) | (set(cb) if cb else set())
            if cset:
                cand = np.array(sorted(cset))
                if len(cand) > 1:
                    k = np.array([(out[-1], int(x)) not in seen for x in cand])
                    if k.any(): cand = cand[k]
                nz = lambda x: (x-x.min())/(np.ptp(x)+1e-9) if len(cand) > 1 else x*0
                bb = G._kn3_logprob(M, a, out[-1], cand)
                fl = out[-7:]; cen = X.unit(emb[fl].mean(0))
                rl = 0.0
                if w_rel and len(out) >= 2:
                    if mode == "single":
                        ocur, oprev, ocand = o1[out[-1]], o1[out[-2]], o1[cand]
                    else:
                        ocur, oprev = sel[-1], sel[-2]
                        ocand = np.array([disamb(int(w), cen) for w in cand])
                    r = X.octo_mul(ocur, X._inv(oprev)); exp = X.octo_mul(r, ocur)
                    exp = exp/(np.linalg.norm(exp)+1e-12); rl = nz(ocand @ exp)
                sc = bb + nz(emb[cand]@cvec) + 1.5*nz(emb[cand]@s) + 2.0*nz(emb[cand]@cen) + w_goal*nz(emb[cand]@goal) + w_rel*rl
                p = np.exp(sc/temp); p /= p.sum(); nxt = int(rng.choice(cand, p=p))
            else:
                nxt = int(rng.integers(W))
            seen.add((out[-1], nxt)); out.append(nxt); cvec = 0.85*cvec+0.15*emb[nxt]
            s = X.unit(0.82*s+0.18*emb[nxt]); sel.append(disamb(nxt, cen if cset else s))
        return [vocab[i] for i in out]

    seeds = ["the king", "she looked at the", "in the morning", "the meaning of",
             "the old man", "they walked through the", "the war had"]
    def run(name, **cfg):
        a=b=an=r=0.0; samp=""
        for si, sd in enumerate(seeds):
            t = gen(sd, rng_seed=si+1, **cfg); idl = np.array([wi[w] for w in t])
            lc, se, anc, rep = X.metrics(M, idl); a+=lc; b+=se; an+=anc; r+=rep
            if si == 1: samp = " ".join(t)
        k = len(seeds); return "%s coh=%.3f drift=%.3f anchor=%.3f rep=%.3f\n   %s" % (name, a/k, b/k, an/k, r/k, samp[:175])
    L = [run("no relation        ", w_rel=0.0),
         run("single-octonion rel", w_rel=3.0, mode="single"),
         run("SENSE-disambig rel ", w_rel=3.0, mode="sense")]
    print("B64SG:" + base64.b64encode("\n".join(L).encode()).decode())
