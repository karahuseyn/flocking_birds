# exp_relpath.py -- STEP 2 generation: RELATION CONSISTENCY along the fano path.
# Each token -> one octonion o(w)=unit(emb[w][1:9]). The relation cur->next is the transform
# R with o(next)=R (x) o(cur). We score the next token by how well its octonion matches the
# EXPECTED relation, two ways:
#   momentum: keep the current relation r=o(cur)(x)o(prev)^-1 -> expect r(x)o(cur)
#   corpus  : R_a = mean corpus relation after token a -> expect R_a(x)o(cur)
# Added as a w_rel term on top of the shipped KN + goal + flock scorer. base64-verified.
import base64, time
import numpy as np
import octonion_gpt as G, exp_fano_layer as X

def octo_table(M):
    o = M["emb"][:, 1:9].copy(); return o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)

def corpus_relations(M, o):
    """R_a = unit( sum_b count(a,b) * (o_b (x) o_a^-1) ) : typical relation after token a."""
    W = M["W"]; R = np.tile(np.array([1.0,0,0,0,0,0,0,0]), (W, 1))
    for a, ctr in M["bi"].items():
        inva = X._inv(o[a]); acc = np.zeros(8)
        for b, c in ctr.items(): acc += c * X.octo_mul(o[b], inva)
        n = np.linalg.norm(acc)
        if n > 1e-9: R[a] = acc / n
    return R

def gen(M, o, Rtab, seed, n=80, temp=0.5, rng_seed=1, w_goal=3.0, w_rel=0.0, mode="momentum"):
    vocab, wi, W, emb = M["vocab"], M["wi"], M["W"], M["emb"]
    tri, bi, act = M["tri"], M["bi"], M["is_action"]
    RS, RP = M["role_subj"], M["role_pred"]; rng = np.random.default_rng(rng_seed)
    out = [wi[w] for w in seed.lower().split() if w in wi] or [int(rng.integers(W))]
    s = X.unit(emb[out].mean(0)); cvec = s.copy(); goal = s.copy(); SK = np.zeros((M["K"], 8))
    for x in out: SK = 0.8*SK + X.octo_mul(RP if act[x] else RS, M["embK"][x])
    recent = {}; since = 0; seen = set()
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
            gl = nz(emb[cand] @ goal)
            # RELATION CONSISTENCY term
            rel = 0.0
            if w_rel:
                cur = out[-1]
                if mode == "momentum" and len(out) >= 2:
                    r = X.octo_mul(o[cur], X._inv(o[out[-2]]))
                else:
                    r = Rtab[cur]
                exp = X.octo_mul(r, o[cur]); exp = exp/(np.linalg.norm(exp)+1e-12)
                rel = nz(o[cand] @ exp)
            sc = bb + nz(emb[cand]@cvec) + 1.5*nz(emb[cand]@s) + 2.0*nz(emb[cand]@cen) \
                 + w_goal*gl + w_rel*rel
            p = np.exp(sc/temp); p /= p.sum(); nxt = int(rng.choice(cand, p=p))
        else:
            nxt = int(rng.integers(W))
        seen.add((out[-1], nxt)); out.append(nxt)
        recent = {k: v*0.6 for k, v in recent.items()}; recent[nxt] = recent.get(nxt, 0)+1
        since = 0 if act[nxt] else since+1
        cvec = 0.85*cvec + 0.15*emb[nxt]; s = X.unit(0.82*s + 0.18*emb[nxt])
        SK = 0.8*SK + X.octo_mul(RP if act[nxt] else RS, M["embK"][nxt])
    return [vocab[i] for i in out]

if __name__ == "__main__":
    t0 = time.time()
    TEXT = open("corpus_books.txt", encoding="utf-8", errors="ignore").read()[:6_000_000]
    M = G.build(TEXT, vocab_size=8000, verbose=False); o = octo_table(M); Rtab = corpus_relations(M, o)
    print("built %.0fs" % (time.time()-t0))
    seeds = ["the king", "she looked at the", "in the morning", "the meaning of",
             "the old man", "they walked through the", "the war had"]
    def run(name, **cfg):
        a=b=an=r=0.0; samp=""
        for si, sd in enumerate(seeds):
            t = gen(M, o, Rtab, sd, rng_seed=si+1, **cfg); idl = np.array([M["wi"][w] for w in t])
            lc, se, anc, rep = X.metrics(M, idl); a+=lc; b+=se; an+=anc; r+=rep
            if si == 1: samp = " ".join(t)
        k = len(seeds)
        return "%s coh=%.3f drift=%.3f anchor=%.3f rep=%.3f\n   %s" % (name, a/k, b/k, an/k, r/k, samp[:180])
    L = [run("baseline (w_rel=0)    ", w_rel=0.0),
         run("relation momentum 3.0 ", w_rel=3.0, mode="momentum"),
         run("relation corpus 3.0   ", w_rel=3.0, mode="corpus"),
         run("relation momentum 5.0 ", w_rel=5.0, mode="momentum")]
    print("B64REL:" + base64.b64encode("\n".join(L).encode()).decode())
