# exp_fano_path.py -- the CORRECT "fano path": positional roles are RUNNING PRODUCTS along a
# cyclic sequence of Fano-unit multiplications (interpretation C), not single units per slot.
#   R_0 = 1 ;  R_k = R_{k-1} * e_{g_{k-1}}  for a cyclic generator path [g_0..g_{L-1}].
# Compared against the shipped single-unit "point cycle" echo, same everything else.
import base64, re, numpy as np
import exp_fano_layer as X

def path_roles(path, K):
    """Running-product roles R_0..R_{len(path)-1} along a cyclic Fano path."""
    roles = []; acc = np.zeros(8); acc[0] = 1.0
    for g in path:                                  # R_k = product of first k generators
        roles.append(np.tile(acc, (K, 1)))
        e = np.zeros(8); e[g] = 1.0; acc = X.octo_mul(acc, e)
    return roles                                     # length == len(path)

def gen(M, seed, roles, n=80, temp=0.5, rng_seed=1, w_goal=3.0, w_fano=4.0):
    emb, embK, act = M["emb"], M["embK"], M["act"]; RS, RP = M["RS"], M["RP"]
    tri, bi, W, vocab, K = M["tri"], M["bi"], M["W"], M["vocab"], M["K"]
    rng = np.random.default_rng(rng_seed)
    out = [M["wi"][w] for w in seed.split() if w in M["wi"]] or [int(rng.integers(W))]
    s = X.unit(emb[out].mean(0)); cvec = s.copy(); goal = s.copy(); SK = np.zeros((K, 8))
    recent = {}; seen = set(); since = 0; L = len(roles)
    for x in out: SK = 0.8*SK + X.octo_mul(RP if act[x] else RS, embK[x])
    for _ in range(n):
        c = tri.get((out[-2], out[-1])) if len(out) >= 2 else None
        cb = bi.get(out[-1]); cset = (set(c) if c else set()) | (set(cb) if cb else set())
        if cset:
            cand = np.array(sorted(cset))
            if len(cand) > 1:
                k = np.array([(out[-1], int(x)) not in seen for x in cand])
                if k.any(): cand = cand[k]
            nz = lambda x: (x-x.min())/(np.ptp(x)+1e-9) if len(cand) > 1 else x*0
            es = X.unit(X.octo_mul(X._inv(RS), SK).reshape(-1)); ep = X.unit(X.octo_mul(X._inv(RP), SK).reshape(-1))
            al = np.array([(1.0 if since >= 2 else -0.5) if act[x] else 0.0 for x in cand])
            rp = np.array([recent.get(int(x), 0) for x in cand], float)
            fl = out[-L:]; cen = X.unit(emb[fl].mean(0)); mot = X.unit(emb[fl[-1]]-emb[fl[0]]) if len(fl) > 1 else cen
            bb = np.array([np.log((c.get(int(x)) if c else None) or (cb.get(int(x)) if cb else 1)) for x in cand])
            # FANO PATH echo: holographic register with running-product roles
            base = len(out)-len(fl); H = np.zeros((K, 8))
            for kk, tok in enumerate(fl): H = H + X.octo_mul(roles[(base+kk) % L], embK[tok])
            pred = X.unit(X.octo_mul(X._inv(roles[len(out) % L]), H).reshape(-1))
            fa = nz(emb[cand] @ pred)
            sc = (bb + nz(emb[cand]@cvec) + 1.5*nz(emb[cand]@s) + 1.5*nz(emb[cand]@es) + 3.5*nz(emb[cand]@ep)
                  + 2.0*al + 2.0*nz(emb[cand]@cen) + 1.0*nz(emb[cand]@mot) + w_goal*nz(emb[cand]@goal)
                  + w_fano*fa - 2.0*rp)
            p = np.exp(sc/temp); p /= p.sum(); nxt = int(rng.choice(cand, p=p))
        else:
            nxt = int(rng.integers(W))
        seen.add((out[-1], nxt)); out.append(nxt); recent = {k: v*0.6 for k, v in recent.items()}; recent[nxt] = recent.get(nxt, 0)+1
        since = 0 if act[nxt] else since+1; cvec = 0.85*cvec+0.15*emb[nxt]; s = X.unit(0.82*s+0.18*emb[nxt])
        SK = 0.8*SK + X.octo_mul(RP if act[nxt] else RS, embK[nxt])
    return [vocab[i] for i in out]

if __name__ == "__main__":
    TEXT = open("corpus_books.txt", encoding="utf-8", errors="ignore").read()[:6_000_000]
    M = X.build(TEXT, vocab_size=8000); K = M["K"]
    unit_roles = [np.tile(np.eye(8)[i], (K, 1)) for i in range(1, 8)]        # shipped: single units
    singer_path = path_roles([1, 2, 3, 4, 5, 6, 7], K)                       # running product, Singer order
    line_path  = path_roles([1, 2, 4, 1, 2, 4, 1], K)                        # running product along a Fano LINE {1,2,4}
    seeds = ["the king", "she looked at the", "in the morning", "the meaning of",
             "the old man", "they walked through the", "the war had"]
    def run(name, roles, w_fano=4.0):
        a = b = an = r = 0.0; samp = ""
        for si, sd in enumerate(seeds):
            t = gen(M, sd, roles, rng_seed=si+1, w_fano=w_fano); idl = [M["wi"][w] for w in t]
            lc, se, anc, rep = X.metrics(M, np.array(idl)); a += lc; b += se; an += anc; r += rep
            if si == 1: samp = " ".join(t)
        k = len(seeds); return "%s coh=%.3f drift=%.3f anchor=%.3f rep=%.3f\n   %s" % (name, a/k, b/k, an/k, r/k, samp[:190])
    L = [run("no fano (w_fano=0)      ", unit_roles, w_fano=0.0),
         run("single-unit (shipped)   ", unit_roles),
         run("PATH running-prod Singer", singer_path),
         run("PATH running-prod line  ", line_path)]
    print("B64FP:" + base64.b64encode("\n".join(L).encode()).decode())
