# exp_token_sets.py -- STEP 1: a token is a VARIABLE-SIZE SET of octonions (ARC-object style,
# no fixed K=12). The set = the token's SENSES, found gradient-free by clustering its actual
# usage contexts. Polysemous words get bigger sets; monosemous words get one. Each sense
# centroid -> a unit octonion on the globally-consistent top-8 PMI-SVD axes (ready for step 2:
# fano paths between sets). base64-verified: shows sense words + set sizes.
import base64, re, time
import numpy as np
import exp_fano_layer as X

def kmeans(Xd, k, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    C = Xd[rng.choice(len(Xd), k, replace=False)].copy()
    for _ in range(iters):
        d = ((Xd[:, None, :] - C[None]) ** 2).sum(-1); lab = d.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any(): C[j] = Xd[m].mean(0)
    return lab, C

def sense_sets(M, ids, target_ids, window=4, max_occ=400, kmax=4):
    """For each target token: cluster its windowed-context vectors into senses (variable count)."""
    emb = M["emb"]; pos = {}
    for i, w in enumerate(ids):
        if w in target_ids: pos.setdefault(w, []).append(i)
    res = {}
    for w, P in pos.items():
        P = P[:max_occ]
        ctx = []
        for i in P:
            lo, hi = max(0, i-window), min(len(ids), i+window+1)
            nb = [ids[j] for j in range(lo, hi) if j != i]
            if nb: ctx.append(emb[nb].mean(0))
        if len(ctx) < 8: res[w] = None; continue
        ctx = X.unit(np.array(ctx))
        # choose number of senses: grow k while clusters stay well-separated & supported
        best = (1, *kmeans(ctx, 1))
        for k in range(2, kmax+1):
            if len(ctx) < k*15: break
            lab, C = kmeans(ctx, k, seed=1)
            sizes = [np.mean(lab == j) for j in range(k)]
            Cn = X.unit(C); sep = max((Cn @ Cn.T)[np.triu_indices(k, 1)])  # worst pair similarity
            if min(sizes) >= 0.15 and sep < 0.55: best = (k, lab, C)   # accept this finer split
        res[w] = best
    return res

def to_octonion(centroid8):
    o = centroid8.copy(); return o / (np.linalg.norm(o) + 1e-12)   # top-8 global axes -> unit octonion

if __name__ == "__main__":
    t0 = time.time()
    TEXT = open("corpus_books.txt", encoding="utf-8", errors="ignore").read()
    M = X.build(TEXT, vocab_size=30000); wi = M["wi"]; emb = M["emb"]; vocab = M["vocab"]
    words = re.findall(r"[a-z']+", TEXT.lower())
    ids = [wi[w] for w in words if w in wi]
    print("built %.0fs, %d tokens" % (time.time()-t0, len(ids)))
    targets = ["bank", "spring", "light", "bear", "rock", "fair", "ground", "court",
               "therefore", "hydrogen", "nevertheless", "elizabeth"]
    tids = {wi[w] for w in targets if w in wi}
    R = sense_sets(M, ids, tids)
    def near(v, k=6):
        return [vocab[i] for i in (emb @ X.unit(v)).argsort()[::-1] if vocab[i] not in targets][:k]
    out = []
    for w in targets:
        if w not in wi or R.get(wi[w]) is None: out.append("%-12s (too rare)" % w); continue
        k, lab, C = R[wi[w]]
        out.append("%-12s  SET SIZE = %d" % (w, k))
        for j in range(k):
            oct8 = to_octonion(C[j][:8])
            out.append("    sense %d  (%.0f%%):  %s   |oct[:3]=%s" %
                       (j+1, 100*np.mean(lab == j), ", ".join(near(C[j])),
                        np.array2string(oct8[:3], precision=2)))
    print("B64TS:" + base64.b64encode("\n".join(out).encode()).decode())
