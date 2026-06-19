# exp_arc_analogy.py -- ARC philosophy probe: relation = octonion TRANSFORM (Fano path),
# not vector difference. Token = a SET of K octonions (embK is (W,K,8)); the relation a->b
# is the per-block exact rotation R_i = b_i (x) a_i^{-1} (the same IMPLIES operator that makes
# multi-hop logic exact). Analogy a:b::c:d is then d_i = R_i (x) c_i, vs the additive b-a+c
# that FAILED earlier. Tested on questions-words.txt. base64-verified.
import base64, re, time
import numpy as np
import exp_fano_layer as X

def per_block_unit(M):
    """emb as a set of K unit octonions per token: (W,K,8), each block unit-norm."""
    e = M["emb"].reshape(M["W"], M["K"], 8)
    return e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-12)

def load_analogies(path, wi):
    secs = {}; cur = None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith(":"): cur = line[2:]; secs[cur] = []
        else:
            p = line.lower().split()
            if len(p) == 4 and all(w in wi for w in p): secs[cur].append([wi[w] for w in p])
    return secs

if __name__ == "__main__":
    t0 = time.time()
    TEXT = open("corpus_books.txt", encoding="utf-8", errors="ignore").read()
    M = X.build(TEXT, vocab_size=40000); wi = M["wi"]; emb = M["emb"]; W = M["W"]; K = M["K"]
    EB = per_block_unit(M)                              # (W,K,8) set-of-octonions catalog
    EBf = EB.reshape(W, K*8)                            # flat, for retrieval in the same space
    EBf = EBf / (np.linalg.norm(EBf, axis=1, keepdims=True) + 1e-12)
    secs = load_analogies("questions-words.txt", wi)
    print("built %.0fs  covered analogies: %d in %d sections" %
          (time.time()-t0, sum(len(v) for v in secs.values()), len(secs)))

    def vec_pred(a, b, c, topn=5):
        v = emb[b] - emb[a] + emb[c]; v = v / (np.linalg.norm(v)+1e-12)
        order = (emb @ v).argsort()[::-1]
        return [i for i in order if i not in (a, b, c)][:topn]

    def octo_pred(a, b, c, topn=5):                     # relation as Fano-path transform
        ai, bi, ci = EB[a], EB[b], EB[c]                # (K,8) sets
        Ri = X.octo_mul(bi, X._inv(ai))                # per-block rotation b (x) a^-1
        di = X.octo_mul(Ri, ci)                         # apply to c
        d = di.reshape(-1); d = d / (np.linalg.norm(d)+1e-12)
        order = (EBf @ d).argsort()[::-1]
        return [i for i in order if i not in (a, b, c)][:topn]

    out = []
    Vt1 = Vt5 = Ot1 = Ot5 = N = 0
    bysec = []
    for name, quads in secs.items():
        if not quads: continue
        v1 = v5 = o1 = o5 = 0
        for a, b, c, d in quads:
            vp = vec_pred(a, b, c); op = octo_pred(a, b, c)
            v1 += d == vp[0]; v5 += d in vp; o1 += d == op[0]; o5 += d in op
        n = len(quads); N += n; Vt1 += v1; Vt5 += v5; Ot1 += o1; Ot5 += o5
        bysec.append("  %-26s n=%-4d vec@1=%.2f octo@1=%.2f | vec@5=%.2f octo@5=%.2f"
                     % (name, n, v1/n, o1/n, v5/n, o5/n))
    out.append("TOTAL n=%d  vec@1=%.3f octo@1=%.3f  |  vec@5=%.3f octo@5=%.3f"
               % (N, Vt1/N, Ot1/N, Vt5/N, Ot5/N))
    out += bysec
    print("B64ARC:" + base64.b64encode("\n".join(out).encode()).decode())
