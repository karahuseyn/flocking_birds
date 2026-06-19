# exp_arc_analogy2.py -- the FAITHFUL ARC test: a relation is a rule ABSTRACTED from many
# examples, not one pair's difference. For each relation class (questions-words section) we
# estimate the transform from TRAIN instance-pairs and apply it to held-out TEST pairs:
#   octo (Fano-path): R_class_i = renorm(mean_train b_i (x) a_i^{-1});  pred y_i = R_class_i (x) x_i
#   vec  (linear)   : off = mean_train (b - a);                          pred y   = x + off
# This is closer to ARC (abstract the rule, then apply) and a fair octo-vs-linear comparison.
import base64, time
import numpy as np
import exp_fano_layer as X
from exp_arc_analogy import per_block_unit, load_analogies

if __name__ == "__main__":
    t0 = time.time()
    TEXT = open("corpus_books.txt", encoding="utf-8", errors="ignore").read()
    M = X.build(TEXT, vocab_size=40000); wi = M["wi"]; emb = M["emb"]; W = M["W"]; K = M["K"]
    EB = per_block_unit(M); EBf = EB.reshape(W, K*8); EBf = EBf/(np.linalg.norm(EBf,axis=1,keepdims=True)+1e-12)
    embU = emb/(np.linalg.norm(emb,axis=1,keepdims=True)+1e-12)
    secs = load_analogies("questions-words.txt", wi)
    rng = np.random.default_rng(0)
    print("built %.0fs" % (time.time()-t0))

    def instance_pairs(quads):                          # each quad gives two instances of the relation
        P = []
        for a, b, c, d in quads: P += [(a, b), (c, d)]
        return P

    Vt1 = Vt5 = Ot1 = Ot5 = N = 0; rows = []
    for name, quads in secs.items():
        if len(quads) < 6: continue
        P = instance_pairs(quads); rng.shuffle(P)
        cut = len(P)//2; tr, te = P[:cut], P[cut:]
        # abstract the rule from TRAIN
        off = np.mean([emb[b]-emb[a] for a, b in tr], 0)                       # linear rule
        Rcl = np.mean([X.octo_mul(EB[b], X._inv(EB[a])) for a, b in tr], 0)    # Fano-path rule (K,8)
        Rcl = Rcl/(np.linalg.norm(Rcl,axis=-1,keepdims=True)+1e-12)
        v1=v5=o1=o5=0
        for x, y in te:
            vv = embU[x]*0 + emb[x] + off; vv = vv/(np.linalg.norm(vv)+1e-12)
            vo = [i for i in (embU @ vv).argsort()[::-1] if i != x][:5]
            yo = X.octo_mul(Rcl, EB[x]).reshape(-1); yo = yo/(np.linalg.norm(yo)+1e-12)
            oo = [i for i in (EBf @ yo).argsort()[::-1] if i != x][:5]
            v1 += y==vo[0]; v5 += y in vo; o1 += y==oo[0]; o5 += y in oo
        n=len(te); N+=n; Vt1+=v1;Vt5+=v5;Ot1+=o1;Ot5+=o5
        rows.append("  %-26s n=%-4d vec@1=%.2f octo@1=%.2f | vec@5=%.2f octo@5=%.2f"%(name,n,v1/n,o1/n,v5/n,o5/n))
    out=["CLASS-ABSTRACTED  n=%d  vec@1=%.3f octo@1=%.3f | vec@5=%.3f octo@5=%.3f"%(N,Vt1/N,Ot1/N,Vt5/N,Ot5/N)]+rows
    print("B64ARC2:"+base64.b64encode("\n".join(out).encode()).decode())
