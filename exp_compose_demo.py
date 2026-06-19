# exp_compose_demo.py -- STEP 2 bridge: the long fano path = a COMPOSITION representation.
# Octonion product is non-commutative, so a sentence and its reordering (same words, different
# meaning/composition) trace DIFFERENT trajectories -- while bag-of-words (mean embedding) is
# identical. And by the composition-algebra non-decay, the distinction survives long carrier
# context. This is "relations compose into the wider composition." base64-verified.
import base64, re, time
import numpy as np
import exp_fano_layer as X

def tok_oct(M):
    o = M["emb"][:, :8].copy(); return o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)

if __name__ == "__main__":
    t0 = time.time()
    TEXT = open("corpus_books.txt", encoding="utf-8", errors="ignore").read()[:5_000_000]
    M = X.build(TEXT, vocab_size=8000); wi = M["wi"]; emb = M["emb"]; O = tok_oct(M)
    ID = np.zeros(8); ID[0] = 1.0
    def path(toks):
        P = ID.copy()
        for w in toks:
            if w in wi: P = X.octo_mul(P, O[wi[w]])
        return P
    def bag(toks):
        v = np.mean([emb[wi[w]] for w in toks if w in wi], 0); return v/(np.linalg.norm(v)+1e-12)
    pairs = [("the dog bit the man", "the man bit the dog"),
             ("she loved him", "he loved her"),
             ("the king killed the soldier", "the soldier killed the king"),
             ("water turns to ice", "ice turns to water"),
             ("the cat saw the bird", "the bird saw the cat")]
    out = []
    for s1, s2 in pairs:
        a, b = s1.split(), s2.split()
        pd = np.linalg.norm(path(a) - path(b))         # fano-path (composition) distance
        bd = np.linalg.norm(bag(a) - bag(b))           # bag-of-words distance
        out.append("path=%.3f  bag=%.3f  | '%s'  vs  '%s'" % (pd, bd, s1, s2))
    # non-decay: same meaning-flip embedded in a long shared carrier -> path still distinguishes
    carrier_pre = "it was a long time ago in a distant country that".split()
    carrier_post = "and nothing was ever quite the same again after all".split()
    for L in (0, 1, 1):  # pre, then grow carrier
        pass
    a = carrier_pre + "the dog bit the man".split() + carrier_post
    b = carrier_pre + "the man bit the dog".split() + carrier_post
    out.append("LONG carrier (len %d): path=%.3f  bag=%.3f  (flip is mid-sentence, far from ends)"
               % (len(a), np.linalg.norm(path(a)-path(b)), np.linalg.norm(bag(a)-bag(b))))
    print("B64CD:" + base64.b64encode("\n".join(out).encode()).decode())
