# octonion_pr.py -- PROMPT -> RESPONSE as a gradient-free Fano-path transport.
#
# Builds on octonion_transport.FanoTransport (a composition of exact S^7 Fano rotations
# fit by matching pursuit -- proven to learn object->object functions, held-out align 0.95).
# Here the "objects" are SUB-REPRESENTATIONS of the base octonion network M:
#   prompt  P -> Q_P = o(w1) (x) ... (x) o(wn)   (left-folded fano-path endpoint on S^7)
#   answer  R -> Q_R = same over the answer tokens
# We learn the universal map f: Q_P -> Q_R (the shortest composition of Fano rotations that
# carries prompts to responses), gradient-free, then DECODE by retrieving the real answer
# whose Q_R best matches the predicted anchor f(Q_P'). All octonionic, no backprop.
import json, base64, time, re
import numpy as np
import octonion_gpt as G
from exp_fano_layer import octo_mul, unit
from octonion_transport import FanoTransport, alignment

IDENT = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])

def toks(text, wi):
    return [wi[w] for w in re.findall(r"[a-z']+", text.lower()) if w in wi]

def encode_batch(id_lists, o, Lcap=50):
    """Vectorised left-folded fano-path product Q = o[w1](x)...(x)o[wn] for many spans."""
    N = len(id_lists)
    Q = np.tile(IDENT, (N, 1))
    for p in range(min(max((len(x) for x in id_lists), default=1), Lcap)):
        fac = np.tile(IDENT, (N, 1))
        for i, ids in enumerate(id_lists):
            if p < min(len(ids), Lcap): fac[i] = o[ids[p]]
        Q = octo_mul(Q, fac)
    return Q

def meanemb(id_lists, emb):
    return unit(np.array([emb[t].mean(0) if t else np.zeros(emb.shape[1]) for t in id_lists]))

if __name__ == "__main__":
    t0 = time.time()
    QA = json.load(open("webmdQAs.json"))
    pairs = [(d["question"], d["answer"]) for d in QA
             if d.get("question") and d.get("answer") and 3 <= len(d["question"].split()) <= 60]
    rng = np.random.default_rng(0); perm = rng.permutation(len(pairs)); pairs = [pairs[i] for i in perm]
    train, test = pairs[:16000], pairs[16000:16400]
    corpus = "\n".join(q + " " + a for q, a in train)
    M = G.build(corpus, vocab_size=9000, verbose=False); wi = M["wi"]; emb = M["emb"]; o = unit(emb[:, 1:9])
    tPtr = [toks(q, wi) for q, _ in train]; tAtr = [toks(a, wi) for _, a in train]
    tPte = [toks(q, wi) for q, _ in test];  tAte = [toks(a, wi) for _, a in test]
    QP = encode_batch(tPtr, o); QR = encode_batch(tAtr, o)
    QPt = encode_batch(tPte, o); QRt = encode_batch(tAte, o)
    EP = meanemb(tPtr, emb); EA = meanemb(tAtr, emb); EPt = meanemb(tPte, emb); EAt = meanemb(tAte, emb)
    print("built %.0fs  train=%d test=%d vocab=%d" % (time.time()-t0, len(train), len(test), M["W"]))
    out = []

    # (1) representation sanity: nearest-neighbour train prompts under each rep
    for qi in [0, 7, 19, 33]:
        if len(tPte[qi]) < 3: continue
        nnp = (QP @ QPt[qi]).argsort()[::-1][:3]; nnm = (EP @ EPt[qi]).argsort()[::-1][:3]
        out.append("Q: " + test[qi][0][:85])
        out.append("  fano-NN: " + " || ".join(train[i][0][:48] for i in nnp))
        out.append("  mean-NN: " + " || ".join(train[i][0][:48] for i in nnm)); out.append("")

    # (2) transport generalisation: learn f: Q_P -> Q_R, held-out alignment to true Q_R
    a_id = alignment(QPt, QRt)
    mq = unit(QR.sum(0))[None]
    a_const = float(np.mean(np.sum(unit(np.repeat(mq, len(QRt), 0)) * unit(QRt), axis=1)))
    out.append("TRANSPORT GENERALISATION (held-out alignment of predicted -> true Q_R):")
    out.append("  baseline no-transport align(QPt,QRt) = %.3f" % a_id)
    out.append("  baseline constant-mean               = %.3f" % a_const)
    for steps in [8, 16, 24]:
        f = FanoTransport().fit(QP, QR, steps=steps)
        out.append("  learned f, steps=%2d                  = %.3f" % (steps, alignment(f(QPt), QRt)))
    out.append("")

    # (3) decode as retrieval; relevance = cos(mean-emb of returned answer, gold answer)
    f = FanoTransport().fit(QP, QR, steps=16); QRpred = unit(f(QPt))
    def relevance(ret_idx): return float(np.mean(np.sum(EA[ret_idx] * EAt, axis=1)))
    ret_mean = (EPt @ EP.T).argmax(1)
    ret_fano = (QPt @ QP.T).argmax(1)
    ret_tran = (QRpred @ unit(QR).T).argmax(1)
    rnd = rng.integers(0, len(train), len(test))
    out.append("ANSWER RELEVANCE to gold (cos of mean-emb; higher=better):")
    out.append("  random answer            = %.3f" % relevance(rnd))
    out.append("  B0 mean-emb prompt retr  = %.3f" % relevance(ret_mean))
    out.append("  B1 fano-path prompt retr = %.3f" % relevance(ret_fano))
    out.append("  M2 transport-anchor retr = %.3f" % relevance(ret_tran))
    print("B64PR:" + base64.b64encode("\n".join(out).encode()).decode())
