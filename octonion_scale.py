# octonion_scale.py -- does the prompt->response transport engine benefit from SCALE?
# Pools all 5 medical Q&A sources, holds out a fixed test, and measures the transport
# alignment + answer relevance as a function of TRAIN SIZE. If the curves still climb,
# more data helps the (gradient-free) encode+transport engine; if flat, it's saturated.
import json, base64, time
import numpy as np
import octonion_gpt as G
from exp_fano_layer import unit
from octonion_pr_slot import toks, meanemb, slots, align_slots, fit_slot_transport, apply_slot

SOURCES = ["medquadQAs.json", "webmdQAs.json", "icliniqQAs.json",
           "questionDoctorQAs.json", "ehealthforumQAs.json"]

if __name__ == "__main__":
    t0 = time.time()
    pairs = []
    for f in SOURCES:
        for d in json.load(open(f)):
            q, a = d.get("question"), d.get("answer")
            if q and a and 3 <= len(q.split()) <= 40 and 5 <= len(a.split()) <= 80:
                pairs.append((q, a))
    rng = np.random.default_rng(0); pairs = [pairs[i] for i in rng.permutation(len(pairs))]
    test = pairs[:200]; pool = pairs[200:]
    print("pooled pairs=%d  test=200  pool=%d" % (len(pairs), len(pool)))

    def evaluate(ntr):
        train = pool[:ntr]
        M = G.build("\n".join(q + " " + a for q, a in train), vocab_size=min(12000, 1500 + ntr // 4), verbose=False)
        wi, emb = M["wi"], M["emb"]
        EPr = meanemb([toks(q, wi) for q, _ in train], emb); EAr = meanemb([toks(a, wi) for _, a in train], emb)
        EPe = meanemb([toks(q, wi) for q, _ in test], emb);  EAe = meanemb([toks(a, wi) for _, a in test], emb)
        Xtr, Ttr, Xte, Tte = slots(EPr), slots(EAr), slots(EPe), slots(EAe)
        EPru, EAru, EPeu, EAeu = unit(EPr), unit(EAr), unit(EPe), unit(EAe)
        # local kNN transport on the test prompts
        sims = EPeu @ EPru.T; nn = np.argsort(-sims, axis=1)[:, :60]; pred = Xte.copy()
        for i in range(len(test)):
            F = fit_slot_transport(Xtr[nn[i]], Ttr[nn[i]], steps=10); pred[i] = apply_slot(F, Xte[i:i+1])[0]
        a_none = align_slots(Xte, Tte); a_loc = align_slots(pred, Tte)
        rel = lambda idx: float(np.mean(np.sum(EAru[idx] * EAeu, axis=1)))
        r_ret = rel((EPeu @ EPru.T).argmax(1))
        r_tr = rel((unit(pred.reshape(len(test), 96)) @ EAru.T).argmax(1))
        return a_none, a_loc, r_ret, r_tr, M["W"]

    out = ["TRAIN-SIZE SCALING (fixed 200-pair test):",
           "  ntrain  vocab | align:none -> local | relevance:retr  transport"]
    for ntr in [2000, 8000, 24000, len(pool)]:
        an, al, rr, rt, W = evaluate(ntr)
        out.append("  %6d  %5d | %.3f -> %.3f      | %.3f         %.3f" % (ntr, W, an, al, rr, rt))
    print("built %.0fs" % (time.time()-t0))
    print("B64SCALE:" + base64.b64encode("\n".join(out).encode()).decode())
