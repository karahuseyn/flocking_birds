# octonion_pr_scale_ext.py -- (b) scale the candidate pool: does more data close the oracle
# gap for the EXTRACTIVE decoder? Relaxes filters to grow the Q&A pool, then measures the
# best selection config (anchor+centrality, m=3, lam=0.7) and the oracle at growing train
# sizes, fixed test. If ext_tran climbs toward oracle, pool quality was the bottleneck.
import json, base64, time
import numpy as np
import octonion_gpt as G
from exp_fano_layer import unit
from octonion_pr_slot import toks, meanemb, slots, fit_slot_transport, apply_slot
from octonion_pr_extract import SOURCES, sents, rouge1

def mmr_score(cv, score, m=3, lam=0.7):
    chosen = []
    while len(chosen) < m and len(chosen) < len(cv):
        best, bv = -1, -1e9
        for i in range(len(cv)):
            if i in chosen: continue
            red = max((cv[i] @ cv[j] for j in chosen), default=0.0)
            v = lam * score[i] - (1 - lam) * red
            if v > bv: bv, best = v, i
        chosen.append(best)
    return sorted(chosen, key=lambda i: -score[i])

if __name__ == "__main__":
    t0 = time.time()
    pairs = []
    for f in SOURCES:
        for d in json.load(open(f)):
            q, a = d.get("question"), d.get("answer")
            if q and a and 3 <= len(q.split()) <= 50 and 5 <= len(a.split()) <= 150:
                pairs.append((q, a))
    rng = np.random.default_rng(0); pairs = [pairs[i] for i in rng.permutation(len(pairs))]
    test = pairs[:200]; pool_all = pairs[200:]
    print("relaxed pool=%d test=200 trainable=%d" % (len(pairs), len(pool_all)))

    def evaluate(ntr):
        train = pool_all[:ntr]
        M = G.build("\n".join(q + " " + a for q, a in train), vocab_size=min(11000, 2000 + ntr // 4), verbose=False)
        wi, emb = M["wi"], M["emb"]
        EPr = meanemb([toks(q, wi) for q, _ in train], emb); EAr = meanemb([toks(a, wi) for _, a in train], emb)
        Xtr, Ttr, EPru = slots(EPr), slots(EAr), unit(EPr)
        semb = lambda txt: unit(emb[toks(txt, wi)].mean(0)) if toks(txt, wi) else np.zeros(96)
        ee = er = oe = orr = 0.0; nn_cnt = 0
        for q, gold in test:
            tq = toks(q, wi)
            if not tq: continue
            pe = unit(emb[tq].mean(0)); nn = np.argsort(-(EPru @ pe))[:40]
            F = fit_slot_transport(Xtr[nn], Ttr[nn], steps=10)
            g = unit(apply_slot(F, slots(emb[tq].mean(0)[None]))[0].reshape(96))
            poolS = list(dict.fromkeys([s for j in nn for s in sents(train[int(j)][1])]))[:140]
            if not poolS: continue
            cv = np.array([semb(s) for s in poolS]); cen = cv.mean(0); ge = semb(gold)
            idx = mmr_score(cv, 0.7 * (cv @ g) + 0.3 * (cv @ cen)); ans = " ".join(poolS[i] for i in idx)
            oidx = mmr_score(cv, cv @ ge); oans = " ".join(poolS[i] for i in oidx)
            ee += float(np.sum(semb(ans) * ge)); er += rouge1(ans, gold)
            oe += float(np.sum(semb(oans) * ge)); orr += rouge1(oans, gold); nn_cnt += 1
        return ee/nn_cnt, er/nn_cnt, oe/nn_cnt, orr/nn_cnt, M["W"]

    out = ["EXTRACTIVE SCALING (fixed 200 test, best config):",
           "  ntrain vocab | ext_tran emb/ROUGE | oracle emb/ROUGE"]
    sizes = [4000, 16000, min(40000, len(pool_all)), len(pool_all)]
    for ntr in sorted(set(sizes)):
        e, r, oe, orr, W = evaluate(ntr)
        out.append("  %6d %5d | %.3f / %.3f      | %.3f / %.3f" % (ntr, W, e, r, oe, orr))
    print("built %.0fs" % (time.time()-t0))
    print("B64SX:" + base64.b64encode("\n".join(out).encode()).decode())
