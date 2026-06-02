# octonion_pr_extract2.py -- (a) close the oracle gap: enrich the sentence-selection score.
# Caches pool/anchor/embeddings once per test prompt, then sweeps scoring configs cheaply:
#   score(s) = a*sim(s, transport-anchor) + b*sim(s, prompt) + c*sim(s, pool-centroid)
# plus MMR lambda and #sentences m. Centrality (pool centroid) favours consensus sentences.
import json, base64, time, re
import numpy as np
import octonion_gpt as G
from exp_fano_layer import unit
from octonion_pr_slot import toks, meanemb, slots, fit_slot_transport, apply_slot
from octonion_pr_extract import SOURCES, STOP, sents, content, rouge1

if __name__ == "__main__":
    t0 = time.time()
    pairs = []
    for f in SOURCES:
        for d in json.load(open(f)):
            q, a = d.get("question"), d.get("answer")
            if q and a and 3 <= len(q.split()) <= 40 and 5 <= len(a.split()) <= 80:
                pairs.append((q, a))
    rng = np.random.default_rng(0); pairs = [pairs[i] for i in rng.permutation(len(pairs))]
    train, test = pairs[:16000], pairs[16000:16300]
    M = G.build("\n".join(q + " " + a for q, a in train), vocab_size=9000, verbose=False); wi, emb = M["wi"], M["emb"]
    EPr = meanemb([toks(q, wi) for q, _ in train], emb); EAr = meanemb([toks(a, wi) for _, a in train], emb)
    Xtr, Ttr = slots(EPr), slots(EAr); EPru = unit(EPr)
    semb = lambda txt: unit(emb[toks(txt, wi)].mean(0)) if toks(txt, wi) else np.zeros(96)

    # cache per test prompt: pool sentences, cv matrix, anchor g, prompt pe, gold ge
    cache = []
    for q, gold in test:
        tq = toks(q, wi)
        if not tq: continue
        pe = unit(emb[tq].mean(0)); nn = np.argsort(-(EPru @ pe))[:40]
        F = fit_slot_transport(Xtr[nn], Ttr[nn], steps=10)
        g = unit(apply_slot(F, slots(emb[tq].mean(0)[None]))[0].reshape(96))
        pool = list(dict.fromkeys([s for j in nn for s in sents(train[int(j)][1])]))[:120]
        if not pool: continue
        cv = np.array([semb(s) for s in pool])
        cache.append((pool, cv, g, pe, cv.mean(0), semb(gold), gold))
    print("built %.0fs cached=%d" % (time.time()-t0, len(cache)))

    def mmr_score(cv, score, m, lam):
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

    def run(a, b, c, m=3, lam=0.7):
        embs = rg = 0.0
        for pool, cv, g, pe, cen, ge, gold in cache:
            score = a * (cv @ g) + b * (cv @ pe) + c * (cv @ cen)
            idx = mmr_score(cv, score, m, lam); ans = " ".join(pool[i] for i in idx)
            embs += float(np.sum(semb(ans) * ge)); rg += rouge1(ans, gold)
        n = len(cache); return embs/n, rg/n

    out = ["SELECTION-SCORE SWEEP (emb cos | ROUGE-1 F1):"]
    cfgs = [("anchor only            ", 1, 0, 0, 3, 0.7),
            ("anchor+prompt          ", 0.7, 0.3, 0, 3, 0.7),
            ("anchor+centrality      ", 0.7, 0, 0.3, 3, 0.7),
            ("anchor+prompt+central  ", 0.6, 0.2, 0.2, 3, 0.7),
            ("anchor+central (m=5)   ", 0.7, 0, 0.3, 5, 0.7),
            ("anchor+central (m=2)   ", 0.7, 0, 0.3, 2, 0.7),
            ("anchor+central lam=0.5 ", 0.7, 0, 0.3, 3, 0.5),
            ("anchor+central lam=0.9 ", 0.7, 0, 0.3, 3, 0.9)]
    for name, a, b, c, m, lam in cfgs:
        e, r = run(a, b, c, m, lam); out.append("  %s %.3f | %.3f" % (name, e, r))
    # oracle for reference
    oe = orr = 0.0
    for pool, cv, g, pe, cen, ge, gold in cache:
        idx = mmr_score(cv, cv @ ge, 3, 0.7); ans = " ".join(pool[i] for i in idx)
        oe += float(np.sum(semb(ans) * ge)); orr += rouge1(ans, gold)
    out.append("  %s %.3f | %.3f" % ("ORACLE                 ", oe/len(cache), orr/len(cache)))
    print("B64SW:" + base64.b64encode("\n".join(out).encode()).decode())
