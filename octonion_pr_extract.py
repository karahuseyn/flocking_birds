# octonion_pr_extract.py -- EXTRACTIVE decode: turn the transported answer-region anchor
# into a fluent answer by SELECTING real sentences (no word-salad generation).
#
# Pipeline per prompt P':
#   1. encode P' -> prompt slots; local kNN per-slot Fano transport -> answer-region anchor g.
#   2. candidate pool = the sentences of P's k nearest train answers (the relevant region).
#   3. score each sentence by closeness to g; MMR-select a few (relevance + non-redundancy);
#      stitch in score order -> a fluent answer composed of real text, steered by transport.
# Fluency comes from real sentences; the transport decides WHICH -- and can compose across
# several neighbour answers (something whole-answer retrieval cannot). Gradient-free.
#
# Arms: whole-answer retrieval | extractive by PROMPT sim | extractive by TRANSPORT anchor |
#       oracle (sentences nearest the gold). Metrics: mean-emb cos + content-word ROUGE-1 F1.
import json, base64, time, re
import numpy as np
import octonion_gpt as G
from exp_fano_layer import unit
from octonion_pr_slot import toks, meanemb, slots, fit_slot_transport, apply_slot

SOURCES = ["medquadQAs.json", "webmdQAs.json", "icliniqQAs.json",
           "questionDoctorQAs.json", "ehealthforumQAs.json"]
STOP = set(("the a an of to and in is are was were be been being for on with as at by it "
            "this that these those i you he she we they my your his her our their have has had "
            "do does did but or if so not no can will would should could may might from out up "
            "about into over after before than then there here what which who when where how").split())

def sents(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 3]
def content(text):
    return set(w for w in re.findall(r"[a-z']+", text.lower()) if w not in STOP and len(w) > 2)
def rouge1(cand, gold):
    c, g = content(cand), content(gold)
    if not c or not g: return 0.0
    ov = len(c & g); p, r = ov / len(c), ov / len(g)
    return 0.0 if ov == 0 else 2 * p * r / (p + r)

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
    print("built %.0fs train=%d test=%d" % (time.time()-t0, len(train), len(test)))

    def mmr(cands, cvecs, target, m=3, lam=0.7):
        chosen = []
        sc = cvecs @ target
        while len(chosen) < m and len(chosen) < len(cands):
            best, bv = -1, -1e9
            for i in range(len(cands)):
                if i in chosen: continue
                red = max((cvecs[i] @ cvecs[j] for j in chosen), default=0.0)
                v = lam * sc[i] - (1 - lam) * red
                if v > bv: bv, best = v, i
            chosen.append(best)
        return [cands[i] for i in sorted(chosen, key=lambda i: -sc[i])]

    arms = {"retr": [0.0, 0.0], "ext_prompt": [0.0, 0.0], "ext_tran": [0.0, 0.0], "oracle": [0.0, 0.0]}
    samples = []
    for qi, (q, gold) in enumerate(test):
        tq = toks(q, wi)
        if not tq: continue
        pe = unit(emb[tq].mean(0)); nn = np.argsort(-(EPru @ pe))[:40]
        # local transport -> answer-region anchor
        F = fit_slot_transport(Xtr[nn], Ttr[nn], steps=10); g = unit(apply_slot(F, slots(emb[tq].mean(0)[None]))[0].reshape(96))
        ge = semb(gold)
        # candidate sentence pool from the neighbourhood answers
        pool = []
        for j in nn: pool += sents(train[int(j)][1])
        pool = list(dict.fromkeys(pool))[:120]
        if not pool: continue
        cv = np.array([semb(s) for s in pool])
        ans_retr = train[int(nn[0])][1]
        ans_ep = " ".join(mmr(pool, cv, pe, m=3))
        ans_et = " ".join(mmr(pool, cv, g, m=3))
        ans_or = " ".join(mmr(pool, cv, ge, m=3))
        for key, ans in [("retr", ans_retr), ("ext_prompt", ans_ep), ("ext_tran", ans_et), ("oracle", ans_or)]:
            arms[key][0] += float(np.sum(semb(ans) * ge)); arms[key][1] += rouge1(ans, gold)
        if qi < 5:
            samples += ["Q: " + q[:90], "  GOLD: " + gold[:130], "  RETR: " + ans_retr[:130],
                        "  EXT : " + ans_et[:130], ""]
    n = len(test)
    out = ["DECODE COMPARISON (n=%d):   mean-emb cos | ROUGE-1 F1" % n]
    for k in ["retr", "ext_prompt", "ext_tran", "oracle"]:
        out.append("  %-11s            %.3f       | %.3f" % (k, arms[k][0]/n, arms[k][1]/n))
    out.append(""); out += samples
    print("B64EXT:" + base64.b64encode("\n".join(out).encode()).decode())
