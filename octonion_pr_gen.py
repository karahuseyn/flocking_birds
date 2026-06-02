# octonion_pr_gen.py -- the full chatbot: LOCAL Fano transport + GENERATIVE decode.
#
# Answer a prompt P' end to end, gradient-free & octonionic:
#   1. encode P' -> prompt mean-emb slots (12 octonions).
#   2. LOCAL transport: fit a per-slot Fano rotation on P's k nearest train prompts (the
#      validated local limit, slot-alignment 0.720) and apply it -> a sharp answer-region
#      anchor goal_emb (96-d, same PMI-SVD space the generator uses).
#   3. GENERATIVE decode: octonion_gpt.generate steered by goal_vec=goal_emb composes a
#      fluent answer in that region -- something retrieval (which can only recall) cannot do.
# Arms compared: retrieval (recall nearest answer) | generate w/ prompt goal | generate w/
# transport goal. Relevance = cos(mean-emb of produced answer, gold). base64 output.
import json, base64, time
import numpy as np
import octonion_gpt as G
from exp_fano_layer import unit
from octonion_pr_slot import toks, meanemb, slots, fit_slot_transport, apply_slot

if __name__ == "__main__":
    t0 = time.time()
    QA = json.load(open("webmdQAs.json"))
    pairs = [(d["question"], d["answer"]) for d in QA
             if d.get("question") and d.get("answer") and 3 <= len(d["question"].split()) <= 60]
    rng = np.random.default_rng(0); perm = rng.permutation(len(pairs)); pairs = [pairs[i] for i in perm]
    train, test = pairs[:16000], pairs[16000:16040]
    M = G.build("\n".join(q + " " + a for q, a in train), vocab_size=9000, verbose=False); wi, emb = M["wi"], M["emb"]
    EPr = meanemb([toks(q, wi) for q, _ in train], emb); EAr = meanemb([toks(a, wi) for _, a in train], emb)
    Xtr, Ttr = slots(EPr), slots(EAr); EPru, EAru = unit(EPr), unit(EAr)
    print("built %.0fs train=%d test=%d" % (time.time()-t0, len(train), len(test)))

    def goal_for(q, k=60, steps=10):
        tq = toks(q, wi)
        if not tq: return None, None
        xq = slots(emb[tq].mean(0)[None]); pe = unit(emb[tq].mean(0))
        nn = np.argsort(-(EPru @ pe))[:k]
        F = fit_slot_transport(Xtr[nn], Ttr[nn], steps=steps)
        g = apply_slot(F, xq)[0].reshape(96)                 # transported answer-region anchor
        return unit(g), pe

    def meanrel(ans, gold_e):
        t = toks(ans, wi); return float(np.sum(unit(emb[t].mean(0)) * gold_e)) if t else 0.0

    rret = rpg = rtg = 0.0; samples = []
    for qi, (q, gold) in enumerate(test):
        ge = unit(emb[toks(gold, wi)].mean(0)); g, pe = goal_for(q)
        if g is None: continue
        # retrieval: nearest train prompt's answer
        ridx = int(np.argmax(EPru @ pe)); retr = train[ridx][1]
        # seed the generator with the opener of the answer nearest the transported anchor
        seed = " ".join(train[int(np.argmax(EAru @ g))][1].split()[:3])
        gen_pg = G.generate(M, seed, n=42, rng_seed=1, goal_vec=pe, w_goal=4.0)   # prompt goal
        gen_tg = G.generate(M, seed, n=42, rng_seed=1, goal_vec=g,  w_goal=4.0)   # transport goal
        rret += meanrel(retr, ge); rpg += meanrel(gen_pg, ge); rtg += meanrel(gen_tg, ge)
        if qi < 5:
            samples += ["Q: " + q[:90], "  GOLD : " + gold[:120],
                        "  RETR : " + retr[:120], "  GEN-T: " + gen_tg[:120], ""]
    n = len(test)
    res = ["RELEVANCE to gold (cos mean-emb; n=%d):" % n,
           "  retrieval (recall nearest answer) = %.3f" % (rret/n),
           "  generate w/ prompt goal           = %.3f" % (rpg/n),
           "  generate w/ TRANSPORT goal        = %.3f" % (rtg/n), ""] + samples
    print("B64GEN:" + base64.b64encode("\n".join(res).encode()).decode())
