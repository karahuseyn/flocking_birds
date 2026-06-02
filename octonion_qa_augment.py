# octonion_qa_augment.py -- augment the Q&A pool with the mined definitional pairs and test:
# (1) no-harm: does held-out patient-QA ROUGE-1 stay >= baseline?  (2) coverage: can the bot
# now answer definitional drug/disease questions the patient pool never had?
import json, base64, numpy as np
from octonion_pr_bot import OctonionPRBot
from octonion_pr_big import load_full
from octonion_pr_slot import toks
from octonion_pr_extract import rouge1

if __name__ == "__main__":
    patient = load_full(); test = patient[:200]; train_p = patient[200:]
    mined = [tuple(x) for x in json.load(open("mined_qa.json"))]
    print("patient train=%d  mined=%d  test=200" % (len(train_p), len(mined)))

    def plain(bot, txt):
        t = toks(txt, bot.wi)
        if not t: return np.zeros(96)
        v = bot.emb[t].mean(0); nrm = np.linalg.norm(v); return v/nrm if nrm > 1e-9 else v
    def rouge_on_test(bot):
        rr = n = 0
        for q, gold in test:
            if not toks(q, bot.wi): continue
            rr += rouge1(bot.answer(q), gold); n += 1
        return rr/n

    out = []
    base = OctonionPRBot().fit(train_p, vocab_size=20000, verbose=True)
    aug = OctonionPRBot().fit(train_p + mined, vocab_size=24000, verbose=True)
    out.append("NO-HARM (patient-QA held-out ROUGE-1):")
    out.append("  baseline (patient only)      = %.3f" % rouge_on_test(base))
    out.append("  augmented (+%d mined pairs)  = %.3f" % (len(mined), rouge_on_test(aug)))
    out.append("")
    cov = ["what is misoprostol?", "what is febuxostat?", "what is alcoholic hepatitis?",
           "what is mirabegron?", "what is osteoarthritis?", "what is a calcium channel blocker?",
           "what is sepsis?", "what is metformin used for?"]
    out.append("COVERAGE -- definitional questions (baseline vs augmented):")
    for q in cov:
        out += ["", "Q: " + q,
                "  BASE: " + base.answer(q)[:150],
                "  AUG : " + aug.answer(q)[:150]]
    print("B64AUG:" + base64.b64encode("\n".join(out).encode()).decode())
