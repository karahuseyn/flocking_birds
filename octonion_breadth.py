# octonion_breadth.py -- make the COVERAGE (breadth) gain measurable. Hold out a slice of the
# mined pairs as a breadth test set; evaluate base (patient-only) vs augmented (patient+mined)
# on BOTH the patient-QA test (depth / no-harm) and the mined test (breadth). The metric only
# saw depth before; this shows the breadth gain numerically.
import json, base64, numpy as np
from octonion_pr_bot import OctonionPRBot
from octonion_pr_big import load_full
from octonion_pr_slot import toks
from octonion_pr_extract import rouge1

if __name__ == "__main__":
    patient = load_full(); p_test = patient[:200]; p_train = patient[200:]
    mined = [tuple(x) for x in json.load(open("mined_qa.json"))]
    rng = np.random.default_rng(1); mined = [mined[i] for i in rng.permutation(len(mined))]
    m_test = mined[:300]; m_train = mined[300:]
    print("patient_train=%d mined_train=%d | tests: patient=200 mined=300" % (len(p_train), len(m_train)))

    def rouge_on(bot, testset):
        rr = n = 0
        for q, gold in testset:
            if not toks(q, bot.wi): continue
            rr += rouge1(bot.answer(q), gold); n += 1
        return rr/n, n

    base = OctonionPRBot().fit(p_train, vocab_size=20000, verbose=True)
    aug = OctonionPRBot().fit(p_train + m_train, vocab_size=24000, verbose=True)
    bp, _ = rouge_on(base, p_test); ap, _ = rouge_on(aug, p_test)
    bm, _ = rouge_on(base, m_test); am, _ = rouge_on(aug, m_test)
    out = ["DEPTH vs BREADTH (ROUGE-1):",
           "                         patient-test(depth)   mined-test(breadth)",
           "  base  (patient only)        %.3f                 %.3f" % (bp, bm),
           "  aug   (patient+mined)       %.3f                 %.3f" % (ap, am),
           "  breadth gain: %.3f -> %.3f  (x%.1f)" % (bm, am, am/(bm+1e-9)), ""]
    out.append("MINED-TEST EXAMPLES (base vs aug):")
    for q, gold in m_test[:6]:
        out += ["", "Q: " + q, "  GOLD: " + gold[:110],
                "  BASE: " + base.answer(q)[:110], "  AUG : " + aug.answer(q)[:110]]
    print("B64BR:" + base64.b64encode("\n".join(out).encode()).decode())
