# octonion_bench.py -- real performance measurement on a MUCH bigger pool, incl. unusual /
# out-of-distribution questions. Pool = patient Q&A + 80k mined facts (~117k pairs). Reports:
#   DEPTH   : held-out patient-QA ROUGE-1 (no paraphrase twin in train for the held-out set)
#   BREADTH : held-out mined ROUGE-1
#   UNUSUAL : curated natural OOD questions -> answer + nearest-match confidence (is it guessing?)
import json, base64, time, numpy as np
from octonion_pr_bot import OctonionPRBot
from octonion_pr_big import load_full
from octonion_pr_slot import toks
from octonion_pr_extract import rouge1

UNUSUAL = ["can i take ibuprofen with alcohol?",
           "why do i feel dizzy when i stand up quickly?",
           "what helps with a hangover?",
           "can anxiety cause stomach problems?",
           "why do i get a headache when i drink coffee?",
           "is intermittent fasting safe?",
           "can lack of sleep make you gain weight?",
           "how do i know if a mole is dangerous?",
           "is it normal to feel tired all the time?",
           "can stress cause hair loss?",
           "what should i eat before a workout?",
           "is it bad to crack my knuckles?"]

if __name__ == "__main__":
    t0 = time.time()
    patient = load_full(); mined = [tuple(x) for x in json.load(open("mined_big.json"))]
    rng = np.random.default_rng(3); mined = [mined[i] for i in rng.permutation(len(mined))]
    p_test, p_train = patient[:400], patient[400:]
    m_test, m_train = mined[:400], mined[400:]
    pool = p_train + m_train
    print("POOL=%d (patient %d + mined %d) | tests: patient=400 mined=400" % (len(pool), len(p_train), len(m_train)))
    bot = OctonionPRBot().fit(pool, vocab_size=30000, verbose=True)
    print("fit done %.0fs" % (time.time()-t0))

    def rouge_on(testset):
        rr = n = 0
        for q, gold in testset:
            if not toks(q, bot.wi): continue
            rr += rouge1(bot.answer(q), gold); n += 1
        return rr/n, n
    dp, _ = rouge_on(p_test); br, _ = rouge_on(m_test)
    out = ["REAL PERF (pool=%d, fit %.0fs):" % (len(pool), time.time()-t0),
           "  DEPTH   patient-QA held-out ROUGE-1 = %.3f" % dp,
           "  BREADTH mined held-out   ROUGE-1 = %.3f" % br, ""]
    out.append("UNUSUAL / OUT-OF-DISTRIBUTION questions (answer + match confidence):")
    EPru = bot.EPru
    for q in UNUSUAL:
        pe, nn = bot._match(q, k=1, lex=0.6); conf = float(EPru[int(nn[0])] @ pe)
        ans, match = bot.answer(q, with_match=True)
        out += ["", "Q: " + q + "   [conf=%.2f]" % conf, "A: " + ans[:240], "   (nearest: " + (match or "-")[:55] + ")"]
    print("B64BENCH:" + base64.b64encode("\n".join(out).encode()).decode())
