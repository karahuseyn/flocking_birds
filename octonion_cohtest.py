# octonion_cohtest.py -- cos^2-coherence ordering (no-leakage equiangular chain) vs exposition
# role ordering, on held-out patient-QA: ROUGE-1 + mean chain coherence (prod cos^2).
import json, base64, numpy as np
from octonion_pr_bot import OctonionPRBot
from octonion_pr_big import load_full
from octonion_composer import compose_logic
from octonion_pr_slot import toks
from octonion_pr_extract import rouge1

if __name__ == "__main__":
    pool = load_full() + [tuple(x) for x in json.load(open("mined_qa.json"))]
    test, train = pool[:200], pool[200:]
    bot = OctonionPRBot().fit(train, vocab_size=22000, verbose=True)
    out = ["ORDERING A/B (held-out 200): ROUGE-1 | mean chain-coherence (prod cos^2)"]
    for mode in ["role", "coherence"]:
        rr = co = n = 0
        for q, gold in test:
            if not toks(q, bot.wi): continue
            txt, _, coh = compose_logic(bot, q, order_mode=mode, return_meta=True)
            rr += rouge1(txt, gold); co += coh; n += 1
        out.append("  %-10s  ROUGE-1=%.3f  coherence=%.3f" % (mode, rr/n, co/n))
    print("B64CO:" + base64.b64encode("\n".join(out).encode()).decode())
