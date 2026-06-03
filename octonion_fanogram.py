# octonion_fanogram.py -- does the order-sensitive fano-gram term fix phrase/order
# discrimination (type 2 vs type 1) that bag-of-words mean-emb loses, without harming
# overall matching? Probes nearest-neighbour entity discrimination + held-out ROUGE A/B.
import json, base64, numpy as np
from octonion_pr_bot import OctonionPRBot
from octonion_pr_big import load_full
from octonion_pr_slot import toks
from octonion_pr_extract import rouge1

if __name__ == "__main__":
    pool = load_full() + [tuple(x) for x in json.load(open("mined_qa.json"))]
    test = pool[:200]; train = pool[200:]
    bot = OctonionPRBot().fit(train, vocab_size=22000, verbose=True)

    def topk_contains(q, phrase, k=10, w_fano=0.0):
        _, nn = bot._match(q, k=k, lex=0.6, w_fano=w_fano)
        return sum(1 for j in nn if phrase in bot.train[int(j)][0].lower())

    probes = [("what is type 2 diabetes?", "type 2", "type 1"),
              ("what is type 1 diabetes?", "type 1", "type 2")]
    out = ["FANO-GRAM DISCRIMINATION (top-10 neighbours containing the right vs wrong phrase):"]
    for q, right, wrong in probes:
        for wf in [0.0, 3.0]:
            out.append("  '%s'  w_fano=%.0f :  '%s'=%d  '%s'=%d"
                       % (q, wf, right, topk_contains(q, right, 10, wf), wrong, topk_contains(q, wrong, 10, wf)))
    out.append("")

    def rouge_ab(w_fano):
        rr = n = 0
        for q, gold in test:
            if not toks(q, bot.wi): continue
            rr += rouge1(bot.answer(q, w_fano=w_fano), gold); n += 1
        return rr/n
    out.append("NO-HARM (patient-QA held-out ROUGE-1):")
    out.append("  w_fano=0.0 = %.3f" % rouge_ab(0.0))
    out.append("  w_fano=3.0 = %.3f" % rouge_ab(3.0))
    out.append("")
    out.append("ANSWER A/B (type 2 diabetes):")
    out.append("  w_fano=0: " + bot.answer("what is type 2 diabetes?", w_fano=0.0)[:230])
    out.append("  w_fano=3: " + bot.answer("what is type 2 diabetes?", w_fano=3.0)[:230])
    print("B64FG:" + base64.b64encode("\n".join(out).encode()).decode())
