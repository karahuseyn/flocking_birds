# octonion_pr_rerank.py -- test the rerank (query + source-neighbour relevance + off-topic
# threshold) against the old anchor+centrality selection, on the full 36.7k pool. Reports
# held-out ROUGE-1 / plain-emb for both, then the 10 examples with the reranked bot.
import base64, numpy as np
from octonion_pr_bot import OctonionPRBot
from octonion_pr_big import load_full
from octonion_pr_slot import toks
from octonion_pr_extract import rouge1

OLD = dict(w_anchor=0.7, w_central=0.3, w_query=0.0, w_nbr=0.0, tau=0.0)
NEW = dict(w_anchor=0.6, w_central=0.2, w_query=0.2, w_nbr=0.15, tau=0.25)

if __name__ == "__main__":
    pairs = load_full(); test = pairs[:200]; train = pairs[200:]
    bot = OctonionPRBot().fit(train, vocab_size=20000, verbose=True)
    def plain(txt):
        t = toks(txt, bot.wi)
        if not t: return np.zeros(96)
        v = bot.emb[t].mean(0); nrm = np.linalg.norm(v); return v/nrm if nrm > 1e-9 else v
    out = ["RERANK A/B (full %d pool, 200 test):" % len(train)]
    for name, cfg in [("old anchor+central", OLD), ("reranked          ", NEW)]:
        ee = rr = n = 0
        for q, gold in test:
            if not toks(q, bot.wi): continue
            ans = bot.answer(q, **cfg); ee += float(np.sum(plain(ans) * plain(gold))); rr += rouge1(ans, gold); n += 1
        out.append("  %s  plain-emb=%.3f  ROUGE-1=%.3f" % (name, ee/n, rr/n))
    out.append("")
    qs = ["how can i prevent the flu?", "what are the side effects of ibuprofen?",
          "what are the symptoms of dehydration?", "what causes leg cramps at night?",
          "can stress cause chest pain?", "what is type 2 diabetes?",
          "how is high blood pressure treated?", "what should i do for a sore throat?",
          "i can't sleep at night, what can help?", "what are the symptoms of a heart attack?"]
    out.append("10 EXAMPLES (reranked):")
    for q in qs:
        ans, match = bot.answer(q, with_match=True, **NEW)
        out += ["", "Q: " + q, "A: " + ans[:300], "   (nearest: " + (match or "-")[:55] + ")"]
    print("B64RR:" + base64.b64encode("\n".join(out).encode()).decode())
