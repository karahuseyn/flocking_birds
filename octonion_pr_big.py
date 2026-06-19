# octonion_pr_big.py -- bigger data: use the FULL Q&A pool (all 5 sources, incl. the long
# factual medquad answers we used to filter out -- they add rich candidate sentences) at a
# larger vocab. Reports held-out ROUGE-1 (encoding-independent, the fair metric) + a fixed
# plain-emb relevance, and answers 10 example prompts (factoid + conversational).
import json, base64, time
import numpy as np
from octonion_pr_bot import OctonionPRBot, SOURCES
from octonion_pr_slot import toks
from octonion_pr_extract import rouge1

def load_full():
    pairs = []
    for f in SOURCES:
        for d in json.load(open(f)):
            q, a = d.get("question"), d.get("answer")
            if q and a and 3 <= len(q.split()) <= 50 and 5 <= len(a.split()) <= 400:
                pairs.append((q, a))
    rng = np.random.default_rng(0); return [pairs[i] for i in rng.permutation(len(pairs))]

if __name__ == "__main__":
    t0 = time.time()
    pairs = load_full(); test = pairs[:200]; train = pairs[200:]
    print("FULL pool=%d  train=%d  test=200" % (len(pairs), len(train)))
    bot = OctonionPRBot().fit(train, vocab_size=20000, verbose=True)
    # fixed plain-emb relevance (comparable across runs) + fair ROUGE-1
    def plain(txt):
        t = toks(txt, bot.wi)
        if not t: return np.zeros(96)
        v = bot.emb[t].mean(0); n = np.linalg.norm(v); return v/n if n > 1e-9 else v
    ee = rr = n = 0
    for q, gold in test:
        if not toks(q, bot.wi): continue
        ans = bot.answer(q); ee += float(np.sum(plain(ans) * plain(gold))); rr += rouge1(ans, gold); n += 1
    out = ["BIGGER DATA (%d-pair pool, vocab=%d, %d test):" % (len(train), bot.M["W"], n),
           "  held-out  plain-emb=%.3f  ROUGE-1=%.3f" % (ee/n, rr/n), ""]
    qs = ["what are the symptoms of dehydration?",
          "what causes leg cramps at night?",
          "how is high blood pressure treated?",
          "what is type 2 diabetes?",
          "what are the symptoms of a heart attack?",
          "can stress cause chest pain?",
          "i can't sleep at night, what can help?",
          "what should i do for a sore throat?",
          "what are the side effects of ibuprofen?",
          "how can i prevent the flu?"]
    out.append("10 EXAMPLE PROMPTS (bigger-data IDF bot):")
    for q in qs:
        ans, match = bot.answer(q, with_match=True)
        out += ["", "Q: " + q, "A: " + ans[:300], "   (nearest stored Q: " + (match or "-")[:60] + ")"]
    print("done %.0fs" % (time.time()-t0))
    print("B64BIG:" + base64.b64encode("\n".join(out).encode()).decode())
