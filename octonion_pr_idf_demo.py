# octonion_pr_idf_demo.py -- IDF-weighted matching: quantitative A/B (plain vs IDF) on a
# held-out test, then 10 example prompts answered by the IDF bot (with the nearest stored Q).
import base64, numpy as np
from octonion_pr_bot import OctonionPRBot, _load_pairs
from octonion_pr_slot import toks
from octonion_pr_extract import rouge1

if __name__ == "__main__":
    pairs = _load_pairs()
    test = pairs[:200]; train = pairs[200:20200]
    out = []
    # A/B: relevance of the extractive answer with plain vs IDF-weighted matching
    for use_idf in [False, True]:
        bot = OctonionPRBot().fit(train, use_idf=use_idf, verbose=True)
        ee = rr = n = 0
        for q, gold in test:
            if not toks(q, bot.wi): continue
            ans = bot.answer(q); ge = bot._vec(gold)
            ee += float(np.sum(bot._vec(ans) * ge)); rr += rouge1(ans, gold); n += 1
        out.append("  %-5s  emb=%.3f  ROUGE-1=%.3f" % ("IDF" if use_idf else "plain", ee/n, rr/n))
        if use_idf: idf_bot = bot
    out = ["MATCHING A/B (200 held-out, extractive answer relevance):"] + out + [""]

    # 10 example prompts (natural, mostly unseen phrasings)
    qs = ["i have a headache and a fever, what should i do?",
          "what causes leg cramps at night?",
          "what are the symptoms of dehydration?",
          "can stress cause chest pain?",
          "how can i lower my blood pressure naturally?",
          "is it safe to exercise when i have a cold?",
          "my child has an itchy rash, should i be worried?",
          "i can't sleep at night, what can help?",
          "what should i do for a sore throat?",
          "are migraines and regular headaches different?"]
    out.append("10 EXAMPLE PROMPTS (IDF bot):")
    for q in qs:
        ans, match = idf_bot.answer(q, with_match=True)
        out += ["", "Q: " + q, "A: " + ans[:280], "   (nearest stored Q: " + (match or "-")[:62] + ")"]
    print("B64IDF:" + base64.b64encode("\n".join(out).encode()).decode())
