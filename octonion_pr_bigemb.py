# octonion_pr_bigemb.py -- bigger data via EMBEDDING ENRICHMENT: learn the octonion
# embeddings on the QA text PLUS a slice of the 320MB biomed corpus (richer, more
# distinctive medical-term vectors -> sharper matching), while the retrieval/transport pool
# stays the full Q&A. Compares QA-only embeddings vs QA+biomed embeddings; same pool/test.
import base64, time
import numpy as np
from octonion_pr_bot import OctonionPRBot
from octonion_pr_big import load_full
from octonion_pr_slot import toks
from octonion_pr_extract import rouge1

NEW = dict(w_anchor=0.6, w_central=0.2, w_query=0.2, w_nbr=0.15, tau=0.25)

if __name__ == "__main__":
    pairs = load_full(); test = pairs[:200]; train = pairs[200:]
    qa_text = "\n".join(q + " " + a for q, a in train)
    bio = open("corpus_biomed.txt", encoding="utf-8", errors="ignore").read(60_000_000)  # 60MB slice
    big_text = qa_text + "\n" + bio
    print("qa_text=%.0fMB  big_text=%.0fMB" % (len(qa_text)/1e6, len(big_text)/1e6))

    def evaluate(bot):
        def plain(txt):
            t = toks(txt, bot.wi)
            if not t: return np.zeros(96)
            v = bot.emb[t].mean(0); nrm = np.linalg.norm(v); return v/nrm if nrm > 1e-9 else v
        ee = rr = n = 0
        for q, gold in test:
            if not toks(q, bot.wi): continue
            ans = bot.answer(q, **NEW); ee += float(np.sum(plain(ans) * plain(gold))); rr += rouge1(ans, gold); n += 1
        return ee/n, rr/n, n

    out = ["EMBEDDING ENRICHMENT (full %d pool, 200 test):" % len(train)]
    t0 = time.time()
    b1 = OctonionPRBot().fit(train, vocab_size=20000, verbose=True)
    e1, r1, n = evaluate(b1); out.append("  QA-only emb (vocab %d)     plain-emb=%.3f ROUGE-1=%.3f" % (b1.M["W"], e1, r1))
    b2 = OctonionPRBot().fit(train, vocab_size=30000, emb_corpus=big_text, verbose=True)
    e2, r2, _ = evaluate(b2); out.append("  QA+biomed emb (vocab %d)  plain-emb=%.3f ROUGE-1=%.3f" % (b2.M["W"], e2, r2))
    out.append("")
    qs = ["how can i prevent the flu?", "what are the side effects of ibuprofen?",
          "what are the symptoms of dehydration?", "can stress cause chest pain?",
          "how is high blood pressure treated?", "what should i do for a sore throat?"]
    out.append("EXAMPLES (QA+biomed bot):")
    for q in qs:
        ans, match = b2.answer(q, with_match=True, **NEW)
        out += ["", "Q: " + q, "A: " + ans[:260], "   (nearest: " + (match or "-")[:55] + ")"]
    print("done %.0fs" % (time.time()-t0))
    print("B64BE:" + base64.b64encode("\n".join(out).encode()).decode())
