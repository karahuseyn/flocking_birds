# octonion_pr_match.py -- entity-weighted matching: sweep the lexical-overlap weight `lex`
# and check (a) held-out ROUGE-1 / plain-emb, (b) whether the nearest stored Q for the
# bleed-through prompts (flu, ibuprofen) snaps to the right entity.
import base64, numpy as np
from octonion_pr_bot import OctonionPRBot
from octonion_pr_big import load_full
from octonion_pr_slot import toks
from octonion_pr_extract import rouge1

if __name__ == "__main__":
    pairs = load_full(); test = pairs[:200]; train = pairs[200:]
    bot = OctonionPRBot().fit(train, vocab_size=20000, verbose=True)
    def plain(txt):
        t = toks(txt, bot.wi)
        if not t: return np.zeros(96)
        v = bot.emb[t].mean(0); nrm = np.linalg.norm(v); return v/nrm if nrm > 1e-9 else v
    probes = ["how can i prevent the flu?", "what are the side effects of ibuprofen?",
              "what are the symptoms of dehydration?", "how is high blood pressure treated?"]
    out = ["ENTITY-WEIGHTED MATCH SWEEP (full %d pool, 200 test):" % len(train)]
    for lex in [0.0, 0.6, 1.2]:
        ee = rr = n = 0
        for q, gold in test:
            if not toks(q, bot.wi): continue
            ans = bot.answer(q, lex=lex); ee += float(np.sum(plain(ans) * plain(gold))); rr += rouge1(ans, gold); n += 1
        out.append("  lex=%.1f  plain-emb=%.3f  ROUGE-1=%.3f" % (lex, ee/n, rr/n))
    out.append("")
    out.append("NEAREST STORED Q per lex (does it snap to the right entity?):")
    for q in probes:
        out.append("Q: " + q)
        for lex in [0.0, 0.6, 1.2]:
            _, nn = bot._match(q, k=1, lex=lex)
            out.append("   lex=%.1f -> %s" % (lex, train[int(nn[0])][0][:62]))
    out.append("")
    out.append("EXAMPLES (lex=0.6):")
    for q in probes + ["can stress cause chest pain?", "what should i do for a sore throat?"]:
        ans, match = bot.answer(q, with_match=True, lex=0.6)
        out += ["", "Q: " + q, "A: " + ans[:240], "   (nearest: " + (match or "-")[:55] + ")"]
    print("B64M:" + base64.b64encode("\n".join(out).encode()).decode())
