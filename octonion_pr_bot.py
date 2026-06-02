# octonion_pr_bot.py -- (c) the end-to-end chatbot: encode -> local Fano transport ->
# answer-region anchor -> extractive decode (anchor+centrality MMR over neighbour sentences).
# Gradient-free, octonionic, no backprop. Fit once, then .answer(q). Optional HTTP UI.
#
#   from octonion_pr_bot import OctonionPRBot
#   bot = OctonionPRBot().fit(pairs); print(bot.answer("i have a headache and fever"))
#   bot.serve(8000)          # browser UI at http://localhost:8000  (local/Kaggle)
import json, time, re
import numpy as np
import octonion_gpt as G
from exp_fano_layer import unit
from octonion_pr_slot import toks, meanemb, slots, fit_slot_transport, apply_slot
from octonion_pr_extract import SOURCES, sents, rouge1

class OctonionPRBot:
    def fit(self, pairs, vocab_size=10000, verbose=True):
        self.train = list(pairs)
        t0 = time.time()
        self.M = G.build("\n".join(q + " " + a for q, a in self.train), vocab_size=vocab_size, verbose=False)
        self.wi, self.emb = self.M["wi"], self.M["emb"]
        EPr = meanemb([toks(q, self.wi) for q, _ in self.train], self.emb)
        EAr = meanemb([toks(a, self.wi) for _, a in self.train], self.emb)
        self.Xtr, self.Ttr, self.EPru = slots(EPr), slots(EAr), unit(EPr)
        if verbose: print("fit %.0fs  pairs=%d vocab=%d" % (time.time()-t0, len(self.train), self.M["W"]))
        return self

    def _semb(self, txt):
        t = toks(txt, self.wi); return unit(self.emb[t].mean(0)) if t else np.zeros(96)

    def _mmr(self, cv, score, m=3, lam=0.7):
        chosen = []
        while len(chosen) < m and len(chosen) < len(cv):
            best, bv = -1, -1e9
            for i in range(len(cv)):
                if i in chosen: continue
                red = max((cv[i] @ cv[j] for j in chosen), default=0.0)
                v = lam * score[i] - (1 - lam) * red
                if v > bv: bv, best = v, i
            chosen.append(best)
        return sorted(chosen, key=lambda i: -score[i])

    def answer(self, question, k=40, m=3, steps=10, with_match=False):
        tq = toks(question, self.wi)
        if not tq: return ("i'm not sure i understood that.", None) if with_match else "i'm not sure i understood that."
        pe = unit(self.emb[tq].mean(0)); nn = np.argsort(-(self.EPru @ pe))[:k]
        F = fit_slot_transport(self.Xtr[nn], self.Ttr[nn], steps=steps)            # local transport
        g = unit(apply_slot(F, slots(self.emb[tq].mean(0)[None]))[0].reshape(96))  # answer-region anchor
        pool = list(dict.fromkeys([s for j in nn for s in sents(self.train[int(j)][1])]))[:140]
        if not pool: return ("i don't have information on that yet.", None) if with_match else "i don't have information on that yet."
        cv = np.array([self._semb(s) for s in pool]); cen = cv.mean(0)
        idx = self._mmr(cv, 0.7 * (cv @ g) + 0.3 * (cv @ cen), m=m)
        ans = " ".join(pool[i] for i in idx)
        return (ans, self.train[int(nn[0])][0]) if with_match else ans

    def serve(self, port=8000):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from urllib.parse import urlparse, parse_qs
        bot = self
        PAGE = ("<!doctype html><meta charset=utf-8><title>octonion transport bot</title>"
                "<style>body{font-family:system-ui;max-width:680px;margin:40px auto;padding:0 16px}"
                "input{width:100%;font-size:16px;padding:11px;border:1px solid #ccc;border-radius:8px}"
                ".a{background:#f4f6f8;border-radius:8px;padding:14px;margin-top:14px;white-space:pre-wrap}"
                ".m{color:#888;font-size:12px;margin-top:6px}</style>"
                "<h2>octonion transport bot</h2><div class=m>gradient-free: encode &rarr; Fano "
                "transport &rarr; extractive decode &middot; not medical advice</div>"
                "<input id=q placeholder='ask a health question...' autofocus>"
                "<div id=o></div><script>"
                "q.addEventListener('keydown',async e=>{if(e.key!='Enter')return;"
                "o.innerHTML='<p class=m>thinking...</p>';"
                "let r=await fetch('/ask?q='+encodeURIComponent(q.value));let d=await r.json();"
                "o.innerHTML='<div class=a>'+d.answer+'</div><div class=m>nearest: '+d.match+'</div>'})"
                "</script>")

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                u = urlparse(self.path)
                if u.path == "/":
                    body = PAGE.encode()
                    self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
                elif u.path == "/ask":
                    q = parse_qs(u.query).get("q", [""])[0]
                    ans, match = bot.answer(q, with_match=True)
                    body = json.dumps({"answer": ans, "match": match or ""}).encode()
                    self.send_response(200); self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
                else:
                    self.send_response(404); self.end_headers()
        print("serving on http://localhost:%d" % port)
        ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()

def _load_pairs(limit=None):
    pairs = []
    for f in SOURCES:
        for d in json.load(open(f)):
            q, a = d.get("question"), d.get("answer")
            if q and a and 3 <= len(q.split()) <= 50 and 5 <= len(a.split()) <= 150:
                pairs.append((q, a))
    rng = np.random.default_rng(0); pairs = [pairs[i] for i in rng.permutation(len(pairs))]
    return pairs[:limit] if limit else pairs

if __name__ == "__main__":
    import sys, base64
    bot = OctonionPRBot().fit(_load_pairs(20000))
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        bot.serve(int(sys.argv[2]) if len(sys.argv) > 2 else 8000)
    else:
        qs = ["i have a headache and a fever, what should i do?",
              "is it safe to exercise when i have a cold?",
              "how can i lower my blood pressure naturally?",
              "what causes leg cramps at night?",
              "my child has a rash and is itchy, should i be worried?",
              "can stress cause chest pain?",
              "what are the symptoms of dehydration?",
              "i can't sleep at night, what can help?"]
        out = []
        for q in qs:
            ans, match = bot.answer(q, with_match=True)
            out += ["Q: " + q, "A: " + ans[:300], "   (nearest stored Q: " + (match or "-")[:70] + ")", ""]
        print("B64BOT:" + base64.b64encode("\n".join(out).encode()).decode())
