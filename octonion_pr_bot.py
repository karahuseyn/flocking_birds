# octonion_pr_bot.py -- (c) the end-to-end chatbot: encode -> local Fano transport ->
# answer-region anchor -> extractive decode (anchor+centrality MMR over neighbour sentences).
# Gradient-free, octonionic, no backprop. Fit once, then .answer(q). Optional HTTP UI.
#
#   from octonion_pr_bot import OctonionPRBot
#   bot = OctonionPRBot().fit(pairs); print(bot.answer("i have a headache and fever"))
#   bot.serve(8000)          # browser UI at http://localhost:8000  (local/Kaggle)
import json, time, math, re
import numpy as np
from collections import Counter
import octonion_gpt as G
from exp_fano_layer import unit
from octonion_pr_slot import toks, slots, fit_slot_transport, apply_slot
from octonion_pr_extract import SOURCES, sents, rouge1
from exp_fano_layer import octo_mul

class OctonionPRBot:
    def fit(self, pairs, vocab_size=10000, use_idf=True, emb_corpus=None, verbose=True):
        # emb_corpus: optional larger text to LEARN EMBEDDINGS on (e.g. QA + biomed) while the
        # retrieval/transport POOL stays the QA pairs -- richer vectors, sharper matching.
        self.train = list(pairs)
        t0 = time.time()
        build_text = emb_corpus if emb_corpus is not None else "\n".join(q + " " + a for q, a in self.train)
        self.M = G.build(build_text, vocab_size=vocab_size, verbose=False)
        self.wi, self.emb = self.M["wi"], self.M["emb"]
        self.octo = unit(self.emb[:, 1:9])                 # one octonion per token (for fano-grams)
        # IDF: rare/specific terms (leg cramps, dehydration) outweigh common ones (at, night)
        self.idf = np.ones(self.M["W"])
        if use_idf:
            df = Counter()
            for q, a in self.train:
                for tk in set(toks(q + " " + a, self.wi)): df[tk] += 1
            N = len(self.train)
            for tk, c in df.items(): self.idf[tk] = math.log((N + 1) / (c + 1)) + 1.0
        EPr = np.array([self._vec(q) for q, _ in self.train])
        EAr = np.array([self._vec(a) for _, a in self.train])
        self.Xtr, self.Ttr, self.EPru = slots(EPr), slots(EAr), unit(EPr)
        # inverted index over prompt content tokens (idf>1) for entity-weighted lexical match:
        # sharing a rare entity token (flu, ibuprofen) outweighs sharing a template word (prevent)
        post = {}
        for i, (q, _) in enumerate(self.train):
            for t in set(toks(q, self.wi)):
                if self.idf[t] > 1.0: post.setdefault(t, []).append(i)
        self.post = {t: np.array(v) for t, v in post.items()}
        # FANO-GRAM index: sparse inverted index over ORDER-SENSITIVE bigrams (the token pairs
        # whose fano-gram octonion is o(w_i)(x)o(w_{i+1})). Sharing 'type 2' binds differently
        # from 'type 1', giving the phrase/order discrimination the bag-of-words mean-emb loses,
        # without the capacity crosstalk of a single-octonion superposition.
        bpost = {}
        for i, (q, _) in enumerate(self.train):
            t = toks(q, self.wi)
            for a, b in zip(t, t[1:]): bpost.setdefault((a, b), []).append(i)
        self.bpost = {bg: np.array(v) for bg, v in bpost.items()}
        if verbose: print("fit %.0fs  pairs=%d vocab=%d idf=%s" % (time.time()-t0, len(self.train), self.M["W"], use_idf))
        return self

    def _bigrams(self, txt):
        t = toks(txt, self.wi); return list(zip(t, t[1:]))

    def _vec(self, txt):
        t = toks(txt, self.wi)
        if not t: return np.zeros(96)
        w = self.idf[t][:, None]
        return unit((self.emb[t] * w).sum(0))

    _semb = _vec

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

    def _match(self, question, k=40, lex=0.6, w_fano=0.0):
        # hybrid neighbour ranking: dense IDF-emb cosine + lex * IDF-weighted token overlap
        # + w_fano * fano-gram (order-sensitive phrase) similarity
        pe = self._vec(question); dense = self.EPru @ pe
        qtok = [t for t in set(toks(question, self.wi)) if self.idf[t] > 1.0]
        if lex > 0 and qtok:
            L = np.zeros(len(self.train)); tot = 0.0
            for t in qtok:
                w = self.idf[t]; tot += w; p = self.post.get(t)
                if p is not None: L[p] += w
            dense = dense + lex * (L / (tot + 1e-9))
        if w_fano:
            bg = self._bigrams(question)
            if bg:
                B = np.zeros(len(self.train))
                for g in bg:
                    p = self.bpost.get(g)
                    if p is not None: B[p] += 1.0
                dense = dense + w_fano * (B / len(bg))             # order-sensitive bigram overlap
        return pe, np.argsort(-dense)[:k]

    def answer(self, question, k=40, m=3, steps=10, with_match=False, lex=0.6, w_fano=3.0,
               w_anchor=0.6, w_central=0.2, w_query=0.2, w_nbr=0.15, tau=0.25):
        # rerank weights: anchor (transported region) + centrality (consensus) + query
        # relevance + source-neighbour relevance; tau hard-drops off-topic sentences (<tau of
        # the max query similarity) to kill cross-topic bleed-through. Defaults = reranked.
        tq = toks(question, self.wi)
        if not tq: return ("i'm not sure i understood that.", None) if with_match else "i'm not sure i understood that."
        pe, nn = self._match(question, k=k, lex=lex, w_fano=w_fano)                # entity + fano-gram match
        F = fit_slot_transport(self.Xtr[nn], self.Ttr[nn], steps=steps)            # local transport
        g = unit(apply_slot(F, slots(pe[None]))[0].reshape(96))                    # answer-region anchor
        seen, pool, nbr = set(), [], []                                            # keep each sentence's best source-neighbour sim
        for j in nn:
            nbs = float(self.EPru[int(j)] @ pe)
            for s in sents(self.train[int(j)][1]):
                if s not in seen: seen.add(s); pool.append(s); nbr.append(nbs)
        pool, nbr = pool[:160], np.array(nbr[:160])
        if not pool: return ("i don't have information on that yet.", None) if with_match else "i don't have information on that yet."
        cv = np.array([self._semb(s) for s in pool]); cen = cv.mean(0)
        qs = cv @ pe
        keep = qs >= tau * qs.max()                                                # drop off-topic sentences
        if keep.sum() >= m: pool = [pool[i] for i in np.where(keep)[0]]; cv = cv[keep]; nbr = nbr[keep]; qs = qs[keep]
        score = w_anchor * (cv @ g) + w_central * (cv @ cen) + w_query * qs + w_nbr * nbr
        idx = self._mmr(cv, score, m=m)
        ans = " ".join(pool[i] for i in idx)
        return (ans, self.train[int(nn[0])][0]) if with_match else ans

    _SEGSTOP = set(("im i'm you're dont don't cant can't really very much pretty quite lately "
                    "always think feel feeling getting going they them their your his her our "
                    "this that these those coming back gone been have having from with about").split())

    def _segments(self, prompt):
        """Split a long, multi-part prompt into sub-questions at clause / conjunction breaks;
        keep each segment that carries a salient (rare, >=4-letter, non-filler) content token."""
        segs = []
        for p in re.split(r"\band\b|\balso\b|\bplus\b|\bas well as\b|\bbut\b|[,;.?]", prompt.lower()):
            ct = sorted((t for t in dict.fromkeys(toks(p, self.wi))
                         if self.idf[t] > 1.5 and len(self.M["vocab"][t]) >= 4
                         and self.M["vocab"][t] not in self._SEGSTOP), key=lambda t: -self.idf[t])
            if ct: segs.append((p.strip(), ct))
        return segs

    _ASPECT = set(("treated treat treatment prevent prevention prevented cure cured manage "
                   "managed diagnosed recur recurrence relieve relieved").split())
    _REQUEST = set(("should advice help anything suggestion suggestions recommend please "
                    "what do").split())

    def respond(self, prompt, m=3):
        """Consolidated entry point. Short prompt -> one answer; long multi-part prompt ->
        capture each part, route it through match->transport->extract, answer each. Drops pure
        request phrases ('what should i do'); for entity-less aspect clauses ('how are they
        treated') it carries the prompt's dominant entity so context isn't lost."""
        segs = self._segments(prompt)
        if len(segs) <= 1:
            return self.answer(prompt, m=m)
        # dominant entity PHRASE = top-2 idf among real-entity tokens (not aspect/request words)
        ent_tok = [t for _, ct in segs for t in ct
                   if self.M["vocab"][t] not in self._ASPECT and self.M["vocab"][t] not in self._REQUEST]
        gphrase = " ".join(self.M["vocab"][t] for t in sorted(set(ent_tok), key=lambda t: -self.idf[t])[:2])
        out, used = [], set()
        for text, ct in segs:
            words = [self.M["vocab"][t] for t in ct]
            if all(w in self._REQUEST for w in words):
                continue                                                    # drop request noise
            query = (gphrase + " " + text) if all(w in self._ASPECT for w in words) else text
            ans, match = self.answer(query, m=m, with_match=True)
            if match in used:
                continue
            used.add(match)
            out.append("[%s] %s" % (" / ".join(words[:2]), ans))
        return "\n".join(out)

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
