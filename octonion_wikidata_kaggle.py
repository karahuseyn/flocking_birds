#!/usr/bin/env python3
"""
OCTONIONIC GRADIENT-FREE KNOWLEDGE BOT  --  single-file Kaggle script (Wikidata).

No backprop, no gradients, no training loop. The whole model is linear algebra over the
octonions:
  * PMI-SVD word embeddings (one-shot truncated SVD)            -> meaning
  * one octonion o(w) = unit(emb[w][1:9]) in S^7 per token       -> structure
  * entity-weighted hybrid match: dense + IDF lexical + FANO-GRAM (order-sensitive bigram)
  * local SO(8) transport: matching pursuit over the 28-element so(8) basis {E_i E_j}
    (Freedman-Shokrian-Zini-Wang) -> a predicted answer-region anchor
  * extractive decode (MMR) over the matched neighbourhood
  * multi-part respond(): split a long prompt, route each concern, carry the topic entity

USAGE on Kaggle:
  1. Add a Wikidata dataset (any of: JSON/JSONL dump with labels+descriptions, or a CSV/
     parquet with label/description or question/answer columns) to the notebook.
  2. Run this file. It auto-discovers data under /kaggle/input, builds the model, and
     answers the PROMPTS below.
  Tune with env vars:  MAXPAIRS (default 200000), VOCAB (default 40000), GENS (7 or 28).
"""
import os, re, json, math, glob, time
import numpy as np
from collections import Counter, defaultdict

# ===================== KAGGLE PARAMETERS (edit these) =====================
os.environ.setdefault("MAXPAIRS", "400000")   # number of Wikipedia article intros to load
os.environ.setdefault("VOCAB",    "60000")    # vocabulary size (bigger = bigger model)
os.environ.setdefault("GENS",     "28")       # so(8) transport generators: 28 (full) or 7 (fast)
# =========================================================================

def unit(v): return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

# ----------------------------------------------------------------------------- octonions
def _qmul(x, y):
    a1, b1, c1, d1 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    a2, b2, c2, d2 = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
    return np.stack([a1*a2 - b1*b2 - c1*c2 - d1*d2, a1*b2 + b1*a2 + c1*d2 - d1*c2,
                     a1*c2 - b1*d2 + c1*a2 + d1*b2, a1*d2 + b1*c2 - c1*b2 + d1*a2], -1)
def octo_mul(a, b):
    p, q = a[..., :4], a[..., 4:]; r, s = b[..., :4], b[..., 4:]
    cs = s.copy(); cs[..., 1:] *= -1; cr = r.copy(); cr[..., 1:] *= -1
    return np.concatenate([_qmul(p, r) - _qmul(cs, q), _qmul(s, p) + _qmul(q, cr)], -1)

_E = np.eye(8)
FANO_GEN = [np.stack([octo_mul(_E[g], _E[k]) for k in range(8)], axis=1) for g in range(8)]
SO8_GEN = np.stack([FANO_GEN[i] @ FANO_GEN[j] for i in range(8) for j in range(i + 1, 8)])  # 28
GEN_IDX = list(range(int(os.environ.get("GENS", 28)) if int(os.environ.get("GENS", 28)) in (7, 28) else 28))

def rotate(X, g, th): return np.cos(th) * X + np.sin(th) * (X @ SO8_GEN[g].T)
def best_move(X, T):
    A = float(np.sum(X * T)); best = (GEN_IDX[0], 0.0, -1e18)
    for g in GEN_IDX:
        B = float(np.sum((X @ SO8_GEN[g].T) * T)); th = np.arctan2(B, A)
        v = A * np.cos(th) + B * np.sin(th)
        if v > best[2]: best = (g, th, v)
    return best[0], best[1]
class FanoTransport:
    def __init__(self): self.path = []
    def fit(self, X, T, steps=10):
        cur = X.copy()
        for _ in range(steps): g, th = best_move(cur, T); cur = rotate(cur, g, th); self.path.append((g, th))
        return self
    def __call__(self, X):
        for g, th in self.path: X = rotate(X, g, th)
        return X
def slots(E): return unit(E.reshape(E.shape[0], E.shape[1] // 8, 8))
def fit_slot(Xs, Ts, steps=10): return [FanoTransport().fit(Xs[:, k], Ts[:, k], steps) for k in range(Xs.shape[1])]
def apply_slot(F, Xs): return np.stack([F[k](Xs[:, k]) for k in range(len(F))], 1)

# ----------------------------------------------------------------------- PMI-SVD embeddings
def build_embeddings(words, vocab_size, dim=96, window=5, shift=5.0, verbose=True):
    vc = Counter(words); vocab = [w for w, _ in vc.most_common(vocab_size)]
    wi = {w: i for i, w in enumerate(vocab)}; W = len(vocab)
    ids = np.array([wi[w] for w in words if w in wi], dtype=np.int64)
    if verbose: print("  corpus %d words -> %d in-vocab, vocab %d" % (len(words), len(ids), W), flush=True)
    from scipy.sparse import coo_matrix, csr_matrix
    from scipy.sparse.linalg import svds
    C = csr_matrix((W, W), dtype=np.float32)
    for d in range(1, window + 1):
        a, b = ids[:-d], ids[d:]; data = np.full(len(a), np.float32(1.0 / d), dtype=np.float32)
        Cd = coo_matrix((data, (a, b)), shape=(W, W)).tocsr(); C = C + Cd + Cd.T
    tot = C.sum(); Pa = np.asarray(C.sum(1)).ravel() / tot; Cx = C.tocoo()
    pmi = np.log(Cx.data / tot / (Pa[Cx.row] * Pa[Cx.col] + 1e-30) + 1e-12) - np.log(shift)
    keep = pmi > 0
    PPMI = coo_matrix((pmi[keep], (Cx.row[keep], Cx.col[keep])), shape=(W, W)).tocsr()
    if W <= dim + 1:
        Ud, Sd, _ = np.linalg.svd(PPMI.toarray(), full_matrices=False); U, S = Ud[:, :dim], Sd[:dim]
    else:
        U, S, _ = svds(PPMI, k=min(dim, W - 1)); o = np.argsort(S)[::-1]; U, S = U[:, o], S[o]
    if U.shape[1] < dim: U = np.pad(U, ((0, 0), (0, dim - U.shape[1]))); S = np.pad(S, (0, dim - len(S)))
    return vocab, wi, unit(U * np.sqrt(S))

def toks(text, wi): return [wi[w] for w in re.findall(r"[a-z0-9']+", text.lower()) if w in wi]

# ------------------------------------------------------------------------------------ bot
class OctoBot:
    _STOP = set("a an the of to in on at for and or but is are was were be been being this that these those it its he she they them his her their what who whom whose when where why how which do does did done can will would should could may might must as with from by about into over under after before than then there here we you i me my your our not no nor yes if so up out off above below again once also more most very just only own same s t".split())
    _ASPECT = set("treated treat treatment prevent prevention cure cured manage diagnosed cause causes caused discover discovered invented born died founded".split())
    _REQUEST = set("should advice help anything what do tell give explain about".split())
    _SEGSTOP = set("im you're dont cant really very much lately always think feel getting going they them their your his her our this that these those coming back been have having from with about who when where which".split())

    def fit(self, pairs, vocab_size=40000, dim=96, verbose=True):
        t0 = time.time(); self.pairs = pairs                              # pairs = (TITLE, answer-intro)
        search = [t + " " + a for t, a in pairs]                          # match against title + intro
        words = re.findall(r"[a-z0-9']+", (" \n ".join(search)).lower())
        self.vocab, self.wi, self.emb = build_embeddings(words, vocab_size, dim, verbose=verbose)
        self.octo = unit(self.emb[:, 1:9])
        df = Counter()
        for s in search:
            for t in set(toks(s, self.wi)): df[t] += 1
        N = len(pairs); self.idf = np.ones(len(self.vocab))
        for t, c in df.items(): self.idf[t] = math.log((N + 1) / (c + 1)) + 1.0
        self.EPr = np.array([self._vec(s) for s in search]); self.EPru = unit(self.EPr)
        EAr = np.array([self._vec(a) for _, a in pairs])
        self.Xtr, self.Ttr = slots(self.EPr), slots(EAr)
        post = defaultdict(list); bpost = defaultdict(list); tpost = defaultdict(list)
        self.tidf = np.zeros(N)                                            # sum of title-token idf per article
        for i, (title, _) in enumerate(pairs):
            tk = toks(search[i], self.wi)
            for t in set(tk):
                if self.idf[t] > 1.0: post[t].append(i)
            for a, b in zip(tk, tk[1:]): bpost[(a, b)].append(i)
            ttok = set(toks(title, self.wi))
            for t in ttok: tpost[t].append(i)                             # TITLE tokens (for title boost)
            self.tidf[i] = sum(self.idf[t] for t in ttok)
        self.post = {t: np.array(v) for t, v in post.items()}
        self.bpost = {g: np.array(v) for g, v in bpost.items()}
        self.tpost = {t: np.array(v) for t, v in tpost.items()}
        if verbose: print("  bot ready in %.0fs (pairs=%d vocab=%d gens=%d)" % (time.time()-t0, N, len(self.vocab), len(GEN_IDX)), flush=True)
        return self

    def _vec(self, txt):
        t = toks(txt, self.wi)
        if not t: return np.zeros(self.emb.shape[1])
        return unit((self.emb[t] * self.idf[t][:, None]).sum(0))

    def _match(self, q, k=40, lex=0.6, wf=3.0, wt=5.0):
        pe = self._vec(q); d = self.EPru @ pe; tk = toks(q, self.wi)
        qt = [t for t in set(tk) if self.idf[t] > 1.0 and self.vocab[t] not in self._STOP]   # content only
        if qt:
            L = np.zeros(len(self.pairs)); Tb = np.zeros(len(self.pairs)); tot = 0.0
            for t in qt:
                w = self.idf[t]; tot += w
                p = self.post.get(t)
                if p is not None: L[p] += w
                tp = self.tpost.get(t)
                if tp is not None: Tb[tp] += w                                     # title-token hit
            # title boost by COVERAGE: Tb/tidf is 1.0 when the query covers the whole title
            # (exact entity, e.g. 'Albert Einstein') and small for 'Albert Einstein College ...'
            d = d + lex * (L / (tot + 1e-9)) + wt * (Tb / (self.tidf + 1e-9))
        bg = [(a, b) for a, b in zip(tk, tk[1:])                                              # at least one content word
              if self.vocab[a] not in self._STOP or self.vocab[b] not in self._STOP]
        if wf and bg:
            B = np.zeros(len(self.pairs))
            for g in bg:
                p = self.bpost.get(g)
                if p is not None: B[p] += 1.0
            d = d + wf * (B / len(bg))
        return pe, np.argsort(-d)[:k]

    def _mmr(self, cv, score, m, lam=0.7):
        chosen = []
        while len(chosen) < m and len(chosen) < len(cv):
            best, bv = -1, -1e18
            for i in range(len(cv)):
                if i in chosen: continue
                red = max((cv[i] @ cv[j] for j in chosen), default=0.0)
                v = lam * score[i] - (1 - lam) * red
                if v > bv: bv, best = v, i
            chosen.append(best)
        return sorted(chosen, key=lambda i: -score[i])

    def answer(self, q, k=40, m=3, with_match=False):
        pe, nn = self._match(q, k)
        sents0 = [s.strip() for s in re.split(r"(?<=[.!?])\s+", self.pairs[int(nn[0])][1]) if s.strip()]
        lead = next((s for s in sents0 if len(s.split()) >= 5 and re.search(r"\b(is|was|are|were|refers)\b", s.lower())),
                    sents0[0] if sents0 else self.pairs[int(nn[0])][1])   # first definitional sentence (skip captions)
        seen, pool = {lead}, []                                            # additional context from the neighbourhood
        for j in nn:
            for s in re.split(r"(?<=[.!?])\s+", self.pairs[int(j)][1]):
                s = s.strip()
                if len(s.split()) >= 3 and s not in seen: seen.add(s); pool.append(s)
        extras = []
        if pool and m > 1:
            F = fit_slot(self.Xtr[nn], self.Ttr[nn], 10)
            g = unit(apply_slot(F, slots(pe[None]))[0].reshape(-1))        # SO(8) answer-region anchor
            cv = unit(np.array([self._vec(s) for s in pool])); qs = cv @ pe; cen = cv.mean(0)
            keep = qs >= 0.55 * qs.max()
            if keep.any(): ix = np.where(keep)[0]; pool = [pool[i] for i in ix]; cv, qs = cv[keep], qs[keep]
            idx = self._mmr(cv, 0.6 * qs + 0.25 * (cv @ g) + 0.15 * (cv @ cen), min(m - 1, len(pool)))
            extras = [pool[i] for i in idx]
        ans = (lead + " " + " ".join(extras)).strip()
        return (ans, self.pairs[int(nn[0])][0]) if with_match else ans

    def _segments(self, prompt):
        segs = []
        for p in re.split(r"\band\b|\balso\b|\bplus\b|\bbut\b|\bas well as\b|[,;.?]", prompt.lower()):
            ct = sorted((t for t in dict.fromkeys(toks(p, self.wi))
                         if self.idf[t] > 1.5 and len(self.vocab[t]) >= 4 and self.vocab[t] not in self._SEGSTOP),
                        key=lambda t: -self.idf[t])
            if ct: segs.append((p.strip(), ct))
        return segs

    def respond(self, prompt, m=3):
        segs = self._segments(prompt)
        if len(segs) <= 1: return self.answer(prompt, m=m)
        ent = [t for _, ct in segs for t in ct if self.vocab[t] not in self._ASPECT and self.vocab[t] not in self._REQUEST]
        gph = " ".join(self.vocab[t] for t in sorted(set(ent), key=lambda t: -self.idf[t])[:2])
        out, used = [], set()
        for text, ct in segs:
            words = [self.vocab[t] for t in ct]
            if all(w in self._REQUEST for w in words): continue
            query = gph if (gph and all(w in self._ASPECT for w in words)) else text   # aspect clause -> entity only
            ans, match = self.answer(query, m=m, with_match=True)
            if match in used: continue
            used.add(match)
            out.append("[%s] %s" % (" / ".join(words[:2]), ans))
        return "\n".join(out)

# ------------------------------------------------------------------ Wikidata data loading
LABELKEYS = ("label", "name", "title", "entity", "term", "question", "aliases", "labels")
ANSWKEYS = ("description", "desc", "abstract", "text", "answer", "summary", "definition", "content", "descriptions")

def _pick(d, keys):
    for k in d:
        if k.lower() in keys:
            v = d[k]
            if isinstance(v, dict): v = v.get("en", v.get("value", "")) if v else ""
            if isinstance(v, dict): v = v.get("value", "")
            if isinstance(v, list): v = " ".join(str(x.get("value", x) if isinstance(x, dict) else x) for x in v[:5])
            if v: return str(v)
    return ""

def _wikidata_entity(d):
    """Parse a raw Wikidata dump entity -> (search_text, answer_text)."""
    lab = (((d.get("labels") or {}).get("en") or {}).get("value", "")) or d.get("label", "")
    desc = (((d.get("descriptions") or {}).get("en") or {}).get("value", "")) or d.get("description", "")
    al = ((d.get("aliases") or {}).get("en") or [])
    alias = " ".join(a.get("value", "") for a in al[:5]) if isinstance(al, list) else ""
    if not lab or not desc: return None
    return (lab, (lab + " is " + desc + ".").strip())          # (title, answer)

# ---- raw text (WikiText / Wikipedia): title -> intro paragraph, or paragraph retrieval ----
def _clean(t):
    t = str(t)
    t = re.sub(r"\{\{[^{}]*\}\}", " ", t)                       # {{templates}}
    t = re.sub(r"<ref[^>]*>.*?</ref>", " ", t, flags=re.S)      # <ref>...</ref>
    t = re.sub(r"<[^>]+>", " ", t)                              # html tags
    t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", t)     # [[link|text]] -> text
    t = t.replace("'''", "").replace("''", "")                  # bold / italic
    t = re.sub(r"=+", " ", t).replace("&nbsp;", " ").replace("&amp;", "&")
    return t.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".").replace(" @ ", " ")

def _intro(text, n=3, maxwords=90):
    """First few sentences of an article/abstract -- the definition, not the whole article."""
    t = _clean(re.sub(r"\s+", " ", str(text))).strip()
    out = " ".join(re.split(r"(?<=[.!?])\s+", t)[:n]).strip()
    return " ".join(out.split()[:maxwords])

def _mkpair(label, text):
    """(label, article/abstract text) -> (TITLE, answer-intro), dropping disambiguation pages."""
    label = re.sub(r"\s+", " ", _clean(label)).strip()
    ai = _intro(text); low = ai.lower()
    if not label or not ai or len(ai.split()) < 4: return None
    if "may refer to" in low or "may also refer" in low or "disambiguation" in low: return None
    return (label, ai)

def parse_wikitext(text, maxpairs):
    # WikiText: level-1 articles ' = Title = ', sections ' = = X = = '.
    # Build (title, intro-paragraph) pairs from each article's first paragraph.
    text = _clean(text)
    hs = list(re.compile(r"(?m)^ ?= ([^=\n]+?) =\s*$").finditer(text))
    pairs = []
    for i, h in enumerate(hs):
        title = h.group(1).strip()
        body = text[h.end():(hs[i + 1].start() if i + 1 < len(hs) else len(text))]
        body = re.split(r"(?m)^ ?= = ", body)[0]                      # cut at first section
        intro = " ".join(re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", body).strip())[:3]).strip()
        if title and 1 <= len(title.split()) <= 8 and len(intro.split()) >= 6:
            pairs.append((title, intro))                       # (title, answer)
        if len(pairs) >= maxpairs: break
    return pairs

def parse_paragraphs(text, maxpairs):
    text = _clean(text); pairs = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if len(para.split()) >= 8 and not para.startswith("="):
            pairs.append((" ".join(para.split()[:6]), para))   # (pseudo-title, paragraph)
        if len(pairs) >= maxpairs: break
    return pairs

JUNK = ("vocab", "merges", "tokenizer", "config", "special_tokens", "added_tokens",
        "readme", "license", "sample_text", "gitattributes", "metadata", "index")
def _is_junk(path):
    b = os.path.basename(path).lower()
    return any(j in b for j in JUNK)

def _looks_like_prose(t):
    s = t[:8000]
    if any(x in s for x in ("[PAD]", "[unused", "[CLS]", "[SEP]", "[MASK]")): return False
    lines = [l for l in s.split("\n") if l.strip()][:60]
    return bool(lines) and sum(len(l.split()) for l in lines) / len(lines) >= 4.0

def read_raw_text(f, cap=150_000_000):
    try:
        if f.lower().endswith((".txt", ".tokens", ".raw")):
            return open(f, encoding="utf-8", errors="ignore").read(cap)
        if f.lower().endswith(".parquet"):
            import pandas as pd; df = pd.read_parquet(f)
            col = next((c for c in df.columns if c.lower() in ("text", "content", "page", "article", "wikitext", "sentence", "paragraph")), None)
            if col is None: return None
            return "\n".join(df[col].astype(str).tolist())[:cap]
        if f.lower().endswith((".json", ".jsonl", ".ndjson")):
            out = []
            for line in open(f, encoding="utf-8", errors="ignore"):
                try: d = json.loads(line)
                except Exception: continue
                if isinstance(d, dict):
                    t = d.get("text") or d.get("content") or ""
                    if t: out.append(str(t))
                if sum(len(x) for x in out) > cap: break
            return "\n".join(out) if out else None
    except Exception as e:
        print("  raw-read skip %s (%s)" % (os.path.basename(f), e), flush=True)
    return None

def load_sqlite(f, maxpairs):
    """Stream a SQLite Wikipedia DB (e.g. enwiki: articles(article_title, section_text, ...)).
    Takes the FIRST section per article (the lead/intro) -> (title, intro). Never loads the
    whole 20GB file: it streams rows with a cursor and stops at maxpairs."""
    import sqlite3
    con = sqlite3.connect("file:%s?mode=ro" % f, uri=True); cur = con.cursor()
    pairs = []
    try:
        tabs = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for tab in tabs:
            cols = [r[1] for r in cur.execute("PRAGMA table_info(%s)" % tab)]
            tl = next((c for c in cols if "title" in c.lower() or c.lower() in ("name", "entity")), None)
            tx = next((c for c in cols if "text" in c.lower() or c.lower() in ("content", "abstract", "section_text")), None)
            if not tl or not tx: continue
            print("  sqlite %s.%s: title=%s text=%s" % (os.path.basename(f), tab, tl, tx), flush=True)
            seen = set()
            for title, text in cur.execute("SELECT %s, %s FROM %s" % (tl, tx, tab)):
                if not title or title in seen: continue          # first section per article = lead
                seen.add(title); pr = _mkpair(title, text)
                if pr: pairs.append(pr)
                if len(pairs) >= maxpairs: break
            if pairs: break
    except Exception as e:
        print("  sqlite skip %s (%s)" % (os.path.basename(f), e), flush=True)
    con.close(); return pairs

def load_data(maxpairs=200000, datadir="/kaggle/input"):
    allf = [f for f in glob.glob(os.path.join(datadir, "**", "*"), recursive=True) if os.path.isfile(f)]
    files = [f for f in allf if not _is_junk(f)]
    print("found %d files (%d after dropping vocab/config/tokenizer junk)" % (len(allf), len(files)), flush=True)
    pairs = []
    # 0) SQLite Wikipedia DB (streamed, stops at maxpairs -- never loads the whole file)
    for f in files:
        if f.lower().endswith((".db", ".sqlite", ".sqlite3")):
            print("  SQLite DB: %s (streaming first %d article intros)" % (os.path.basename(f), maxpairs), flush=True)
            pairs = load_sqlite(f, maxpairs)
            if pairs: break
    # 1) STRUCTURED: csv/tsv/parquet with label+answer columns, or json/jsonl Wikidata entities
    for f in (sorted(files, key=os.path.getsize, reverse=True) if not pairs else []):
        ext = f.lower().rsplit(".", 1)[-1]
        try:
            if ext in ("csv", "tsv", "parquet"):
                import pandas as pd
                df = pd.read_parquet(f) if ext == "parquet" else pd.read_csv(f, sep="\t" if ext == "tsv" else ",", on_bad_lines="skip")
                cols = {c.lower(): c for c in df.columns}
                lc = next((cols[c] for c in cols if c in LABELKEYS), None)
                ac = next((cols[c] for c in cols if c in ANSWKEYS and cols[c] != cols.get("text")), None) or next((cols[c] for c in cols if c in ANSWKEYS), None)
                if lc and ac and lc != ac:
                    print("  table %s: search=%s answer=%s" % (os.path.basename(f), lc, ac), flush=True)
                    for s, a in zip(df[lc].astype(str), df[ac].astype(str)):
                        pr = _mkpair(s, a)
                        if pr:
                            pairs.append(pr)
                            if len(pairs) >= maxpairs: break
            elif ext in ("json", "jsonl", "ndjson"):
                with open(f, encoding="utf-8", errors="ignore") as fh:
                    head = fh.read(1); fh.seek(0)
                    if head == "[":
                        data = json.load(fh)
                        it = data if isinstance(data, list) else (data.get("rows", []) if isinstance(data, dict) else [])
                    else:
                        it = fh                                           # JSON lines
                    for line in it:
                        try: d = line if isinstance(line, dict) else json.loads(line)
                        except Exception: continue
                        if not isinstance(d, dict): continue
                        pr = _wikidata_entity(d)
                        if pr is None:
                            s, a = _pick(d, LABELKEYS), _pick(d, ANSWKEYS)
                            pr = _mkpair(s, a) if (s and a) else None
                        if pr: pairs.append(pr)
                        if len(pairs) >= maxpairs: break
        except Exception as e:
            print("  skip table/json %s (%s)" % (os.path.basename(f), e), flush=True)
        if len(pairs) >= maxpairs: break
    # 2) RAW TEXT fallback (WikiText / Wikipedia): .txt/.tokens/.raw + parquet/json 'text' column,
    #    skipping vocab/config junk and any source that doesn't look like prose.
    if len(pairs) < 200:
        print("  structured parse -> %d pairs; reading RAW TEXT..." % len(pairs), flush=True)
        raw = ""
        for f in sorted(files, key=os.path.getsize, reverse=True):
            if f.lower().rsplit(".", 1)[-1] not in ("txt", "tokens", "raw", "parquet", "json", "jsonl", "ndjson"): continue
            t = read_raw_text(f)
            if not t or len(t) < 500 or not _looks_like_prose(t):
                print("  - skip %s (not prose / empty)" % os.path.basename(f), flush=True); continue
            raw += "\n" + t
            print("  + %s (%.0f MB total)" % (os.path.basename(f), len(raw) / 1e6), flush=True)
            if len(raw) > 150_000_000: break
        if raw:
            pairs = parse_wikitext(raw, maxpairs)
            if len(pairs) < 200:
                print("  few '= Title =' articles -> paragraph retrieval mode", flush=True)
                pairs = parse_paragraphs(raw, maxpairs)
    seen, out = set(), []
    for s, a in pairs:
        if s and a and s not in seen: seen.add(s); out.append((s, a))
    if out:
        print("  -> %d pairs. sample: PROMPT=%r ANSWER=%r" % (len(out), out[0][0][:70], out[0][1][:70]), flush=True)
    return out[:maxpairs]

DEMO = [("Paris", "Paris is the capital and most populous city of France."),
        ("Albert Einstein", "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity."),
        ("Photosynthesis", "Photosynthesis is the process by which plants convert light energy into chemical energy."),
        ("Black hole", "A black hole is a region of spacetime where gravity is so strong that nothing can escape."),
        ("DNA", "DNA is the molecule that carries the genetic instructions for life."),
        ("Mona Lisa", "The Mona Lisa is a portrait painting by Leonardo da Vinci.")]

PROMPTS = [
    "what is the capital of france?",
    "who was albert einstein?",
    "what is photosynthesis?",
    "what is a black hole?",
    "who painted the mona lisa?",
    "what is dna?",
    "who was isaac newton, what did he discover, and when did he live?",
    "what is the speed of light and why is it important?",
]

if __name__ == "__main__":
    MAXPAIRS = int(os.environ.get("MAXPAIRS", 200000)); VOCAB = int(os.environ.get("VOCAB", 40000))
    t0 = time.time()
    pairs = load_data(MAXPAIRS)
    if len(pairs) < 50:
        print("No usable dataset found under /kaggle/input -- running on a tiny built-in DEMO.", flush=True)
        pairs = DEMO
    print("loaded %d pairs in %.0fs" % (len(pairs), time.time()-t0), flush=True)
    import random
    sample = random.sample(pairs, min(15, len(pairs)))
    print("SAMPLE ENTITIES in this dataset (ask about these!):", flush=True)
    for p, _ in sample:
        print("   - " + " ".join(p.split()[:6]), flush=True)
    bot = OctoBot().fit(pairs, vocab_size=VOCAB)
    print("\n" + "=" * 70 + "\nANSWERS\n" + "=" * 70, flush=True)
    for q in PROMPTS:
        print("\nPROMPT  : " + q, flush=True)
        print("RESPONSE: " + bot.respond(q), flush=True)
