"""Octonion retrieval-augmented answerer -- long answers to serious prompts (step 8).

Our octonion stack recalls rather than generates, so a ChatGPT-shaped *long* answer is
built by retrieving real, authoritative sentences (MedQuAD / NIH) and composing them.
Techniques, all on the octonion base:

  - effective search : every sentence is an octonion HDC hypervector (343 = 7**3
    octonions, IDF-weighted); a bit-sampling LSH index finds the relevant few fast.
  - symbolic plan    : "symptoms/causes/treatment of X" -> aspect headings; "what
    disease could this be" -> a differential, naming candidate conditions (grouping the
    evidence by the disease each MedQuAD entry is about).
  - anchor + filter  : the rarest query term must appear (no topic drift), boilerplate
    sentences are dropped, MMR keeps the picks diverse.

A digit-aware tokenizer keeps "type 2" distinct from "type 1".  No gradients.
"""
import re, sys, os, json, math
import numpy as np
from octonion_lm import LSHIndex
from octonion_attention import rand_unit

SLOTS = 343
SENT = re.compile(r"[^.!?]+[.!?]")
TOK = re.compile(r"[a-z0-9]+")
STOP = set("a an and the is it its to of in on for or as at by with from be are was were "
           "this that these those there here will would can could may might also such has "
           "have had been being which who what when where how their they them you your we our "
           "but not no if then so into out up down over than more most some any all".split())
BOILER = ("a condition is considered rare", "learn more", "this information",
          "human phenotype ontology", "(hpo)", "approximate number", "for more information",
          "the information in", "this is a rare", "the following list", "see the",
          "in this condition", "see screening", "talk to your doctor about")
ASPECTS = [
    ("Overview",        []),
    ("Symptoms",        ["symptom", "symptoms", "sign", "signs"]),
    ("Causes & risks",  ["cause", "causes", "caused", "causing", "risk", "risks"]),
    ("Diagnosis",       ["diagnosis", "diagnosed", "diagnose", "test", "tests", "screening", "detect"]),
    ("Treatment",       ["treatment", "treat", "treated", "treating", "therapy", "therapies",
                         "medication", "medications", "medicine", "drug", "drugs", "manage", "managed"]),
    ("Prevention",      ["prevent", "prevention", "prevented", "avoid", "reduce"]),
]
DIFFERENTIAL = ("what disease", "which disease", "what condition", "which condition",
                "should i suspect", "what could", "what might", "what is wrong", "could it be",
                "what causes my", "why do i", "diagnos", "what's wrong")

def tokenize(s):
    return [w for w in TOK.findall(s.lower()) if w not in STOP and (len(w) > 1 or w.isdigit())]

def good_sentence(s):
    sl = s.lower()
    if not (35 <= len(s) <= 260) or s.endswith("?"):
        return False
    if "<" in s or "http" in s or ":" in s[:18] or any(b in sl for b in BOILER):
        return False
    if sl.split()[0] in {"this", "these", "it", "that", "those", "there", "sometimes",
                         "or", "and", "but", "so", "also", "however", "thus"}:
        return False
    return sum(c.isalpha() for c in s) > 0.6 * len(s)

_FOCUS_PREFIX = re.compile(
    r"^(what (is|are)( the)?( \(are\))?|what (causes|happens in|are the (symptoms|treatments|signs|"
    r"stages|causes|complications|genetic changes) (of|for|related to))|who is at risk (for|of)|"
    r"how (to prevent|to diagnose|many people are affected by|is)|is there|do you have)\b", re.I)

def extract_focus(q):
    q = q.strip().rstrip("?").strip()
    for sep in (" of ", " for ", " related to "):
        if sep in q.lower():
            return q[q.lower().rindex(sep) + len(sep):].strip(" .()")
    return _FOCUS_PREFIX.sub("", q).strip(" .()") or q

def _load_symptom_terms():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "symptoms_train.csv")
    if not os.path.exists(p):
        return set()
    words = set()
    for h in open(p, encoding="utf-8").readline().strip().split(","):
        for w in h.lower().replace("_", " ").split():
            if len(w) > 3:
                words.add(w)
    return words

SYMPTOM_TERMS = _load_symptom_terms()

def is_symptom_word(t):
    return any(s.startswith(t[:5]) or t.startswith(s[:5]) for s in SYMPTOM_TERMS)

def load_medquad():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medquadQAs.json")
    if not os.path.exists(path):
        from octonion_chat import _build_medquad; _build_medquad(path)
    return json.load(open(path, encoding="utf-8"))

class Encoder:
    def __init__(self, vocab, idf, slots=SLOTS, seed=int("0709", 10) ^ 1916):
        rng = np.random.default_rng(seed)
        self.stoi = {w: i for i, w in enumerate(vocab)}; self.idf = idf
        emb = rng.standard_normal((len(vocab), slots, 8))
        emb /= np.sqrt((emb * emb).sum(-1, keepdims=True))
        self.emb = emb.reshape(len(vocab), slots * 8)
        self.D = slots * 8; self.W = (self.D + 63) // 64

    def encode(self, text):
        ids = [self.stoi[t] for t in tokenize(text) if t in self.stoi]
        if not ids:
            return np.zeros(self.D)
        v = (self.emb[ids] * self.idf[ids][:, None]).sum(0)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def pack(self, vec):
        bits = np.zeros(self.W * 64, dtype=np.uint8); bits[:self.D] = (vec > 0)
        return np.packbits(bits).view(np.uint64)

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "answerer_cache2.npz")

class OctonionAnswerer:
    def fit(self, qa):
        if os.path.exists(CACHE):
            d = np.load(CACHE, allow_pickle=True)
            self.sents = list(d["sents"]); vocab = list(d["vocab"]); idf = d["idf"]
            self.bank = d["bank"]; self.stoks = list(d["stoks"]); self.focus = list(d["focus"])
        else:
            sents, focus, seen = [], [], set()
            for r in qa:
                f = extract_focus(r["question"])
                for m in SENT.finditer(r["answer"]):
                    s = " ".join(m.group().split())
                    if good_sentence(s) and s.lower()[:90] not in seen:
                        seen.add(s.lower()[:90]); sents.append(s); focus.append(f)
            self.sents, self.focus = sents, focus
            docs = [tokenize(s) for s in sents]; df = {}
            for toks in docs:
                for w in set(toks):
                    df[w] = df.get(w, 0) + 1
            vocab = sorted(df); N = len(docs)
            idf = np.array([math.log((N + 1) / (df[w] + 1)) + 1.0 for w in vocab])
            enc = Encoder(vocab, idf)
            self.bank = np.stack([enc.pack(enc.encode(s)) for s in sents])
            self.stoks = [set(t) for t in docs]
            np.savez(CACHE, sents=np.array(sents, dtype=object), vocab=np.array(vocab, dtype=object),
                     idf=idf, bank=self.bank, stoks=np.array(self.stoks, dtype=object),
                     focus=np.array(focus, dtype=object))
        self.enc = Encoder(vocab, idf)
        print(f"  knowledge base: {len(self.sents):,} authoritative sentences; building LSH...")
        self.lsh = LSHIndex(self.bank, self.enc.D, bits=22, tables=8)
        return self

    def _anchors(self, query, n=2):
        allkw = set(k for _, kw in ASPECTS for k in kw) | {"disease", "condition", "suspect", "drink"}
        toks = [t for t in tokenize(query) if t not in allkw and t in self.enc.stoi]
        # prefer query words that are actual symptoms (our 132-symptom lexicon), so a
        # rare-but-irrelevant noun like "tea" can't out-rank the clinical signal "palpitation"
        sympt = [t for t in toks if is_symptom_word(t)]
        pool = sympt or toks
        pool = sorted(set(pool), key=lambda t: -self.enc.idf[self.enc.stoi[t]])
        return pool[:n]

    def _pool(self, query, anchors, pool=200):
        q = self.enc.pack(self.enc.encode(query))
        cand = self.lsh.query(q)
        if len(cand) < pool:
            cand = np.arange(len(self.sents))
        ham = np.bitwise_count(self.bank[cand] ^ q).sum(1)
        idx = cand[ham.argsort()[:pool]]
        if anchors:
            keep = [i for i in idx if self.stoks[i] & set(anchors)]
            if len(keep) >= 5:
                idx = np.array(keep)
        sims = self.enc.D - np.bitwise_count(self.bank[idx] ^ q).sum(1)
        return list(idx), sims

    def _mmr(self, idx, sims, k, mmr=0.35):
        rel = {int(idx[i]): int(sims[i]) for i in range(len(idx))}
        chosen, remaining = [], list(idx)
        while remaining and len(chosen) < k:
            best, best_s = None, -1e18
            for c in remaining:
                div = max((self.enc.D - int(np.bitwise_count(self.bank[c] ^ self.bank[ch]).sum())
                           for ch in chosen), default=0)
                sc = rel[int(c)] - mmr * div
                if sc > best_s:
                    best_s, best = sc, c
            chosen.append(best); remaining.remove(best)
        return [int(c) for c in chosen]

    def differential(self, query, k=6):
        anchors = self._anchors(query)
        idx, sims = self._pool(query, anchors, pool=300)
        best = {}
        for i, sim in zip(idx, sims):
            f = self.focus[int(i)]
            if not f or len(f) > 55 or len(f) < 3:
                continue
            if f not in best or sim > best[f][0]:
                best[f] = (int(sim), self.sents[int(i)])
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:k]
        out = [f"*(symptoms read as: {', '.join(anchors)})*",
               "Conditions that can present with these symptoms "
               "(ranked by match; not medical advice — see a clinician):"]
        for n, (dis, (sc, ev)) in enumerate(ranked, 1):
            out.append(f"{n}. **{dis}** — {ev}")
        return "\n".join(out)

    def aspect_answer(self, query):
        anchors = self._anchors(query)
        ql = query.lower()
        present = [a for a, kw in ASPECTS if kw and any(k in ql for k in kw)] \
            or ["Overview", "Symptoms", "Causes & risks", "Treatment"]
        out = [f"*(topic: {', '.join(anchors)})*"] if anchors else []
        used = set()
        for asp in present:
            kw = dict(ASPECTS)[asp]
            idx, sims = self._pool(query + " " + " ".join(kw), anchors)
            picks = [i for i in self._mmr(idx, sims, k=4) if i not in used][:3]
            used.update(picks)
            if picks:
                out.append(f"## {asp}\n" + " ".join(self.sents[i] for i in picks))
        return "\n\n".join(out)

    def answer(self, query):
        if any(t in query.lower() for t in DIFFERENTIAL):
            return self.differential(query)
        return self.aspect_answer(query)

def main():
    print("building octonion retrieval-augmented answerer (MedQuAD)...")
    bot = OctonionAnswerer().fit(load_medquad())
    prompts = sys.argv[1:] and [" ".join(sys.argv[1:])] or [
        "What are the symptoms, causes, and treatment of type 2 diabetes?"]
    for p in prompts:
        print("\n" + "=" * 92 + f"\nPROMPT: {p}\n" + "=" * 92)
        print(bot.answer(p))

if __name__ == "__main__":
    main()
