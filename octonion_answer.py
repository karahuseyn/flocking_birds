"""Octonion retrieval-augmented answerer -- long answers to serious prompts (step 8).

Our octonion stack is strong at *recall*, not free generation, so a ChatGPT-shaped
*long* answer is built the honest way: retrieve real, authoritative human-written
sentences and compose them.  Three techniques, all on the octonion base:

  - effective search : every knowledge sentence is an octonion HDC hypervector
    (343 = 7**3 octonions, IDF-weighted word bundle); a bit-sampling LSH index
    (octonion_lm.LSHIndex) finds the relevant few fast.
  - symbolic plan    : decompose the question into aspects (symptoms / causes /
    diagnosis / treatment / prevention), search each aspect separately, and lay the
    answer out under those headings -- a symbolic structure over neural-free recall.
  - MMR composition  : within an aspect, pick sentences that are relevant *and*
    mutually diverse, so the answer doesn't repeat itself.

Knowledge base: MedQuAD (~16k authoritative NIH/cancer.gov Q&A), split to sentences.
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
ASPECTS = [
    ("Overview",        []),
    ("Symptoms",        ["symptom", "symptoms", "sign", "signs"]),
    ("Causes & risks",  ["cause", "causes", "caused", "causing", "risk", "risks"]),
    ("Diagnosis",       ["diagnosis", "diagnosed", "diagnose", "test", "tests", "screening", "detect"]),
    ("Treatment",       ["treatment", "treat", "treated", "treating", "therapy", "therapies",
                         "medication", "medications", "medicine", "drug", "drugs", "manage", "managed"]),
    ("Prevention",      ["prevent", "prevention", "prevented", "avoid", "reduce"]),
]

def tokenize(s):
    return [w for w in TOK.findall(s.lower()) if w not in STOP and (len(w) > 1 or w.isdigit())]

def load_medquad():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medquadQAs.json")
    if not os.path.exists(path):
        from octonion_chat import _build_medquad; _build_medquad(path)
    return json.load(open(path, encoding="utf-8"))

def good_sentence(s):
    if not (30 <= len(s) <= 280) or s.endswith("?"):
        return False
    if "<" in s or "http" in s or ":" in s[:18]:
        return False
    return sum(c.isalpha() for c in s) > 0.6 * len(s)

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

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "answerer_cache.npz")

class OctonionAnswerer:
    def fit(self, qa):
        if os.path.exists(CACHE):
            d = np.load(CACHE, allow_pickle=True)
            self.sents = list(d["sents"]); vocab = list(d["vocab"])
            idf = d["idf"]; self.bank = d["bank"]; self.stoks = list(d["stoks"])
        else:
            sents, seen = [], set()
            for r in qa:
                for m in SENT.finditer(r["answer"]):
                    s = " ".join(m.group().split())
                    if good_sentence(s) and s.lower()[:90] not in seen:
                        seen.add(s.lower()[:90]); sents.append(s)
            self.sents = sents
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
                     idf=idf, bank=self.bank, stoks=np.array(self.stoks, dtype=object))
        self.enc = Encoder(vocab, idf)
        print(f"  knowledge base: {len(self.sents):,} authoritative sentences; building LSH...")
        self.lsh = LSHIndex(self.bank, self.enc.D, bits=22, tables=8)
        return self

    def _anchors(self, query, n=2):
        allkw = set(k for _, kw in ASPECTS for k in kw)
        toks = [t for t in tokenize(query) if t not in allkw and t in self.enc.stoi]
        toks = sorted(set(toks), key=lambda t: -self.enc.idf[self.enc.stoi[t]])
        return toks[:n]

    def _search(self, query, anchors, k, pool=150, mmr=0.35):
        q = self.enc.pack(self.enc.encode(query))
        cand = self.lsh.query(q)
        if len(cand) < pool:
            cand = np.arange(len(self.sents))
        ham = np.bitwise_count(self.bank[cand] ^ q).sum(1)
        pool_idx = cand[ham.argsort()[:pool]]
        if anchors:                                              # symbolic on-topic constraint
            keep = [i for i in pool_idx if self.stoks[i] & set(anchors)]
            if len(keep) >= k:
                pool_idx = np.array(keep)
        rel = {int(i): self.enc.D - int(np.bitwise_count(self.bank[i] ^ q).sum()) for i in pool_idx}
        chosen, remaining = [], list(pool_idx)
        while remaining and len(chosen) < k:
            best, best_s = None, -1e18
            for c in remaining:
                div = max((self.enc.D - int(np.bitwise_count(self.bank[c] ^ self.bank[ch]).sum())
                           for ch in chosen), default=0)
                s = rel[int(c)] - mmr * div
                if s > best_s:
                    best_s, best = s, c
            chosen.append(best); remaining.remove(best)
        return [int(c) for c in chosen]

    def answer(self, query):
        ql = query.lower()
        present = [a for a, kw in ASPECTS if kw and any(k in ql for k in kw)]
        if not present:
            present = ["Overview", "Symptoms", "Causes & risks", "Treatment"]
        anchors = self._anchors(query)
        out = [f"*(topic: {', '.join(anchors)})*"] if anchors else []
        for asp in present:
            kw = dict(ASPECTS)[asp]
            lines = [self.sents[i] for i in self._search(query + " " + " ".join(kw), anchors, k=3)]
            if lines:
                out.append(f"## {asp}\n" + " ".join(lines))
        return "\n\n".join(out)

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
