"""Octonion retrieval QA bot  (the "rule of 7", answering questions).

A ChatGPT-shaped interface on top of the same gradient-free octonion memory:
you ask a question, it answers.  But where a transformer *generates* an answer,
this *recalls* one -- it encodes your question as a rule-of-7 octonion
hypervector, finds the seven nearest questions it has ever seen (the
7-neighbourhood), and replies with the best stored answer.  No weights, no
backprop: the knowledge base IS the model.

Hypervector follows the number 7 throughout (see octonion_symptom.py):
    octonion (8 = 1 + 7 Fano units) -> heptad (7) -> super-heptad (49)
    -> hypervector (343 = 7**3 octonions, 2744 reals).

Knowledge base: ~6k real patient/doctor medical Q&A pairs.
"""
import json, os, re, math, sys, urllib.request
import numpy as np
from octonion_lm import octo_norm

HERE = os.path.dirname(os.path.abspath(__file__))
SEVEN = 7
SLOTS = SEVEN ** 3                       # 343 octonions -> 2744-dim hypervector
BASE = ("https://raw.githubusercontent.com/LasseRegin/"
        "medical-question-answer-data/master/")
FILES = ["icliniqQAs.json", "questionDoctorQAs.json"]
TOK = re.compile(r"[a-z]+")

def tokenize(s):
    return TOK.findall(s.lower())

def load_qa():
    qa, seen = [], set()
    for f in FILES:
        path = os.path.join(HERE, f)
        if not os.path.exists(path):
            print(f"fetching medical Q&A ({f})...")
            d = urllib.request.urlopen(
                urllib.request.Request(BASE + f, headers={"User-Agent": "Mozilla/5.0"}),
                timeout=60).read()
            open(path, "wb").write(d)
        for r in json.load(open(path, encoding="utf-8")):
            q = (r.get("question") or "").strip()
            a = (r.get("answer") or "").strip()
            qt = (r.get("question_text") or "").strip()
            tags = set(t.strip().lower() for t in (r.get("tags") or []) if t.strip())
            key = q.lower()
            if q and a and key not in seen:               # drop exact-duplicate questions
                seen.add(key)
                qa.append((q, qt, a, tags))
    return qa

# --------------------------------------------------------------------------
# Octonion word encoder: IDF-weighted bag-of-words bundle (rule of 7)
# --------------------------------------------------------------------------
class WordEncoder:
    def __init__(self, vocab, idf, slots=SLOTS, seed=int("0709", 10) ^ 1916):
        rng = np.random.default_rng(seed)
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.idf = idf
        emb = rng.standard_normal((len(vocab), slots, 8))
        self.emb = (emb / octo_norm(emb)).reshape(len(vocab), slots * 8)
        self.D = slots * 8
        self.W = (self.D + 63) // 64

    def encode(self, text):
        ids = [self.stoi[t] for t in tokenize(text) if t in self.stoi]
        if not ids:
            return np.zeros(self.D)
        w = self.idf[ids][:, None]                      # rare words count more
        v = (self.emb[ids] * w).sum(0)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def pack(self, vecs):
        vecs = np.atleast_2d(vecs)
        bits = np.zeros((len(vecs), self.W * 64), dtype=np.uint8)
        bits[:, :self.D] = (vecs > 0)
        return np.packbits(bits, axis=1).view(np.uint64)

# --------------------------------------------------------------------------
# The bot: a packed memory of question hypervectors, ranked by Hamming
# --------------------------------------------------------------------------
class OctonionChat:
    def fit(self, qa):
        self.qa = qa
        # vocabulary + idf over the stored questions
        df = {}
        docs = [tokenize(q + " " + qt) for q, qt, _, _ in qa]
        for toks in docs:
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        vocab = sorted(df)
        N = len(docs)
        idf = np.array([math.log((N + 1) / (df[w] + 1)) + 1.0 for w in vocab])
        self.enc = WordEncoder(vocab, idf)
        # encode every stored question into the bipolar memory bank
        bank = np.empty((len(qa), self.enc.W), dtype=np.uint64)
        for i, (q, qt, _, _) in enumerate(qa):
            bank[i] = self.enc.pack(self.enc.encode(q + " " + qt))[0]
        self.bank = bank
        return self

    def search(self, question, k=SEVEN):
        """Indices of the k nearest stored questions, plus their similarities."""
        q = self.enc.pack(self.enc.encode(question))[0]
        ham = np.bitwise_count(self.bank ^ q).sum(1)
        order = np.argsort(ham)[:k]
        return order, 100.0 * (1.0 - ham[order] / self.enc.D)

    def ask(self, question, k=SEVEN):
        order, sim = self.search(question, k)
        return [(self.qa[i][0], self.qa[i][2], float(s)) for i, s in zip(order, sim)]


def evaluate(bot, qa, ntest=600, keep=0.6):
    """Paraphrase robustness: rephrase a question (keep ~60% of its words, in a
    shuffled order -- the way a user would ask it differently) and check whether
    the bot still recalls the exact original from the whole knowledge base."""
    rng = np.random.default_rng(0)
    r1 = r7 = n = 0
    for i in rng.permutation(len(qa))[: ntest * 3]:
        words = tokenize(qa[i][0])
        if len(words) < 5:
            continue
        m = max(3, int(len(words) * keep))
        kept = list(rng.choice(words, size=m, replace=False))
        rng.shuffle(kept)
        order, _ = bot.search(" ".join(kept), k=SEVEN)
        r1 += (order[0] == i); r7 += (i in order); n += 1
        if n >= ntest:
            break
    print(f"paraphrase robustness ({n} held-out, recall exact original Q):")
    print(f"  top-1 : {100*r1/n:5.1f}%")
    print(f"  top-7 : {100*r7/n:5.1f}%   (the 7-neighbourhood)\n")

# --------------------------------------------------------------------------
def _wrap(s, width=88, lead="    "):
    out, line = [], ""
    for w in s.split():
        if len(line) + len(w) + 1 > width:
            out.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return ("\n" + lead).join(out)

def main():
    qa = load_qa()
    print(f"knowledge base: {len(qa)} medical Q&A  "
          f"(hypervector {SLOTS} octonions = {SLOTS*8} dims)\n")
    bot = OctonionChat().fit(qa)
    if len(sys.argv) <= 1:
        evaluate(bot, qa)

    if len(sys.argv) > 1:                                # one-shot: ask from CLI
        questions = [" ".join(sys.argv[1:])]
    else:
        questions = [
            "is it ok to exercise when my knee hurts?",
            "how reliable is my hiv test result?",
            "what can I do to lower my high blood pressure?",
            "I keep getting headaches, what should I do?",
            "my child has a fever and sore throat, what should I give?",
        ]
    for question in questions:
        print("=" * 92)
        print("Q:", question)
        ranked = bot.ask(question, k=SEVEN)
        best_q, best_a, best_s = ranked[0]
        print(f"\nA (recalled, {best_s:.0f}% match to: \"{best_q}\"):")
        print("    " + _wrap(best_a[:700]))
        print("\n  nearest 7 questions in memory:")
        for r, (mq, _, s) in enumerate(ranked, 1):
            print(f"    {r}. [{s:4.1f}%] {mq[:80]}")
        print()

if __name__ == "__main__":
    main()
