"""Integrated gradient-free octonion sequence model, tested on real text (step 6+).

Steps 1-5 wired into one causal, O(n), softmax-free, gradient-free model, now with a
capacity boost: **multiple heads**.  Each head is an independent octonion induction
memory with its own random codebook, so the crosstalk noise is independent across
heads; soft-voting their predictions cuts the noise ~sqrt(H) and lifts in-context
recall -- still O(n), still no gradients, octonions everywhere.

  context (depth) : last m chars bound into per-head roles, recency-gated, bundled
  memory (recall) : per-block fast weights W_h += value_h (x) key_h   (online, O(n))
  readout         : sum each head's similarity scores over the codebook, argmax

We test the transformer-distinctive ability -- in-context learning -- the honest way
(Olsson's induction test on real text): feed a real passage, repeat it, see what the
model predicts on the repeat.  A global bigram cannot; the octonion memory can.

CLI:  python3 octonion_seq.py "your prompt with a repeated pattern ... the pattern"
"""
import sys
import numpy as np
from octonion_lm import octo_mul, build_corpus
from octonion_attention import rand_unit

SLOTS = 343

class OctonionSeq:
    def __init__(self, V, heads=8, m=10, slots=49, decay=0.75, seed=1916):
        rng = np.random.default_rng(seed)
        self.H, self.m, self.slots, self.V = heads, m, slots, V
        self.emb = rand_unit((heads, V, slots, 8), rng)       # per-head octonion codebook
        self.roles = rand_unit((heads, m, slots, 8), rng)     # per-head positional roles
        self.w = decay ** np.arange(m)
        cb = self.emb.reshape(heads, V, -1)
        self.cb = cb / np.linalg.norm(cb, axis=2, keepdims=True)

    def _context(self, recent):                               # -> (H, slots, 8)
        acc = np.zeros((self.H, self.slots, 8))
        for j, ch in enumerate(reversed(recent[-self.m:])):   # j=0 = most recent
            acc += self.w[j] * octo_mul(self.roles[:, j], self.emb[:, ch])
        n = np.linalg.norm(acc.reshape(self.H, -1), axis=1)
        return acc / np.maximum(n, 1e-9)[:, None, None]

    def _key(self, recent):
        ctx = self._context(recent)
        return ctx / (np.linalg.norm(ctx, axis=2, keepdims=True) + 1e-9)

    def _score(self, W, recent):
        retr = np.einsum("hkij,hkj->hki", W, self._key(recent)).reshape(self.H, -1)
        return np.einsum("hvd,hd->hv", self.cb, retr).sum(0)               # vote over heads

    def run(self, tokens):
        """Causal pass; returns (preds, scores) from the in-context memory."""
        W = np.zeros((self.H, self.slots, 8, 8))
        preds, scores, recent = [], [], []
        for tok in tokens:
            if recent:
                s = self._score(W, recent)
                preds.append(int(s.argmax())); scores.append(s)
                W = W + np.einsum("hki,hkj->hkij", self.emb[:, tok], self._key(recent))
            else:
                preds.append(-1); scores.append(None)
            recent.append(tok)
        return preds, scores

    def ingest(self, tokens):
        """Build the in-context memory from a prompt; return (memory, recent)."""
        W = np.zeros((self.H, self.slots, 8, 8)); recent = []
        for tok in tokens:
            if recent:
                W = W + np.einsum("hki,hkj->hkij", self.emb[:, tok], self._key(recent))
            recent.append(tok)
        return W, recent

    def generate(self, W, recent, n):
        """Continue, recalling from the *frozen* prompt memory (no feedback drift)."""
        recent, out = list(recent), []
        for _ in range(n):
            c = int(self._score(W, recent).argmax()); out.append(c); recent.append(c)
        return out

def predict_next_word(prompt, m=3, heads=8, slots=49, decay=0.8, seed=1916):
    """Word-level in-context induction: build an octonion memory from the words of
    the prompt and recall the most likely next word.  Discrete words are near-
    orthogonal codes, so recall is far cleaner than char-level -- it even resolves
    ambiguity ('alice lives in paris . ... alice lives in' -> 'paris', not 'tokyo')."""
    words = prompt.lower().replace(".", " . ").replace(",", " , ").split()
    if len(words) < 2:
        return []
    vocab = sorted(set(words)); stoi = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
    rng = np.random.default_rng(seed)
    emb = rand_unit((heads, V, slots, 8), rng)
    roles = rand_unit((heads, m, slots, 8), rng)
    w = decay ** np.arange(m)
    cb = emb.reshape(heads, V, -1); cb = cb / np.linalg.norm(cb, axis=2, keepdims=True)
    def key(recent):
        acc = np.zeros((heads, slots, 8))
        for j, wd in enumerate(reversed(recent[-m:])):
            acc += w[j] * octo_mul(roles[:, j], emb[:, stoi[wd]])
        n = np.linalg.norm(acc.reshape(heads, -1), axis=1)
        kn = acc / np.maximum(n, 1e-9)[:, None, None]
        return kn / (np.linalg.norm(kn, axis=2, keepdims=True) + 1e-9)
    W = np.zeros((heads, slots, 8, 8)); recent = []
    for x in words:
        if recent:
            W = W + np.einsum("hki,hkj->hkij", emb[:, stoi[x]], key(recent))
        recent.append(x)
    retr = np.einsum("hkij,hkj->hki", W, key(recent)).reshape(heads, -1)
    s = np.einsum("hvd,hd->hv", cb, retr).sum(0)
    top = s.argsort()[::-1][:5]
    return [(vocab[i], float(s[i] / (s[top[0]] + 1e-9))) for i in top]

def global_bigram(ids, V):
    P = np.zeros((V, V)); np.add.at(P, (ids[:-1], ids[1:]), 1.0)
    return P.argmax(1)

def induction_test(model, ids, P=60, trials=30, seed=0):
    rng = np.random.default_rng(seed); nxt = global_bigram(ids, model.V)
    o1 = o2 = bg = n = rh = rt = 0
    for _ in range(trials):
        s = rng.integers(0, len(ids) - P - 5)
        seq = list(ids[s:s + P]) * 2
        preds, _ = model.run(seq)
        for t in range(1, P):
            o1 += (preds[t] == seq[t])
        for t in range(P, 2 * P):
            oc = (preds[t] == seq[t]); bc = (nxt[seq[t - 1]] == seq[t])
            o2 += oc; bg += bc; n += 1
            if not bc:
                rt += 1; rh += oc
    return 100*o1/((P-1)*trials), 100*o2/n, 100*bg/n, 100*rh/max(rt, 1)

def main():
    text, stoi, itos, V, ids = build_corpus("shakespeare")
    enc = lambda s: [stoi[c] for c in s if c in stoi]
    dec = lambda xs: "".join(itos[i] for i in xs if i >= 0)

    if len(sys.argv) > 1:                                     # --- test YOUR prompt ---
        prompt = " ".join(sys.argv[1:])
        print(f'prompt:  "{prompt}"\n')
        ranked = predict_next_word(prompt)
        print("most likely next word, recalled in-context from the prompt:")
        for word, sc in ranked:
            bar = "#" * int(max(sc, 0) * 28)
            print(f"   {word:<14} {sc:+.2f}  {bar}")
        print("\n(gradient-free, octonion in-context memory -- it recalls associations you")
        print(" set up in the prompt itself; give it a repeated pattern to complete.)")
        return

    print(f"real text: tinyShakespeare, {len(ids):,} chars, vocab {V}")
    print("In-context induction on real text -- capacity vs number of heads:\n")
    print(f"{'heads':>6} | {'first pass':>11} | {'on repeat':>10} | {'bigram':>8} | "
          f"{'rescue (beats n-gram)':>22}")
    for H in (1, 4, 8, 16, 32):
        f, r, b, rc = induction_test(OctonionSeq(V, heads=H), ids)
        print(f"{H:>6} | {f:>10.1f}% | {r:>9.1f}% | {b:>7.1f}% | {rc:>21.1f}%")
    print("\nmore heads -> independent crosstalk -> higher in-context recall, still O(n)")
    print("and gradient-free.  Try your own prompt:  python3 octonion_seq.py \"...\"")

if __name__ == "__main__":
    main()
