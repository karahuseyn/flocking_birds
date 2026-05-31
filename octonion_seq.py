"""Integrated gradient-free octonion sequence model, tested on real text (step 6).

Brings steps 1-5 together into one causal, O(n), softmax-free, gradient-free model
and runs it on *real* prompts:

  - context layer (depth):  the last m chars are bound into positional roles and
    bundled with a recency gate into one octonion context hypervector;
  - induction memory (recall + gating):  a per-block fast-weight memory built
    *online over the current prompt*  W += value(x)key,  key = context, value =
    the char that actually followed -- so "what came after this context before?"
  - readout:  unbind + nearest-neighbour cleanup over the octonion char codebook.

The distinctive thing this adds over an n-gram is **in-context learning**: it can
recall a continuation it saw earlier in *this very prompt*, even a rare/novel one.
We measure that the honest way -- the induction test of Olsson et al.: feed a real
passage, then repeat it, and see whether the model predicts the repeat.  A global
bigram cannot (it only knows corpus statistics); the octonion memory can, and its
own accuracy jumps from first pass to repeat -- in-context learning on real text.
"""
import numpy as np
from octonion_lm import octo_mul, build_corpus
from octonion_attention import rand_unit

SLOTS = 343

class OctonionSeq:
    def __init__(self, V, m=14, slots=SLOTS, decay=0.75, seed=1916):
        rng = np.random.default_rng(seed)
        self.emb = rand_unit((V, slots, 8), rng)          # octonion char codebook
        self.roles = rand_unit((m, slots, 8), rng)        # positional roles (layer 1)
        self.w = decay ** np.arange(m)                    # recency gate
        self.m, self.slots, self.V = m, slots, V
        self.cb = self.emb.reshape(V, -1)
        self.cb = self.cb / np.linalg.norm(self.cb, axis=1, keepdims=True)

    def context(self, recent):
        """Bind the last m chars into roles and bundle (octonion composition)."""
        acc = np.zeros((self.slots, 8))
        for j, ch in enumerate(reversed(recent[-self.m:])):     # j=0 = most recent
            acc += self.w[j] * octo_mul(self.roles[j], self.emb[ch])
        n = np.linalg.norm(acc)
        return acc / (n if n > 1e-9 else 1.0)

    def run(self, tokens):
        """Causal pass.  Returns next-char predictions from the in-context memory."""
        W = np.zeros((self.slots, 8, 8))                  # per-block fast weights
        preds, recent = [], []
        for tok in tokens:
            if recent:
                ctx = self.context(recent)
                kn = ctx / (np.linalg.norm(ctx, axis=1, keepdims=True) + 1e-9)
                read = np.einsum("kij,kj->ki", W, kn).reshape(-1)
                preds.append(int((self.cb @ read).argmax()))
                W = W + np.einsum("ki,kj->kij", self.emb[tok], kn)   # bind ctx->char
            else:
                preds.append(-1)
            recent.append(tok)
        return preds

def global_bigram(ids, V):
    P = np.zeros((V, V))
    np.add.at(P, (ids[:-1], ids[1:]), 1.0)
    return P.argmax(1)                                    # most likely next char

def main():
    text, stoi, itos, V, ids = build_corpus("shakespeare")
    print(f"real text: tinyShakespeare, {len(ids):,} chars, vocab {V}\n")
    model = OctonionSeq(V)
    nxt = global_bigram(ids, V)                           # global stats baseline
    rng = np.random.default_rng(0)

    P, trials = 70, 150
    o1 = o2 = bg = n = 0
    rescue_hit = rescue_tot = 0
    for _ in range(trials):
        s = rng.integers(0, len(ids) - P - 5)
        passage = list(ids[s:s + P])
        seq = passage + passage                           # real passage, then repeated
        preds = model.run(seq)
        for t in range(1, P):                             # first pass (nothing to recall)
            o1 += (preds[t] == seq[t])
        for t in range(P, 2 * P):                         # repeat (induction can recall)
            oc = (preds[t] == seq[t]); bc = (nxt[seq[t - 1]] == seq[t])
            o2 += oc; bg += bc; n += 1
            if not bc:                                    # where global stats fail...
                rescue_tot += 1; rescue_hit += oc        # ...does in-context rescue it?
    print("Induction on real text (feed a passage, then repeat it):")
    print(f"  octonion in-context, FIRST pass  : {100*o1/((P-1)*trials):5.1f}%   (nothing to recall yet)")
    print(f"  octonion in-context, on REPEAT   : {100*o2/n:5.1f}%   (recalls the passage)")
    print(f"  global bigram baseline, on REPEAT: {100*bg/n:5.1f}%   (knows only corpus stats)")
    print(f"\n  in-context learning = the {100*o1/((P-1)*trials):.0f}% -> {100*o2/n:.0f}% jump when the context repeats.")
    print(f"  beyond global stats: of the chars the bigram gets WRONG, the octonion memory")
    print(f"  recalls {100*rescue_hit/rescue_tot:.1f}% correctly from context -- info no n-gram has.")

if __name__ == "__main__":
    main()
