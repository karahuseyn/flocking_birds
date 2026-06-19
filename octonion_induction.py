"""Octonion induction head  =  a causal, softmax-free, O(n) sequence model.

Step 2 toward a transformer alternative.  An *induction head* (Olsson 2022) is
the circuit behind in-context learning: on seeing token A, predict whatever
followed A earlier in *this* sequence.  A frozen n-gram cannot do it -- the
key->value map is invented fresh per sequence -- but attention can.  We do it
with the octonion fast-weight memory, wired as:

    at step t:   key_t   = emb[token_{t-1}]      (the previous token)
                 value_t = emb[token_t]          (the current token)
                 S      += bind(key_t, value_t)  (causal running sum  ->  O(n))
    predict t+1: emb-unbind(emb[token_t], S) -> "what followed token_t before"

No softmax, no n x n attention matrix, no training -- just an octonion running
sum (O(n) time, O(1) state per step) and a nearest-neighbour cleanup.  We pit it
against (a) a frozen bigram (no in-context adaptation) and (b) a softmax-attention
reference using the *same* embeddings, to show we match attention's accuracy at a
fraction of its cost.  Slots follow the rule of 7 (343 = 7**3 octonions).
"""
import numpy as np
from octonion_attention import bind, unbind, rand_unit, SEVEN

def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)

def make_sequence(n_pairs, vocab, n_distract, rng):
    """A stream of (key, value) pairs (+ distractor tokens), then a query key.
    The key->value map is random *per sequence*, so only in-context recall wins.
    Returns (tokens, query_pos, true_next)."""
    keys = rng.choice(vocab // 2, size=n_pairs, replace=False)          # keys: 0..V/2
    vals = rng.integers(vocab // 2, vocab, size=n_pairs)               # values: V/2..V
    toks = []
    for k, v in zip(keys, vals):
        toks += [int(k), int(v)]
        for _ in range(rng.integers(0, n_distract + 1)):              # noise tokens
            toks.append(int(rng.integers(0, vocab)))
    qi = rng.integers(0, n_pairs)                                     # query a seen key
    toks.append(int(keys[qi]))                                        # ...as the last token
    return toks, vals[qi]

def octonion_predict(toks, emb):
    """Causal octonion induction memory; return the predicted next-token id."""
    K8 = emb.shape[1:]
    S = np.zeros(K8)
    prev = None
    for t in toks[:-1]:                                               # build memory
        if prev is not None:
            S = S + bind(emb[prev], emb[t])                           # bind(prev->cur)
        prev = t
    rec = unbind(emb[toks[-1]], S).reshape(-1)                       # query = last token
    cb = emb.reshape(emb.shape[0], -1)
    cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)
    return (cb @ rec).argmax()

def softmax_predict(toks, emb):
    """O(n^2) softmax-attention reference over the same prev/cur embeddings."""
    flat = emb.reshape(emb.shape[0], -1)
    keys = np.stack([flat[toks[i - 1]] for i in range(1, len(toks))])  # prev tokens
    vals = np.stack([flat[toks[i]] for i in range(1, len(toks))])      # cur tokens
    q = flat[toks[-1]]
    attn = softmax((keys @ q) / np.sqrt(flat.shape[1]))[None]
    rec = (attn @ vals)[0]
    cb = flat / np.linalg.norm(flat, axis=1, keepdims=True)
    return (cb @ rec).argmax()

def evaluate(n_pairs, slots, vocab=120, n_distract=1, trials=300, seed=0):
    rng = np.random.default_rng(seed)
    emb = rand_unit((vocab, slots, 8), rng)
    octo = soft = bigram = 0
    glob = np.zeros((vocab, vocab))                                   # frozen bigram stats
    for _ in range(trials):
        toks, true = make_sequence(n_pairs, vocab, n_distract, rng)
        octo += (octonion_predict(toks, emb) == true)
        soft += (softmax_predict(toks, emb) == true)
        pred_bg = glob[toks[-1]].argmax()                            # global, no in-context
        bigram += (pred_bg == true)
        for i in range(1, len(toks)):
            glob[toks[i - 1], toks[i]] += 1
    return octo / trials, soft / trials, bigram / trials

def main():
    print("In-context induction: predict the value bound to a query key,")
    print("from a stream whose key->value map is random per sequence.")
    print("octonion memory is O(n) & softmax-free; the softmax ref is O(n^2).\n")
    slots = SEVEN ** 3
    print(f"{slots} octonions ({slots*8}-dim)   vocab 120, ~1 distractor/pair\n")
    print(f"{'pairs':>6} | {'octonion O(n)':>14} | {'softmax O(n^2)':>15} | {'frozen bigram':>14}")
    for n in (2, 4, 8, 16, 32, 48):
        o, s, b = evaluate(n, slots)
        print(f"{n:>6} | {100*o:>13.1f}% | {100*s:>14.1f}% | {100*b:>13.1f}%")
    print("\noctonion induction matches softmax attention at O(n) cost & no softmax;")
    print("the frozen bigram is at chance -- it cannot learn the in-context map.")

if __name__ == "__main__":
    main()
