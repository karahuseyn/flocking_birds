"""Gradient-free semantic generation: PMI-SVD meaning + n-gram fluency + manifold flow.

This is the root-level attempt at what the project was really after: smooth,
prompt-faithful generation, gradient-free and linear-time -- and crucially built on
*semantic* coordinates, not the random codes the rest of the repo leaned on.

Three gradient-free ingredients, no backprop anywhere:

  1. MEANING (Levy-Goldberg 2014): word2vec is provably an implicit factorisation of
     the shifted-PMI matrix.  So we get genuine semantic embeddings with no gradients:
     count co-occurrences -> shifted positive PMI -> truncated SVD.  Unlike random HDC
     codes, these have real geometry ("king" clusters with Edward/Warwick/Henry).

  2. FLUENCY: a trigram transition table (gradient-free counts) is the grammatical
     backbone -- it proposes locally well-formed continuations.

  3. COHERENCE / SMOOTHNESS: a context vector flows on the unit sphere
     (c <- 0.85 c + 0.15 emb[next]); candidate next-words from the n-gram are
     re-ranked by semantic alignment with that flowing context.  This is the "smooth"
     the brief asked for -- a continuous trajectory in meaning space.

Honest scope: markedly more fluent and prompt-faithful than the char-level LM
("the king of my son should be the great king ... which Warwick says is right, 'tis
the lord Hastings who attended him"), but NOT GPT-level -- long-range coherence is
still missing, and the fluency backbone is an n-gram, not the octonion algebra.  It is
the first result here that combines real semantic geometry with a flowing context;
the octonion/Fano contribution proper would replace the n-gram backbone with the
bind/unbind induction memory (octonion_induction.py), which is the next root step.
"""
import re, sys
import numpy as np
from collections import Counter, defaultdict

def build(text, vocab_size=4000, dim=80, window=5, shift=5.0):
    words = re.findall(r"[a-z']+", text.lower())
    vc = Counter(words)
    vocab = [w for w, _ in vc.most_common(vocab_size)]
    wi = {w: i for i, w in enumerate(vocab)}; W = len(vocab)
    ids = [wi[w] for w in words if w in wi]
    # co-occurrence -> shifted PPMI -> SVD  (gradient-free semantic embedding)
    C = np.zeros((W, W))
    for t in range(len(ids)):
        for d in range(1, window + 1):
            if t + d < len(ids):
                a, b = ids[t], ids[t + d]; C[a, b] += 1.0 / d; C[b, a] += 1.0 / d
    tot = C.sum(); Pa = C.sum(1) / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        PMI = np.log((C / tot) / np.outer(Pa, Pa) + 1e-12)
    PPMI = np.maximum(PMI - np.log(shift), 0.0)
    U, S, _ = np.linalg.svd(PPMI, full_matrices=False)
    emb = U[:, :dim] * np.sqrt(S[:dim])
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    # n-gram fluency tables (gradient-free counts)
    tri = defaultdict(Counter); bi = defaultdict(Counter)
    for t in range(len(ids) - 1):
        bi[ids[t]][ids[t + 1]] += 1
    for t in range(len(ids) - 2):
        tri[(ids[t], ids[t + 1])][ids[t + 2]] += 1
    return dict(vocab=vocab, wi=wi, W=W, emb=emb, tri=tri, bi=bi)

def neighbors(M, word, k=6):
    wi, emb, vocab = M["wi"], M["emb"], M["vocab"]
    if word not in wi:
        return []
    sims = emb @ emb[wi[word]]
    return [vocab[i] for i in sims.argsort()[::-1][1:k + 1]]

def generate(M, seed, n=28, temp=0.5, sem_weight=1.2, flow=0.15, rng_seed=1):
    vocab, wi, W, emb = M["vocab"], M["wi"], M["W"], M["emb"]
    tri, bi = M["tri"], M["bi"]
    rng = np.random.default_rng(rng_seed)
    out = [wi[w] for w in seed.lower().split() if w in wi] or [int(rng.integers(W))]
    cvec = emb[out].mean(0)
    for _ in range(n):
        cnt = tri.get((out[-2], out[-1])) if len(out) >= 2 else None
        if not cnt:
            cnt = bi.get(out[-1])
        if cnt:
            cand = np.array(list(cnt)); freq = np.array([cnt[c] for c in cand], float)
            sem = emb[cand] @ cvec; sem = (sem - sem.min()) / (np.ptp(sem) + 1e-9)
            score = np.log(freq) + sem_weight * sem        # fluency + semantic coherence
            p = np.exp(score / temp); p /= p.sum()
            nxt = int(rng.choice(cand, p=p))
        else:
            nxt = int(rng.integers(W))
        out.append(nxt)
        cvec = (1 - flow) * cvec + flow * emb[nxt]         # smooth flow on the manifold
    return " ".join(vocab[i] for i in out)

CORPORA = {
    "shakespeare": ("corpus_shakespeare.txt", 4000, 80, None,
                    ["the king", "my love is", "what news"], ["king", "love", "death"]),
    "biomed":      ("corpus_biomed.txt", 8000, 120, 30_000_000,
                    ["patients with", "the aim of this study", "we found that"],
                    ["patients", "treatment", "cancer"]),
    "science":     ("corpus_science.txt", 8000, 120, 25_000_000,
                    ["the results show", "patients with", "we propose a"],
                    ["patients", "algorithm", "protein"]),
}

def main():
    import base64
    which = sys.argv[1] if len(sys.argv) > 1 else "shakespeare"
    if which not in CORPORA:                                  # treat arg as a file path
        path, vs, dim, cap, prompts, probes = sys.argv[1], 6000, 100, 30_000_000, \
            ["the", "we found"], ["the"]
    else:
        path, vs, dim, cap, prompts, probes = CORPORA[which]
    text = open(path, encoding="utf-8").read(cap) if cap else open(path, encoding="utf-8").read()
    M = build(text, vocab_size=vs, dim=dim)
    print(f"built gradient-free semantic model on '{which}': {M['W']} words, "
          f"PMI-SVD(dim {dim}) + trigram + manifold flow\n")
    print("semantic neighbours (real geometry, not random codes):")
    for w in probes:
        print(f"  {w:10s}-> {', '.join(neighbors(M, w))}")
    print("\ngeneration (prompt ||| continuation):")
    outs = []
    for s in prompts:
        g = generate(M, s, n=30, temp=0.45 if which != "shakespeare" else 0.5)
        outs.append(g); print(f"  {g}")
    print("B64GEN:" + base64.b64encode("\n".join(outs).encode()).decode())

if __name__ == "__main__":
    main()
