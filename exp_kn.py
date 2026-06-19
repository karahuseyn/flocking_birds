# exp_kn.py -- #3: interpolated modified Kneser-Ney higher-order n-gram backbone.
# The honest metric for a language model is held-out PERPLEXITY (KN's documented win),
# not our self-referential drift score. We compare the current "raw trigram + bigram
# backoff" against modified KN at orders 3/4/5 on a held-out slice. Gradient-free.
# (Chen & Goodman 1998; modified Kneser-Ney.)  base64-verified.
import re, math, base64, time
from collections import Counter, defaultdict

def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())

class KN:
    """Interpolated modified Kneser-Ney up to `order`."""
    def __init__(self, order=4):
        self.N = order
        self.cnt = [defaultdict(Counter) for _ in range(order + 1)]  # cnt[k][ctx(k-1)] -> Counter(w)
        self.D = [None] * (order + 1)
        self._g = {}                      # cache: (k, ctx) -> (denom, gamma)

    def train(self, ids):
        N = self.N
        # raw k-gram counts for the HIGHEST order; we derive lower orders by continuation
        for i in range(len(ids)):
            for k in range(1, N + 1):
                if i - k + 1 < 0: break
                ctx = tuple(ids[i - k + 1:i]); w = ids[i]
                self.cnt[k][ctx][w] += 1
        # replace lower-order (k<N) counts by CONTINUATION counts: number of distinct
        # left-extensions (unique word types preceding the k-gram).
        for k in range(N - 1, 0, -1):
            cont = defaultdict(Counter)
            for ctx, ctr in self.cnt[k + 1].items():       # ctx has length k
                suffix_ctx = ctx[1:]                         # drop earliest -> length k-1
                for w in ctr:                               # each distinct (k+1)-gram type
                    cont[suffix_ctx][w] += 1
            self.cnt[k] = cont
        # discounts per order from that order's count-of-counts (n1..n4)
        for k in range(1, N + 1):
            n = [0, 0, 0, 0, 0]
            for ctr in self.cnt[k].values():
                for c in ctr.values():
                    if 1 <= c <= 4: n[c] += 1
            n1, n2, n3, n4 = n[1], n[2], n[3], n[4]
            Y = n1 / (n1 + 2 * n2) if (n1 + 2 * n2) else 0.0
            D1 = 1 - 2 * Y * (n2 / n1) if n1 else 0.0
            D2 = 2 - 3 * Y * (n3 / n2) if n2 else 0.0
            D3 = 3 - 4 * Y * (n4 / n3) if n3 else 0.0
            self.D[k] = (max(D1, 0.0), max(D2, 0.0), max(D3, 0.0))
        self.uni_ctr = self.cnt[1].get((), Counter())          # w -> continuation count
        self.uni_tot = sum(self.uni_ctr.values()) or 1         # total distinct bigram types
        return self

    def _disc(self, k, c):
        D1, D2, D3 = self.D[k]
        return D1 if c == 1 else D2 if c == 2 else D3

    def _denom_gamma(self, k, ctx):
        key = (k, ctx); g = self._g.get(key)
        if g is not None: return g
        ctr = self.cnt[k].get(ctx)
        if not ctr:
            self._g[key] = (0.0, 1.0); return (0.0, 1.0)
        denom = sum(ctr.values()); D1, D2, D3 = self.D[k]
        n1 = sum(1 for c in ctr.values() if c == 1)
        n2 = sum(1 for c in ctr.values() if c == 2)
        n3 = sum(1 for c in ctr.values() if c >= 3)
        gamma = (D1 * n1 + D2 * n2 + D3 * n3) / denom if denom else 1.0
        self._g[key] = (denom, gamma); return (denom, gamma)

    def prob(self, ctx, w):
        """ctx = tuple of up to N-1 preceding tokens (longest suffix used)."""
        ctx = tuple(ctx[-(self.N - 1):])
        return self._prob(len(ctx) + 1, ctx, w)

    def _prob(self, k, ctx, w):
        if k == 1:                                   # KN unigram = continuation prob (O(1))
            return self.uni_ctr.get(w, 0) / self.uni_tot or 1e-10
        denom, gamma = self._denom_gamma(k, ctx)
        lower = self._prob(k - 1, ctx[1:], w)
        if denom == 0:
            return lower
        c = self.cnt[k][ctx].get(w, 0)
        return max(c - self._disc(k, c), 0.0) / denom + gamma * lower

class AddK:
    """Baseline ~ a naive smoothed trigram: hierarchical add-k, analytically normalized
    (each level sums to 1 over V), so perplexity is O(tokens) -- no per-vocab loop."""
    def __init__(self, order=3, k=0.1):
        self.N = order; self.k = k
        self.cnt = [defaultdict(Counter) for _ in range(order + 1)]
        self.uni = Counter()
    def train(self, ids):
        for i in range(len(ids)):
            self.uni[ids[i]] += 1
            for kk in range(2, self.N + 1):
                if i - kk + 1 < 0: break
                self.cnt[kk][tuple(ids[i - kk + 1:i])][ids[i]] += 1
        self.V = len(self.uni); self.tot = sum(self.uni.values())
        return self
    def prob(self, ctx, w):
        ctx = tuple(ctx[-(self.N - 1):]); kk = self.k; V = self.V
        for k in range(len(ctx) + 1, 1, -1):          # highest non-empty context wins
            c = self.cnt[k].get(ctx[-(k - 1):])
            if c: return (c.get(w, 0) + kk) / (sum(c.values()) + kk * V)
        return (self.uni.get(w, 0) + kk) / (self.tot + kk * V)

def perplexity(model, ids):
    s = 0.0; n = 0
    for i in range(1, len(ids)):
        ctx = ids[max(0, i - (model.N - 1)):i]
        p = model.prob(ctx, ids[i]); s += math.log(max(p, 1e-12)); n += 1
    return math.exp(-s / n)

if __name__ == "__main__":
    t0 = time.time()
    TEXT = open("corpus_books.txt", encoding="utf-8", errors="ignore").read()[:6_000_000]
    words = tokenize(TEXT)
    vocab = set(w for w, _ in Counter(words).most_common(8000))
    ids = [w if w in vocab else "<unk>" for w in words]
    cut = int(len(ids) * 0.9)
    train, test = ids[:cut], ids[cut:cut + 50000]
    print("train %d test %d  (%.0fs tokenize)" % (len(train), len(test), time.time() - t0))
    lines = []
    for k in (0.1, 0.01):
        ab = AddK(order=3, k=k).train(train)
        lines.append("add-%.2f trigram (naive base)  ppl=%.1f" % (k, perplexity(ab, test)))
    for o in (2, 3, 4, 5):
        t1 = time.time(); m = KN(order=o).train(train)
        ppl = perplexity(m, test)
        lines.append("modified KN order %d           ppl=%.1f  (%.0fs)" % (o, ppl, time.time() - t1))
    print("B64KN:" + base64.b64encode("\n".join(lines).encode()).decode())
