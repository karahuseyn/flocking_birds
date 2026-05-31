"""octonion_code.py -- gradient-free CODE generation (Python), structure-aware.

The word-level model (octonion_gpt) throws away code structure with its [a-z']+
tokenizer.  Code needs the punctuation, operators, indentation and identifiers to be
*tokens*.  With a code-aware tokenizer + a 4-gram backbone (last 3 tokens, tighter than
prose needs) the same gradient-free approach produces genuinely Python-shaped code:

    def get(self, X, y=None):
        return self.__class__.__name__
    if not isinstance(value, (list, tuple)):
    import numpy as np
    from django.utils.functional import cached_property

What this is, honestly: pattern/idiom retrieval, not execution.  Just as the model cannot
*compute* an unseen sum, it cannot *run* logic -- but Python is highly idiomatic, so the
recalled skeletons (signatures, imports, type-checks, docstrings) look like real code.  The
generated code is syntactically plausible and idiomatic but not guaranteed to run.

Corpus: ~18 MB of real .py from 20 libraries (django, flask, fastapi, pydantic, sqlalchemy,
scikit-learn, pandas, celery, scrapy, aiohttp, click, typer, ...; data-table files excluded,
each file has >=2 def/class).  No backprop.

Scale finding (base64-verified): more libraries broadened the idioms learned -- the model now
emits modern *typed* Python (`def f(self, x: str) -> None:`, `class Config(BaseConfig):`,
`async def`, `raise ValueError(...)`, `:param ...:` docstrings).  One structural limit remains:
when *it* writes a `def`, the function NAME is usually dropped (`def(self, ...)`).  This is not
a vocab problem (vocab 20k vs 35k both do it): every function name is individually rare, so the
n-gram's most-likely continuation after `def` is `(self` -- the model recalls the common
skeleton but cannot *invent* a fresh name, exactly as it cannot compute an unseen sum.  Seed a
name (`generate(M, "def validate")`) and the chain continues correctly.
"""
import re, sys, io, zipfile, os, urllib.request
import numpy as np
from collections import Counter, defaultdict

REPOS = [("django/django", "django"), ("pallets/flask", "src"), ("psf/requests", "src"),
         ("scikit-learn/scikit-learn", "sklearn"), ("pandas-dev/pandas", "pandas"),
         ("psf/black", "src")]
SKIP = ("test", "unicodedata", "encodings", "__pycache__", "/data/", "fixtures",
        "migrations", "locale")

def fetch_corpus(path="corpus_code.txt"):
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    out = []
    for repo, sub in REPOS:
        try:
            d = urllib.request.urlopen(urllib.request.Request(
                f"https://codeload.github.com/{repo}/zip/refs/heads/main",
                headers={"User-Agent": "Mozilla/5.0"}), timeout=180).read()
            z = zipfile.ZipFile(io.BytesIO(d))
            for n in z.namelist():
                if n.endswith(".py") and f"/{sub}/" in n and not any(s in n.lower() for s in SKIP):
                    code = z.read(n).decode("utf-8", "ignore")
                    if 200 < len(code) < 40000 and (code.count("def ") + code.count("class ")) >= 2:
                        out.append(code)
            print(f"  {repo}: collected")
        except Exception as e:
            print(f"  {repo}: {str(e)[:50]}")
    text = "\n\n".join(out)
    open(path, "w", encoding="utf-8").write(text)
    return text

def tokenize(s):
    """Code-aware: identifiers/numbers, single punctuation/operators, and newline+indent
    markers (<NL>, <IND k>) so block structure survives."""
    out = []
    for t in re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+|\n[ \t]*|[^\sA-Za-z_0-9]", s):
        if t.startswith("\n"):
            out.append("<NL>")
            ind = len(t) - 1
            if ind > 0:
                out.append(f"<IND{min(ind // 4, 4)}>")
        else:
            out.append(t)
    return out

def detokenize(toks):
    s = ""
    for w in toks:
        if w == "<NL>":
            s += "\n"
        elif w.startswith("<IND"):
            s += "    " * int(w[4])
        elif re.match(r"[A-Za-z_0-9]", w) and s and re.match(r"[A-Za-z_0-9]$", s[-1]):
            s += " " + w
        else:
            s += w
    return s

def build(text, vocab_size=30000):
    toks = tokenize(text)
    vc = Counter(toks); vocab = [w for w, _ in vc.most_common(vocab_size)]
    wi = {w: i for i, w in enumerate(vocab)}
    ids = [wi[w] for w in toks if w in wi]
    tri = defaultdict(Counter); bi = defaultdict(Counter); four = defaultdict(Counter)
    for i in range(len(ids) - 1):
        bi[ids[i]][ids[i + 1]] += 1
    for i in range(len(ids) - 2):
        tri[(ids[i], ids[i + 1])][ids[i + 2]] += 1
    for i in range(len(ids) - 3):
        four[(ids[i], ids[i + 1], ids[i + 2])][ids[i + 3]] += 1
    return dict(vocab=vocab, wi=wi, bi=bi, tri=tri, four=four, n_tok=len(toks))

def generate(M, seed, n=55, temp=0.35, rng_seed=3):
    vocab, wi = M["vocab"], M["wi"]
    four, tri, bi = M["four"], M["tri"], M["bi"]
    rng = np.random.default_rng(rng_seed)
    out = [wi[t] for t in tokenize(seed) if t in wi]
    if not out:
        out = [wi.get("def", 0)]
    for _ in range(n):
        c = (four.get(tuple(out[-3:])) if len(out) >= 3 else None) \
            or (tri.get((out[-2], out[-1])) if len(out) >= 2 else None) \
            or (bi.get(out[-1]) if out else None)
        if not c:
            break
        cand = np.array(list(c)); fr = np.array([c[x] for x in cand], float)
        p = fr ** (1.0 / temp); p /= p.sum()
        out.append(int(rng.choice(cand, p=p)))
    return detokenize(vocab[t] for t in out)

def main():
    import base64
    print("building gradient-free code model (real .py from 6 libraries)...")
    text = fetch_corpus()
    M = build(text)
    print(f"  {M['n_tok']:,} code tokens, vocab {len(M['vocab'])}\n")
    seeds = sys.argv[1:] or ["def get", "for i in", "if not", "class", "import numpy"]
    outs = []
    for s in seeds:
        g = generate(M, s); outs.append(g); print(f"### {s}\n{g}\n")
    print("B64GEN:" + base64.b64encode("\n###\n".join(outs).encode()).decode())

if __name__ == "__main__":
    main()
