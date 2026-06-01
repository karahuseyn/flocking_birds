# Running Octonion-GPT on Kaggle

A step-by-step guide to scaling the gradient-free octonion generator (`octonion_gpt.py`)
to a large corpus on Kaggle. **No GPU, no backprop** — the bottleneck is RAM + sparse
linear algebra, which is exactly what a free Kaggle CPU notebook gives you.

## TL;DR
1. New Kaggle Notebook → **Settings → Accelerator = None (CPU)**, **Persistence = on**.
2. Upload / open `kaggle_octonion_gpt.ipynb` (in this repo) and Run All.
   (it clones the `claude/octonionic-compression-tokenizer-nxbMq` branch — that's where the code lives until merged to `main`)
3. It clones the repo, streams WikiText-103, builds the model (minutes), and generates.

## Why CPU, not GPU
"Training" here is: count co-occurrences → shifted **PPMI** matrix → **truncated SVD** →
n-gram tables. There are **no gradients and no weights to learn**, so a GPU does nothing.
Pick CPU and you keep your free GPU quota. The real resource that matters is **RAM**.

## What scales, and the numbers
Measured on a 15 GB box (base64-verified), full 335 MB PubMed corpus = **46.5M words**:

| vocab | PPMI build | full build | peak RAM | PPMI nonzeros |
|------:|-----------:|-----------:|---------:|--------------:|
| 30k   | 38 s       | 149 s      | 7.2 GB   | 8.8M          |
| 50k   | 39 s       | 153 s      | 7.5 GB   | 10.8M         |

RAM barely grows with vocab (the n-gram tables dominate, not vocab²). **On Kaggle's 30 GB
you can comfortably run vocab 100k and a corpus several times larger.**

### The one fix that unlocks scale
The naive build holds all sparse co-occurrence index arrays at once (~9 GB at 46M words)
and OOMs. `octonion_gpt.build()` instead accumulates **offset-by-offset into a CSR**, so
peak memory stays low. This is already in the code — nothing to change.

## Choosing a corpus
On Kaggle the HuggingFace / Wikimedia endpoints that are blocked in some sandboxes work,
so you have real options:

- **WikiText-103** (`datasets`, streaming) — ~500 MB clean English. The notebook uses this.
- **C4 / OpenWebText shards** — web text, very large; take a handful of shards.
- **Attach a Kaggle Dataset** (PubMed full, arXiv full, books) and set `CORPUS_PATH`.

Bigger + cleaner corpus is the **main lever for quality**. The model is corpus-agnostic:
on medical text `treatment → drugs, therapeutic, therapies`; on literature
`love → affection, soul, friendship`, `war → napoleon, prussia, emperor` — and generation
takes the corpus's *style* (clinical-abstract vs Victorian-novel).

## Tuning (all gradient-free)
`G.generate(M, prompt, temp=, drift=, w_subj=, w_pred=, w_alt=, rep_pen=, n=)`

| knob | effect |
|------|--------|
| `temp` | ↑ diversity |
| `rep_pen` | ↑ suppresses repetition (the main quality issue at scale) |
| `drift` | ↑ topic moves faster (argument-like); ↓ stays on the prompt |
| `w_subj`,`w_pred` | proposition (subject/predicate) octonion-bind strength |
| `w_alt` | subject/verb syntactic rhythm |

## Caching
`G.save(M, path)` / `G.load(path)` persist the embedding + n-gram tables to
`/kaggle/working`, so a second run skips the build entirely.

## Honest expectations
This is a genuine **small, gradient-free, linear-time** generator: fluent, prompt-faithful,
corpus-styled prose at ~thousands of tokens/sec on one CPU. It is **not** GPT-level — local
fluency and topic flow are strong, but cross-sentence logical structure and long-range
argument are still open (see the repo README's end-to-end section). Scale improves fluency
and coverage; it does not by itself add reasoning.


## Can it do arithmetic? (a "crazy question", honestly answered)

No — and the reason is fundamental. The model is statistical *association*, not symbolic
*execution*. Base64-verified on a synthetic "A plus B equals C" corpus with all sums
involving 7 held out:

- **seen operand pairs**: 6/6 correct ("three plus five equals **eight**") — pure retrieval.
- **unseen pairs (involve 7)**: 0/5 — it cannot compute a sum it never saw.

Worse, the plain trigram backbone gets even *seen* sums wrong (0/6): an n-gram sees only the
last two tokens, so by "equals" it has already forgotten the first operand `a`. Arithmetic
needs to bind **three** symbols at once (a, b, result); an n-gram window structurally can't.
Only an explicit operand-pair table recovers the memorised answers.

So: the model is a perfect **lookup table** (memorised facts, definitions, a times-table it
was shown) but performs **no procedure**. Computation requires step-by-step state
transformation, which this gradient-free associative architecture does not have. A code or
math corpus will reproduce *patterns and idioms* it has seen, not *execute* logic.

## Code generation (octonion_code.py)

The same gradient-free idea with a **code-aware tokenizer** (keeps punctuation, operators,
indentation as tokens) + a 4-gram backbone produces genuinely Python-shaped code:
`def get(self, X, y=None):`, `if not isinstance(value, (list, tuple)):`,
`import numpy as np`, `from django.utils.functional import cached_property`. It is
idiom/signature **retrieval**, not execution -- syntactically plausible, not guaranteed to run
(consistent with the arithmetic result: it recalls patterns, it does not run procedures).
Run `python3 octonion_code.py "def get"` (downloads ~10MB of real .py from 6 libraries).

## A test-time-compute alternative: TTC-as-veto

Classic test-time compute maximizes a score (best-of-N, beam search). For a gradient-free
n-gram that **degenerates into repetition** -- maximizing any fixed score (coherence,
optionality) loops on the single highest-scoring pattern. Base64-verified on biomed:
coherence-max -> trigram repetition rises and topic freezes; optionality-max -> 55%
repetition ("the mean sd and the mean sd and ...").

The fix is to spend TTC on **rejection, not maximization**: keep stochastic sampling for
diversity, but use a one-step lookahead to *veto* candidates that would recreate an
already-emitted bigram (a loop), then sample from the rest. Light (O(candidates), no extra
forward passes). Result: trigram repetition 0.3% -> **0.0%** with coherence/drift unchanged,
and visibly loop-free clinical prose ("the relative efficacy and safety of olanzapine or
risperidone for eight weeks ... clinical follow up was months"). Enabled by default
(`generate(..., veto=True)`).

## Logic-guided reasoning (octonion_infer + octonion_qa)

The notebook also demonstrates **multi-hop logical inference** — the one place the algebra
does true *derivation*, not recall. An implication `A=>B` is an exact octonion rotation, so
chaining rules composes losslessly (verified fidelity 1.000 up to 6 hops). `octonion_qa`
reads one-step rules from a prompt and derives transitive answers never stated:

```
prompt: "fever causes infection. infection causes inflammation.
         inflammation causes tissuedamage. tissuedamage causes pain."
Q: does fever lead to pain?  ->  Yes, via fever -> infection -> inflammation ->
                                  tissuedamage -> pain (derived in 4 steps)
Q: does pain lead to fever?  ->  No (wrong direction, no chain)
Q: does fever lead to coma?  ->  I don't know (unknown concept)
```

This is the procedural reasoning that arithmetic could NOT do: it works because the IMPLIES
rotations are exact and the search is classical forward-chaining over them. The associative
generator narrates; the octonion algebra derives. `python3 octonion_qa.py` runs the demo.