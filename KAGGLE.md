# Running Octonion-GPT on Kaggle

A step-by-step guide to scaling the gradient-free octonion generator (`octonion_gpt.py`)
to a large corpus on Kaggle. **No GPU, no backprop** — the bottleneck is RAM + sparse
linear algebra, which is exactly what a free Kaggle CPU notebook gives you.

## TL;DR
1. New Kaggle Notebook → **Settings → Accelerator = None (CPU)**, **Persistence = on**.
2. **Easiest:** paste `kaggle_run.py` into ONE code cell and run (pure ASCII, no cell-type
   issues). OR upload `kaggle_octonion_gpt.ipynb` and Run All.
   (both clone the `claude/octonionic-compression-tokenizer-nxbMq` branch -- where the code lives until merged to `main`)
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

## No-internet Kaggle (recommended): kaggle_standalone.py

Kaggle notebooks have **internet OFF by default**, so `git clone` and `datasets` downloads
fail (`Could not resolve host: github.com`). `kaggle_standalone.py` needs neither -- it has
all the octonion code inlined and only needs numpy + scipy (pre-installed). Steps:

1. New notebook, Accelerator = None (CPU).
2. **+ Add Input** -> attach any text dataset (search e.g. "wikitext", or upload a .txt),
   or skip it to run on a tiny built-in demo corpus.
3. Paste all of `kaggle_standalone.py` into one cell and run. It auto-finds the biggest
   .txt under `/kaggle/input`, or set `CORPUS_PATH` explicitly.

(If you DO turn on internet in notebook settings, `kaggle_run.py` / the .ipynb also work and
will clone the repo + stream WikiText-103.)

## Getting BIG data into the standalone (the missing piece)

The "no corpus found" message means no dataset is attached. `kaggle_standalone.py` now reads
**.txt / .csv / .parquet / .json** (auto-picks the longest text column), so:

1. In the notebook, right panel -> **"+ Add Input"**.
2. Search and add a text dataset. Known-good ones on Kaggle:
   - **"wikitext"** (wikitext-103, ~500 MB clean English)
   - **"bookcorpus"** / **"gutenberg"** (books, general English)
   - **"arxiv"** abstracts, **"pubmed"** (domain text)
3. Re-run the cell. It prints the candidate files it found and uses the biggest, e.g.
   `using: /kaggle/input/wikitext/.../train.parquet (480 MB of text)`, then builds at
   vocab 40k in a couple of minutes.

Verified locally on a 24 MB book corpus: neighbours sharpen (king -> france, otho, wamba;
science -> mathematics, logic, astronomy) and generation is fluent Victorian prose. With a
500 MB+ dataset the embedding and coverage improve further.

## Verified on Kaggle: full WikiText-103 (user run)

Confirmed working end-to-end on Kaggle CPU with the `rohitgr/wikitext` dataset:
- corpus `wiki.train.raw` ~400 MB used, **60.2M words, vocab 40k**, built in **291 s** (no OOM).
- semantic neighbours become world-knowledge sharp: `king -> reigned, accession, cnut,
  kingship, kinsman, aethelred`; `science -> sociology, psychology, anthropology`;
  `war -> military, hostilities, invasion, boer`.
- generation is fluent encyclopedic prose, loop-free (TTC-veto): "...during roman britain
  within the hundred days napoleon appointed ferino as commander of allied troops ... fort
  ticonderoga...".
- honest limit unchanged: strong local fluency and on-topic spans, but cross-sentence logic
  still drifts -- scale sharpens meaning and coverage, it does not add long-range reasoning.

## Flocking-7 generation + vocab 80k (the boids philosophy, back at the core)

A bird stays in the flock by tracking its ~7 nearest neighbours -- not the whole flock, not a
fixed set, just whichever 7 are nearest right now. Applied to generation: each next word is
scored against the **last 7 tokens** (the flock) by two boids rules, plus separation = the loop
veto:
  - **cohesion**: align to the 7-token centroid (stay on topic, don't leave the flock)
  - **alignment**: follow the flock's motion direction emb[last]-emb[first] (keep the local flow)

Base64-verified on biomed (vs real text local-coherence 0.894): greedy 0.880 -> **flock-7 0.912**
(above real), drift 0.43 -> 0.37. Generation holds a clinical sentence together far better:
"...a pharmacogenetic effect ... patients were randomly assigned to one of three groups ...
progression free survival ... hazard ratio ci p ... no difference in the control group."
`generate(..., w_cohesion=2.0, w_align=1.0, flock=7)` -- flocking on by default.

Vocab is now **80k** by default (richer terms; ~9-10 GB peak build, fine on Kaggle's 30 GB).

## Cutting drift: the boids 4th rule + cyclic Fano echo (on by default)

The honest open problem was **drift** -- every memory in the generator (`cvec`, `s`, `SK`,
flock, n-gram) forgets after ~5-7 tokens, so the prompt's topic washes out within a clause
and the text random-walks. Two gradient-free, octonion/boids-native fixes, both base64-verified:

- **#1 Goal anchor (boids' 4th rule, the migratory urge).** Beyond the local flock, a real
  flock also steers toward a global goal. We keep a **persistent topic target** = the prompt's
  content centroid (non-decaying) and add `+ w_goal * cos(cand, goal)`. Verified to roughly
  **halve start->end drift** (e.g. octonion_gpt 0.449 -> 0.305, anchor 0.804 -> 0.829, repetition
  unchanged) while local coherence holds. **On by default** (`w_goal=3.0`); set `goal_ema~0.02`
  to let the target migrate slowly instead of staying fixed.
- **#4 Cyclic Fano-path echo (the algebra's own contribution).** A "fano path" is a sequence of
  Fano-unit multiplications; the roles are the **running products** along the walk e1..e7 --
  `R_0 = 1`, `R_k = R_{k-1}*e_k` (not single units per slot). Token at position `p` takes role
  `R_{p mod 7}`, so the last-7 flock fills a **7-slot holographic register**
  `H = sum R[p mod 7] (x) embK[token]`. Unbinding the next slot's role (exact -- Artin: a
  two-generator octonion subalgebra is associative) reads back **what filled this slot one cycle
  (7 tokens) ago** -> a period-7 anaphoric/parallel-cadence prior. Base64-verified that the
  *running-product path* beats single-unit roles (drift 0.354 -> 0.300, anchor 0.824 -> 0.848)
  and helps on top of the goal anchor in the full KN generator (drift 0.342 -> 0.289, anchor
  0.830 -> 0.849), so it is **on by default** (`w_fano=4.0`; set 0 to disable). Note the path
  matters: a closed 3-cycle on one Fano line {1,2,4} did *not* help -- the full 7-step Singer
  walk does.

What we tried and dropped, honestly: a separate **slow-cycle topic** term (#5) is subsumed by the
goal anchor (it's the `goal_ema -> 0` limit), and **continuous-key kNN retrieval** (#2) gave no
clear coherence gain gradient-free -- kNN-LM's power comes from a *trained* context encoder we
don't have; our best gradient-free key (the octonion state `SK`) didn't beat a bag-of-words mean,
and the discrete-key limit of kNN is just a higher-order n-gram. Its one real effect was lower
repetition. The reproducible experiments live in `exp_topic_anchor.py`, `exp_fano_layer.py`,
`exp_knn.py`. New tuning knobs: `generate(..., w_goal=3.0, goal_ema=0.0, w_fano=0.0)`.

## Backbone: modified Kneser-Ney (order 3), the right "discrete-key" smoothing

The n-gram backbone was raw trigram counts with a hard bigram fallback. We replaced it with
**interpolated modified Kneser-Ney** (Chen & Goodman 1998) -- absolute discounting plus
*continuation* probabilities (a word's likelihood from how many distinct contexts precede it,
not its raw frequency), interpolated trigram -> bigram -> unigram. Base64-verified on held-out
`corpus_books`:

| model | held-out perplexity |
|------|--------------------:|
| naive add-0.01 trigram | 845.7 |
| modified KN order 2 | 223.4 |
| **modified KN order 3** | **201.5** |
| modified KN order 4 | 198.5 |
| modified KN order 5 | 198.0 |

Two honest findings: **(1)** KN smoothing beats naive smoothing ~**4x** (846 -> 201); **(2)**
going past order 3 buys **<2%** -- so the lever is the *smoothing*, not the order, which validates
the generator's existing trigram. We therefore ship **KN order-3** (not 4/5: not worth the
memory/time). It also helps *generation*, not just perplexity: with everything else fixed,
swapping raw counts for KN-3 log-probs and widening candidates to trigram-union-bigram moved
local-coherence 0.887 -> **0.911**, drift 0.382 -> 0.347, anchor 0.802 -> 0.820, repetition
unchanged. This is now the default backbone in both `octonion_gpt.py` and `kaggle_standalone.py`
(`build()` precomputes the continuation counts and discounts; `generate()` scores candidates by
`_kn3_logprob`). Reproduce: `python3 exp_kn.py`.