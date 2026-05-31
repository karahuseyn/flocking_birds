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
