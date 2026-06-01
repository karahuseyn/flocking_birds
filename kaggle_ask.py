# kaggle_ask.py -- PASTE THIS IN A SECOND CELL, after kaggle_standalone.py has built M.
# Edit the lines at the bottom to ask your own questions. All gradient-free, no training.

# ------------------------------------------------------------------------------
# helpers (use M, emb, wi, vocab, generate, QA from the first cell)
# ------------------------------------------------------------------------------
def neighbours(word, k=10):
    """Most semantically related words to `word` (gradient-free PMI-SVD geometry)."""
    if word not in wi:
        print(word, "-> (not in vocabulary)"); return
    print("%-12s ->" % word, [vocab[i] for i in (emb @ emb[wi[word]]).argsort()[::-1][1:k+1]])

def analogy(a, b, c, k=5):
    """a : b :: c : ?   e.g. analogy('man','woman','king') -> queen (if the corpus has it)."""
    for w in (a, b, c):
        if w not in wi:
            print("analogy (unknown word: %s)" % w); return
    v = emb[wi[b]] - emb[wi[a]] + emb[wi[c]]; v = v / (np.linalg.norm(v) + 1e-12)
    out = [vocab[i] for i in (emb @ v).argsort()[::-1] if vocab[i] not in (a, b, c)][:k]
    print("%s : %s  ::  %s : %s" % (a, b, c, out))

def complete(prompt, n=50, temp=0.5, seed=1):
    """Continue `prompt` (flocking-7 generation). Raise temp for more variety."""
    print("\n>>>", prompt, "\n   " + generate(M, prompt, n=n, temp=temp, rng_seed=seed))

def reason(rules_text, *questions):
    """Logic test. Give rules like 'A causes B. B causes C.' then ask
    'does A lead to C?'. It derives the multi-hop chain (or refuses)."""
    q = QA(); q.read(rules_text)
    print("rules:", rules_text)
    for question in questions:
        print("  ", question, "->", q.answer(question))

# ==============================================================================
# ASK YOUR OWN QUESTIONS  (edit freely, re-run this cell -- the model M is reused)
# ==============================================================================

print("===== 1. WORD MEANING (what is this word near?) =====")
for w in ["king", "doctor", "ocean", "music", "war", "love"]:
    neighbours(w)

print("\n===== 2. ANALOGY (a is to b as c is to ?) =====")
analogy("man", "woman", "king")
analogy("france", "paris", "england")
analogy("small", "smaller", "big")

print("\n===== 3. PROMPT COMPLETION (finish the sentence) =====")
complete("the meaning of life is")
complete("scientists have discovered that")
complete("in the beginning")

print("\n===== 4. LOGIC (give rules, then ask) =====")
reason("rain causes floods. floods cause damage. damage causes cost.",
       "does rain lead to cost?", "does flood cause cost?", "does cost lead to rain?")
reason("study leads to knowledge. knowledge leads to power. power leads to responsibility.",
       "does study lead to responsibility?", "is study connected to power?",
       "does responsibility lead to study?")
