# kaggle_run.py -- paste this whole file into ONE Kaggle code cell and run.
# Gradient-free octonion generator + logic QA. CPU only (set Accelerator = None).
# Pure ASCII, no markdown, no cell-type ambiguity.

import os, subprocess, sys

# 1. get the code (clones the working branch where octonion_*.py live)
if not os.path.isdir("flocking_birds"):
    subprocess.run(["git", "clone", "-q", "-b",
                    "claude/octonionic-compression-tokenizer-nxbMq",
                    "https://github.com/karahuseyn/flocking_birds.git"], check=True)
os.chdir("flocking_birds")
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "scipy", "numpy"], check=True)

# 2. build a corpus. Option A: WikiText-103 via datasets (works on Kaggle).
#    Option B: comment this out and point CORPUS_PATH at any large .txt you attached.
CORPUS_PATH = "/kaggle/working/corpus_big.txt"
TARGET_CHARS = 400_000_000        # ~70M words; raise toward 1-2 GB on Kaggle 30 GB RAM
if not os.path.exists(CORPUS_PATH):
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "datasets"], check=True)
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    n = 0
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for row in ds:
            t = row["text"]
            if t and not t.startswith(" ="):
                f.write(t); n += len(t)
                if n >= TARGET_CHARS:
                    break
    print("wrote %.0f MB" % (n / 1e6))

# 3. build the gradient-free model (CPU, minutes)
import octonion_gpt as G, time
VOCAB = 80_000
t0 = time.time()
M = G.build(open(CORPUS_PATH, encoding="utf-8").read(), vocab_size=VOCAB)
G.save(M, "/kaggle/working/octogpt_wikitext_%d" % VOCAB)
print("build %.0f s" % (time.time() - t0))

# 4. semantic neighbours + generation
import numpy as np
emb, wi, vocab = M["emb"], M["wi"], M["vocab"]
def neighbours(w, k=8):
    return [vocab[i] for i in (emb @ emb[wi[w]]).argsort()[::-1][1:k+1]] if w in wi else ["<oov>"]
print("\n-- semantic neighbours --")
for w in ["king", "science", "war", "music", "love"]:
    print("%-9s ->" % w, neighbours(w))
print("\n-- generation (veto on by default) --")
for prompt in ["the history of", "in the year", "the city was", "she looked at"]:
    print("\n" + prompt + " |||\n  " + G.generate(M, prompt, n=60))

# 5. logic-guided reasoning: derive transitive facts, not recall
import octonion_qa as QA
qa = QA.OctonionQA()
qa.read("fever causes infection. infection causes inflammation. "
        "inflammation causes tissuedamage. tissuedamage causes pain. "
        "fever causes fatigue. fatigue causes dehydration.")
print("\n-- logic-guided QA (multi-hop derivation) --")
for q in ["does fever lead to pain?", "does infection lead to pain?",
          "does fever lead to dehydration?", "does pain lead to fever?",
          "does fever lead to coma?"]:
    print(q, "->", qa.answer(q))
