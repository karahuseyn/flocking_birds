"""Octonion symptom -> disease ranker  (the "rule of 7" applied to diagnosis).

Same gradient-free, octonion-hyperdimensional idea as octonion_lm.py, but for an
*unordered set* of symptoms instead of a character sequence.  Everything follows
the number 7 -- the Fano plane has 7 points and 7 lines, the octonion has 7
imaginary units, so we build the hypervector as a fractal of sevens:

    octonion          = 8 reals          (1 real + 7 Fano-governed imaginaries)
    heptad            = 7 octonions
    super-heptad      = 7 heptads         = 49 octonions
    hypervector       = 7 super-heptads   = 343 octonions  (= 7**3, 2744 reals)

A symptom is a fixed random *unit-octonion* hypervector (a codebook entry, never
trained).  A patient is the bundle (sum) of the symptoms they present.  A disease
is the bundle of all its training patients -- one prototype hypervector each.
Diagnosis = rank the 41 disease prototypes by similarity to the query and return
the nearest *seven* (the 7-neighbourhood, as in the flocking birds).

No backprop, no weights -- one pass over the data, then nearest-neighbour recall.
"""
import csv, io, os, sys, urllib.request
import numpy as np
from octonion_lm import octo_norm

HERE = os.path.dirname(os.path.abspath(__file__))
SEVEN = 7
SLOTS = SEVEN ** 3                       # 343 octonions  -> 2744-dim hypervector
DATA_URL = ("https://raw.githubusercontent.com/anujdutt9/"
            "Disease-Prediction-from-Symptoms/master/dataset/training_data.csv")

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data():
    path = os.path.join(HERE, "symptoms_train.csv")
    if os.path.exists(path):
        text = open(path, encoding="utf-8").read()
    else:
        print("fetching symptom/disease dataset (132 symptoms, 41 diseases)...")
        text = urllib.request.urlopen(
            urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"}),
            timeout=60).read().decode("utf-8", "replace")
        open(path, "w", encoding="utf-8").write(text)
    rows = [r for r in csv.reader(io.StringIO(text)) if r and any(c.strip() for c in r)]
    hdr = rows[0]
    symptoms = [c for c in hdr if c.strip() and c.strip() != "prognosis"]
    sidx = {s: i for i, s in enumerate(symptoms)}
    plabel = hdr.index("prognosis")
    X, y = [], []
    for r in rows[1:]:
        present = [symptoms[i] for i in range(len(symptoms)) if r[i].strip() == "1"]
        X.append(present); y.append(r[plabel].strip())
    return symptoms, sidx, X, y

# --------------------------------------------------------------------------
# Octonion hyperdimensional encoder (the rule of 7)
# --------------------------------------------------------------------------
class SymptomEncoder:
    def __init__(self, n_symptoms, slots=SLOTS, seed=int("0709", 10) ^ 1916):
        rng = np.random.default_rng(seed)
        # one fixed unit-octonion per (symptom, slot) -- the codebook
        emb = rng.standard_normal((n_symptoms, slots, 8))
        self.emb = (emb / octo_norm(emb)).reshape(n_symptoms, slots * 8)
        self.D = slots * 8
        self.W = (self.D + 63) // 64

    def encode(self, symptom_ids):
        """Bundle the present symptoms into one (D,) unit hypervector."""
        if len(symptom_ids) == 0:
            return np.zeros(self.D)
        v = self.emb[symptom_ids].sum(0)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def pack(self, vecs):
        """(N, D) float -> (N, W) uint64 bipolar sign codes."""
        vecs = np.atleast_2d(vecs)
        bits = np.zeros((len(vecs), self.W * 64), dtype=np.uint8)
        bits[:, :self.D] = (vecs > 0)
        return np.packbits(bits, axis=1).view(np.uint64)

# --------------------------------------------------------------------------
# The ranker: one bundled prototype per disease, ranked by Hamming similarity
# --------------------------------------------------------------------------
class OctonionDiagnoser:
    def __init__(self, symptoms, sidx):
        self.symptoms, self.sidx = symptoms, sidx
        self.enc = SymptomEncoder(len(symptoms))

    def fit(self, X, y):
        self.diseases = sorted(set(y))
        self.didx = {d: i for i, d in enumerate(self.diseases)}
        proto = np.zeros((len(self.diseases), self.enc.D))
        for present, disease in zip(X, y):
            ids = [self.sidx[s] for s in present if s in self.sidx]
            proto[self.didx[disease]] += self.enc.encode(ids)        # bundle cases
        norms = np.linalg.norm(proto, axis=1, keepdims=True)
        self.proto_bits = self.enc.pack(proto / np.where(norms > 1e-9, norms, 1.0))
        return self

    def rank(self, present_symptoms, k=SEVEN):
        """Return the k most likely diseases (name, similarity%) for given symptoms."""
        ids = [self.sidx[s] for s in present_symptoms if s in self.sidx]
        q = self.enc.pack(self.enc.encode(ids))[0]
        ham = np.bitwise_count(self.proto_bits ^ q).sum(1)           # (n_diseases,)
        order = np.argsort(ham)[:k]
        sim = 100.0 * (1.0 - ham[order] / self.enc.D)
        return [(self.diseases[i], float(s)) for i, s in zip(order, sim)]

# --------------------------------------------------------------------------
# Train / evaluate / demo
# --------------------------------------------------------------------------
def main():
    symptoms, sidx, X, y = load_data()
    print(f"dataset: {len(X)} cases, {len(symptoms)} symptoms, "
          f"{len(set(y))} diseases  (hypervector {SLOTS} octonions = {SLOTS*8} dims)")

    rng = np.random.default_rng(7)
    perm = rng.permutation(len(X))
    ntest = len(X) // 5
    te, tr = perm[:ntest], perm[ntest:]
    Xtr, ytr = [X[i] for i in tr], [y[i] for i in tr]
    Xte, yte = [X[i] for i in te], [y[i] for i in te]

    diag = OctonionDiagnoser(symptoms, sidx).fit(Xtr, ytr)

    top1 = top3 = top7 = 0
    for present, true in zip(Xte, yte):
        ranked = [d for d, _ in diag.rank(present, k=SEVEN)]
        top1 += (ranked[0] == true)
        top3 += (true in ranked[:3])
        top7 += (true in ranked[:7])
    n = len(Xte)
    print(f"\nheld-out ranking accuracy ({n} cases):")
    print(f"  top-1 : {100*top1/n:5.1f}%")
    print(f"  top-3 : {100*top3/n:5.1f}%")
    print(f"  top-7 : {100*top7/n:5.1f}%   (the 7-neighbourhood)")

    # live demo: hand it symptoms, get a ranked differential
    demos = [
        ["high_fever", "headache", "vomiting", "nausea", "chills", "sweating"],
        ["itching", "skin_rash", "nodal_skin_eruptions"],
        ["cough", "high_fever", "breathlessness", "chest_pain", "phlegm"],
        ["yellowish_skin", "dark_urine", "abdominal_pain", "loss_of_appetite", "nausea"],
        ["fatigue", "weight_loss", "increased_appetite", "polyuria", "restlessness"],
    ]
    print("\nlive differential diagnosis (top-7 ranked diseases):")
    for ds in demos:
        known = [s for s in ds if s in sidx]
        print(f"\n  symptoms: {', '.join(known)}")
        for rank, (d, s) in enumerate(diag.rank(known, k=SEVEN), 1):
            print(f"    {rank}. {d:30s} {s:5.1f}%")

if __name__ == "__main__":
    main()
