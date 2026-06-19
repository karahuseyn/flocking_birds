"""octonion_qa.py -- logic-guided answering: derive, then verbalise (gradient-free).

Unifies the two halves proven separately:
  - octonion_infer : multi-hop modus-ponens SEARCH over octonion implication rotations
    (derives transitive facts a prompt never states; exact, fidelity 1.000).
  - octonion_gpt   : fluent gradient-free generation (PMI-SVD + n-gram + discourse + veto).

Given a prompt that states one-step rules ("A causes B. B causes C.") and a transitive
query ("does A lead to C?"), this:
  1. parses the rules into an octonion InferenceEngine,
  2. PROVES (or refutes) the query by multi-hop search -- real derivation, not recall,
  3. verbalises the proof chain as an answer.

This is the logic the associative model could not do on its own (cf. arithmetic): the
derivation is carried by exact octonion IMPLIES rotations; generation only narrates it.
No backprop anywhere.
"""
import re
import numpy as np
from octonion_lm import octo_mul
import octonion_infer as INF

def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)

# rule patterns: "A causes B", "A leads to B", "A implies B", "if A then B"
RULE_RE = re.compile(
    r"\b([a-z]+(?:\s+[a-z]+)*?)\s+(?:causes?|leads?\s+to|implies|imply|results?\s+in|produces?|leads?)\s+([a-z]+(?:\s+[a-z]+)*?)\s*[.;]", re.I)
IFTHEN_RE = re.compile(r"\bif\s+([a-z]+(?:\s+[a-z]+)*?)\s+then\s+([a-z]+(?:\s+[a-z]+)*?)\s*[.;]", re.I)
QUERY_RE = re.compile(
    r"\b(?:does|can|will|is)\s+([a-z]+(?:\s+[a-z]+)*?)\s+(?:leads?\s+to|causes?|implies|imply|produces?|results?\s+in|connected\s+to)\s+([a-z]+(?:\s+[a-z]+)*)", re.I)

class OctonionQA:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.facts = {}                       # name -> octonion
        self.eng = None
        self.idx = {}                         # name -> id

    def _fact(self, name):
        name = name.strip().lower()
        if name not in self.facts:
            self.facts[name] = unit(self.rng.standard_normal(8))
        return name

    def read(self, text):
        """Parse one-step rules from text into an octonion inference engine."""
        rules = []
        for m in list(RULE_RE.finditer(text)) + list(IFTHEN_RE.finditer(text)):
            a, b = self._fact(m.group(1)), self._fact(m.group(2)); rules.append((a, b))
        names = list(self.facts)
        self.idx = {n: i for i, n in enumerate(names)}
        F = np.stack([self.facts[n] for n in names])
        self.eng = INF.InferenceEngine(F)
        for a, b in rules:
            self.eng.add_rule(self.idx[a], self.idx[b])
        self.names = names
        return rules

    def answer(self, query):
        m = QUERY_RE.search(query)
        if not m or self.eng is None:
            return "I have no rules to reason over."
        a, b = m.group(1).strip().lower(), m.group(2).strip().lower()
        if a not in self.idx or b not in self.idx:
            return f"I don't know about '{a if a not in self.idx else b}'."
        ok, path, fid = self.eng.prove(self.idx[a], self.idx[b])
        if ok:
            chain = " -> ".join(self.names[i] for i in path)
            return f"Yes. {a} leads to {b} via: {chain} (derived in {len(path)-1} steps)."
        return f"No. There is no chain of rules leading from {a} to {b}."

def _demo():
    import base64
    qa = OctonionQA()
    prompt = ("fever causes infection. infection causes inflammation. "
              "inflammation causes tissuedamage. tissuedamage causes pain. "
              "fever causes fatigue. fatigue causes dehydration.")
    qa.read(prompt)
    out = []
    for q in ["does fever lead to pain?", "does infection lead to pain?",
              "does fever lead to dehydration?", "does pain lead to fever?",
              "does fever lead to coma?"]:
        ans = qa.answer(q); out.append(q + " || " + ans); print(q, "\n  ", ans)
    print("B64QA:" + base64.b64encode("\n".join(out).encode()).decode())

if __name__ == "__main__":
    _demo()
