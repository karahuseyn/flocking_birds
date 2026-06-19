# octonion_qa_mine.py -- network is closed, so MINE pseudo-Q&A pairs from the local medical
# corpora (biomed/science) with templates: a real factual sentence becomes the ANSWER and a
# synthetic question is generated from its subject. At inference the user's real question
# matches the synthetic Q via the entity-weighted matcher; the answer is real, fluent text.
import re, json, base64, sys

STOP = set(("study aim paper patients group groups results result trial data method methods "
            "outcome outcomes analysis objective objectives conclusion conclusions purpose "
            "background introduction participants patient mean median total this these those "
            "we our it they them there here table figure both each either neither all some many "
            "their its his her your my the a an chronic acute severe mild primary secondary common "
            "rare whether however although because such most more less high low new novel recent "
            "current present significant overall similar different various several").split())

def ok_subj(x):
    w = x.lower().split()
    if not (1 <= len(w) <= 4): return False
    if w[0] in STOP or w[-1] in STOP: return False        # no leading/trailing adjective/pronoun
    return any(len(t) > 4 for t in w)                     # at least one specific-ish term

# DEF / USE run on ORIGINAL case: subject must be a Capitalised entity (drug/disease name).
DEF = re.compile(r"\b([A-Z][A-Za-z0-9\-]{2,}(?:\s[A-Za-z0-9\-]+){0,3}?) (?:is|are) "
                 r"(?:a |an |the |defined as |characterized by |known as )")
USE = re.compile(r"\b([A-Z][A-Za-z0-9\-]{2,}(?:\s[A-Za-z0-9\-]+){0,3}?) (?:is|are) used "
                 r"(?:to treat|for|in the treatment of|to prevent|as)\b")
# lowercase patterns
PATS = [
    (re.compile(r"\bsymptoms of ([a-z][a-z\- ]{2,40}?) (?:include|are|may include|consist)"),
     "what are the symptoms of %s ?"),
    (re.compile(r"\b([a-z][a-z\- ]{2,40}?) (?:is|are) caused by"), "what causes %s ?"),
    (re.compile(r"\b(?:treatment|therapy) (?:of|for) ([a-z][a-z\- ]{2,40}?)[ ,.]"), "how is %s treated ?"),
    (re.compile(r"\b([a-z][a-z\- ]{2,40}?) (?:is|are|can be) (?:diagnosed|detected) (?:by|with|using|through)"),
     "how is %s diagnosed ?"),
    (re.compile(r"\b(?:prevention of|to prevent) ([a-z][a-z\- ]{2,40}?)[ ,.]"), "how is %s prevented ?"),
    (re.compile(r"\brisk factors for ([a-z][a-z\- ]{2,40}?) (?:include|are)"), "what are the risk factors for %s ?"),
]

def mine(paths, cap=40000, maxchars=400_000_000):
    pairs = []; seen = set()
    for p in paths:
        read = 0
        for line in open(p, encoding="utf-8", errors="ignore"):
            read += len(line)
            if read > maxchars: break
            s = line.strip()
            if not (40 <= len(s) <= 300): continue
            hit = None
            m = DEF.search(s)
            if m and ok_subj(m.group(1)): hit = ("what is %s ?" % m.group(1).lower(), s)
            else:
                m = USE.search(s)
                if m and ok_subj(m.group(1)): hit = ("what is %s used for ?" % m.group(1).lower(), s)
                else:
                    low = s.lower()
                    for rx, tmpl in PATS:
                        mm = rx.search(low)
                        if mm and ok_subj(mm.group(1).strip(" -")):
                            hit = (tmpl % mm.group(1).strip(" -"), s); break
            if hit and hit[0] not in seen:
                seen.add(hit[0]); pairs.append(hit)
            if len(pairs) >= cap: return pairs
    return pairs

if __name__ == "__main__":
    pairs = mine(["corpus_biomed.txt", "corpus_science.txt"])
    out = ["mined pseudo-Q&A pairs: %d" % len(pairs), ""]
    from collections import Counter
    kinds = Counter(q.split()[0] + " " + q.split()[1] for q, _ in pairs)
    out.append("by template head: " + ", ".join("%s=%d" % (k, v) for k, v in kinds.most_common()))
    out.append("")
    for q, a in pairs[:14]:
        out += ["Q: " + q, "A: " + a[:150], ""]
    json.dump(pairs, open("mined_qa.json", "w"))
    print("B64MINE:" + base64.b64encode("\n".join(out).encode()).decode())
