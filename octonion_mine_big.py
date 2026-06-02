# octonion_mine_big.py -- aggressive mining for a MUCH bigger knowledge pool. Looser entity
# definition (any 'Entity is/are/was/were ...') + action patterns ('Entity reduces/causes/
# prevents ...'), dedup by question -> ~one fact-keyed pair per distinct entity*template.
import re, json, base64
from octonion_qa_mine import STOP, ok_subj

GENERIC = set(("when before after during two one three four five serum plasma blood baseline "
    "mean median level levels concentration concentrations assessment assessments case cases "
    "group groups analysis dose doses sample samples value values rate rates time times day "
    "days week weeks month months patient patients response responses outcome outcomes data "
    "results result study studies trial trials score scores test tests measurement number "
    "total all both each their there here this these those eighty seventy sixty fifty forty").split())
SUFFIX = re.compile(r"(ine|ol|ide|ate|azole|mab|nib|tinib|statin|pril|sartan|cycline|mycin|"
                    r"parin|trel|navir|vir|prazole|caine|dipine|olol|itis|osis|emia|aemia|"
                    r"pathy|opathy|oma|asis|iasis|plasia|trophy|algia|ectomy|otomy|oplasty)$")
MARK = set("syndrome disease disorder deficiency infection infections cancer carcinoma tumor "
           "tumour failure injury fracture disorder maneuver procedure surgery therapy".split())

def is_entity(subj):
    w = subj.lower().split()
    if not w or w[0] in GENERIC or w[-1] in GENERIC: return False
    if re.search(r"[0-9]", subj): return True              # drug codes: MK-3207, COX-2
    if SUFFIX.search(w[-1]): return True                   # drug/disease suffixes
    if any(t in MARK for t in w): return True              # 'X syndrome', 'Y maneuver'
    return len(w) >= 2 and all(len(t) > 3 for t in w)      # specific multiword term

DEF = re.compile(r"\b([A-Z][A-Za-z0-9\-]{2,}(?:\s[A-Za-z0-9\-]+){0,3}?) (?:is|are|was|were) ")
USE = re.compile(r"\b([A-Z][A-Za-z0-9\-]{2,}(?:\s[A-Za-z0-9\-]+){0,3}?) (?:is|are) used "
                 r"(?:to treat|for|in the treatment of|to prevent|as)\b")
ACT = re.compile(r"\b([A-Z][A-Za-z0-9\-]{2,}(?:\s[A-Za-z0-9\-]+){0,3}?) "
                 r"(?:reduces|increases|improves|inhibits|prevents|induces|causes|decreases|"
                 r"enhances|blocks|activates|regulates|affects|lowers|raises|stimulates) ")
PATS = [
    (re.compile(r"\bsymptoms of ([a-z][a-z\- ]{2,40}?) (?:include|are|may include|consist)"),
     "what are the symptoms of %s ?"),
    (re.compile(r"\b([a-z][a-z\- ]{2,40}?) (?:is|are) caused by"), "what causes %s ?"),
    (re.compile(r"\b(?:treatment|therapy) (?:of|for) ([a-z][a-z\- ]{2,40}?)[ ,.]"), "how is %s treated ?"),
    (re.compile(r"\b([a-z][a-z\- ]{2,40}?) (?:is|are|can be) (?:diagnosed|detected) (?:by|with|using|through)"),
     "how is %s diagnosed ?"),
    (re.compile(r"\b(?:prevention of|to prevent) ([a-z][a-z\- ]{2,40}?)[ ,.]"), "how is %s prevented ?"),
]

def mine_big(paths, cap=120000, maxchars=400_000_000):
    pairs = []; seen = set()
    for p in paths:
        read = 0
        for line in open(p, encoding="utf-8", errors="ignore"):
            read += len(line)
            if read > maxchars: break
            s = line.strip()
            if not (40 <= len(s) <= 300): continue
            hit = None
            m = USE.search(s)
            if m and ok_subj(m.group(1)): hit = ("what is %s used for ?" % m.group(1).lower(), s)
            if hit is None:
                m = DEF.search(s)
                if m and ok_subj(m.group(1)) and is_entity(m.group(1)): hit = ("what is %s ?" % m.group(1).lower(), s)
            if hit is None:
                m = ACT.search(s)
                if m and ok_subj(m.group(1)) and is_entity(m.group(1)): hit = ("what does %s do ?" % m.group(1).lower(), s)
            if hit is None:
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
    pairs = mine_big(["corpus_biomed.txt", "corpus_science.txt"])
    json.dump(pairs, open("mined_big.json", "w"))
    from collections import Counter
    c = Counter(" ".join(q.split()[:3]) for q, _ in pairs)
    out = ["mined_big pairs: %d" % len(pairs), "by head: " + str(dict(c.most_common(6))), ""]
    import random; random.seed(2)
    for q, a in random.sample(pairs, 12): out += ["Q: " + q, "A: " + a[:110], ""]
    print("B64BIG:" + base64.b64encode("\n".join(out).encode()).decode())
