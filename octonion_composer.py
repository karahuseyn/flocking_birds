# octonion_composer.py -- COMPOSITIONAL construction in the octonion framework.
#
# Mirrors how a mind builds a coherent composition, octonionically and gradient-free:
#   1. WORLD MODEL (muktesebat): the base octonion network M (already built in the bot).
#   2. FUZZY SCAN: from the topic octonion, softly gather its neighbourhood's candidate facts,
#      then CLUSTER them (spherical k-means on the octonion slots) into a few distinct topic-
#      headings (aspects). "Rule of 7": at most 7 aspects.
#   3. LOGICAL COHERENCE (mantik silsilesi): order the aspects as a RELATION-CONSISTENT fano
#      path -- a greedy geodesic chain on S^7 from the aspect nearest the topic, each next
#      chosen for maximal relational consistency (smoothest conceptual flow). "relation=path".
#   4. GRAMMAR / SURFACE: realise each ordered aspect with a real sentence (central to the
#      aspect AND relevant to the topic), stitched in chain order -> a multi-aspect, logically
#      ordered composition. The base model is never modified (reads a fitted OctonionPRBot).
import numpy as np
from exp_fano_layer import unit
from octonion_pr_extract import sents
from octonion_pr_slot import toks
import octonion_infer as INF

# octonion-grounded role prototypes: each role is a point in the same PMI-SVD octonion space,
# embedded from canonical seed phrases. An aspect's role = nearest prototype (cosine on S^7),
# replacing brittle cue-word counting with geometry in the model's own concept space.
SEED = {
    "definition": "is a type of disease condition or disorder a form of refers to defined as",
    "cause": "is caused by due to leads to results from because risk factor associated with",
    "mechanism": "occurs when the body the process by which is produced released blocks inhibits",
    "symptom": "symptoms and signs pain swelling fever fatigue nausea you may feel",
    "diagnosis": "is diagnosed by a test detected examination biopsy imaging scan blood test",
    "treatment": "is treated with medication therapy surgery managed drugs relieve cure dose",
    "prevention": "can be prevented avoid vaccine vaccination lifestyle changes reduce the risk",
    "prognosis": "prognosis outcome survival recovery chronic complications fatal long term",
}
ROLE_ORDER = list(SEED)

def _role_protos(bot):
    if not hasattr(bot, "_role_proto_cache"):
        bot._role_proto_cache = np.array([bot._vec(SEED[r]) for r in ROLE_ORDER])
    return bot._role_proto_cache

def _role_oct(bot, centroid96):
    r = int(np.argmax(_role_protos(bot) @ unit(centroid96)))
    return r, ROLE_ORDER[r]

# canonical logic of exposition: the order in which a reasoned account unfolds. Each aspect's
# role is detected from its sentences' cue words; the composition follows this implication chain.
ROLES = [
    ("definition", ["is a", "is an", "are a", "are an", "refers to", "defined", "is the",
                    "type of", "form of", "is characterized", "means that", "known as"]),
    ("cause", ["cause", "caused", "causes", "due to", "results from", "leads to", "lead to",
               "because", "risk factor", "associated with", "triggered", "develop"]),
    ("mechanism", ["occurs when", "happens when", "mechanism", "process", "released",
                   "blocks", "inhibits", "binds", "produced when"]),
    ("symptom", ["symptom", "sign", "signs", "feel", "painful", "swelling", "fever", "fatigue"]),
    ("diagnosis", ["diagnos", "test", "tested", "detect", "examin", "screening", "biopsy",
                   "imaging", "scan", "x-ray", "blood test"]),
    ("treatment", ["treat", "therapy", "medication", "drug", "surgery", "manage", "dose",
                   "prescrib", "relief", "relieve", "cure"]),
    ("prevention", ["prevent", "avoid", "vaccine", "vaccination", "lifestyle", "reduce the risk"]),
    ("prognosis", ["prognosis", "outcome", "survival", "recovery", "complication", "chronic", "fatal"]),
]

def _role(member_sents):
    text = " ".join(member_sents).lower()
    scores = [sum(text.count(c) for c in cues) for _, cues in ROLES]
    if max(scores) == 0: return len(ROLES), "other"
    r = int(np.argmax(scores)); return r, ROLES[r][0]

TEMPLATE = set(("treated treatment treat treats cause causes caused causing symptom symptoms sign "
                "signs diagnosed diagnosis diagnose prevent prevention prevented risk factor factors "
                "work works working affect affects used use uses help helps managed manage").split())

_NAV = ("booklet", "fact sheet", "see the topics", "what i need to know", "espaol",
        "read more", "click here", "for more information")
_RESEARCH = ("to evaluate", "to determine", "to assess", "to investigate", "to compare",
             "to confirm", "the objective", "the aim of", "the purpose of", "this study",
             "this trial", "this analysis", "we evaluated", "we assessed", "we investigated",
             "randomized", "randomised", "multicenter", "multicentre", "post hoc", "double-blind",
             "placebo", "was defined as", "were administered", "were analyzed", "were analysed",
             "hypothesis", "efficacy and safety", "( n =", "p <", "p<", "p =")

def _listy(s):
    # drop navigation/booklet dumps AND research-methodology sentences (jargon, not patient facts)
    low = s.lower()
    if s.count(" - ") >= 2: return True
    return any(x in low for x in _NAV) or any(x in low for x in _RESEARCH)

def spherical_kmeans(Q, k, iters=25, seed=0):
    rng = np.random.default_rng(seed); Q = unit(Q)
    C = Q[rng.choice(len(Q), k, replace=False)].copy()
    for _ in range(iters):
        lab = (Q @ C.T).argmax(1); newC = C.copy()
        for j in range(k):
            m = lab == j
            if m.any(): newC[j] = unit(Q[m].sum(0))
        if np.allclose(newC, C): break
        C = newC
    return lab, unit(C)

def _geodesic_chain(C, start):
    """Order rows of C as a smooth path: greedy nearest-next by cosine on S^7 (slot space)."""
    order, used = [start], {start}
    while len(order) < len(C):
        cur = C[order[-1]]; best, bv = -1, -2.0
        for j in range(len(C)):
            if j in used: continue
            v = float(cur @ C[j])
            if v > bv: bv, best = v, j
        order.append(best); used.add(best)
    return order

def compose(bot, query, k_nbr=80, n_facts=45, max_aspects=7, lex=0.6, w_rel=0.3):
    pe, nn = bot._match(query, k=k_nbr, lex=lex)                       # fuzzy scan of the world model
    seen, pool = set(), []
    for j in nn:
        for s in sents(bot.train[int(j)][1]):
            if s not in seen and len(s.split()) >= 4: seen.add(s); pool.append(s)
    if not pool: return bot.answer(query)
    cv = unit(np.array([bot._vec(s) for s in pool])); qs = cv @ pe
    idx = np.argsort(-qs)[:n_facts]; pool = [pool[i] for i in idx]; cv, qs = cv[idx], qs[idx]
    k = int(min(max_aspects, max(2, len(pool) // 6)))
    lab, C = spherical_kmeans(cv, k, seed=0)                            # cluster -> topic headings (<=7)
    reps = []
    for j in range(k):
        mem = np.where(lab == j)[0]
        if len(mem):
            sc = (cv[mem] @ C[j]) + w_rel * qs[mem]                     # central AND topic-relevant
            reps.append(mem[int(np.argmax(sc))])
    cent = unit(np.array([cv[r] for r in reps]))
    order = _geodesic_chain(cent, int(np.argmax(cent @ pe)))            # relation-consistent fano chain
    return " ".join(pool[reps[o]] for o in order)

def compose_logic(bot, query, k_nbr=80, n_facts=45, max_aspects=7, lex=0.6, w_rel=0.4,
                  role_mode="cue", gate=0.55, entity_boost=0.15, order_mode="role", return_meta=False):
    """INFERENTIAL composition. order_mode 'role' = exposition logic (definition->cause->...->
    treatment); 'coherence' = the no-leakage equiangular chain that maximises prod cos^2 between
    consecutive aspects (paper P Q P = cos^2 theta P). gate drops off-topic aspects."""
    pe, nn = bot._match(query, k=k_nbr, lex=lex)                       # fuzzy scan of the world model
    qt = toks(query, bot.wi); qbg = set(zip(qt, qt[1:]))              # query bigrams (fano-grams)
    vocab = bot.M["vocab"]                                            # salient entity tokens, minus
    ent = set(t for t in qt if bot.idf[t] > 1.0 and vocab[t] not in TEMPLATE)   # template/role words
    seen, pool = set(), []
    for j in nn:
        for s in sents(bot.train[int(j)][1]):
            if s not in seen and len(s.split()) >= 4 and not _listy(s): seen.add(s); pool.append(s)
    if not pool:
        return (bot.answer(query), [], 1.0) if return_meta else bot.answer(query)
    stoks = [set(toks(s, bot.wi)) for s in pool]; sbg = [set(zip(toks(s, bot.wi), toks(s, bot.wi)[1:])) for s in pool]
    # entity-requirement: keep facts that actually mention the query entity (if enough remain) --
    # cuts the topic-confusion bleed (hypertension -> kidney/dialysis sentences)
    if ent:
        keep = [i for i in range(len(pool)) if ent & stoks[i]]
        if len(keep) >= 3:
            pool = [pool[i] for i in keep]; stoks = [stoks[i] for i in keep]; sbg = [sbg[i] for i in keep]
    cv = unit(np.array([bot._vec(s) for s in pool])); qs = cv @ pe
    # fano-gram: prefer facts sharing the query's ORDER-SENSITIVE bigrams ('type 2' over 'type 1')
    if qbg: qs = qs + 0.5 * np.array([len(qbg & b) for b in sbg])
    if entity_boost and ent:
        qs = qs + entity_boost * np.array([1.0 if ent & t else 0.0 for t in stoks])
    idx = np.argsort(-qs)[:n_facts]; pool = [pool[i] for i in idx]; cv, qs = cv[idx], qs[idx]
    k = int(min(max_aspects, max(2, len(pool) // 6)))
    lab, C = spherical_kmeans(cv, k, seed=0)                            # cluster -> topic headings
    aspects = []
    for j in range(k):
        mem = np.where(lab == j)[0]
        if not len(mem): continue
        rep = mem[int(np.argmax((cv[mem] @ C[j]) + w_rel * qs[mem]))]
        relev = float(C[j] @ pe)                                        # heading's topic relevance
        rank, name = _role_oct(bot, C[j]) if role_mode == "oct" else _role([pool[m] for m in mem])
        aspects.append((pool[rep], unit(unit(C[j]).reshape(12, 8).mean(0)), rank, name, relev, unit(C[j])))
    # topic gate: drop off-topic headings (relevance well below the best)
    best = max(a[4] for a in aspects)
    aspects = [a for a in aspects if a[4] >= gate * best] or aspects
    if order_mode == "coherence":                                      # no-leakage equiangular chain
        cen = np.array([a[5] for a in aspects])
        order = _geodesic_chain(cen, int(np.argmax(cen @ pe)))         # greedy max-cos^2 (= max coherence)
    else:                                                              # exposition logic (default)
        order = sorted(range(len(aspects)), key=lambda i: (aspects[i][2], -aspects[i][4]))
    cen = np.array([a[5] for a in aspects])                            # chain coherence = prod cos^2
    coh = float(np.prod([(cen[order[t]] @ cen[order[t + 1]]) ** 2 for t in range(len(order) - 1)])) if len(order) > 1 else 1.0
    # realise the order as composed octonion IMPLIES rotations; verify by modus-ponens search
    eng = INF.InferenceEngine(np.array([aspects[i][1] for i in order]))
    for t in range(len(order) - 1): eng.add_rule(t, t + 1)
    fid = 1.0
    if len(order) >= 2:
        ok, _, f = eng.prove(0, len(order) - 1, max_depth=len(order) + 1)
        if ok: fid = f
    text = " ".join(aspects[i][0] for i in order)
    return (text, [aspects[i][3] for i in order], coh) if return_meta else text

if __name__ == "__main__":
    import json, base64
    from octonion_pr_bot import OctonionPRBot
    from octonion_pr_big import load_full
    pool = load_full() + [tuple(x) for x in json.load(open("mined_qa.json"))]
    bot = OctonionPRBot().fit(pool, vocab_size=24000, verbose=True)
    qs = ["what is gout?", "what causes leg cramps at night?", "can stress cause hair loss?",
          "what are the symptoms of dehydration?", "what is type 2 diabetes?",
          "how can i lower my blood pressure?"]
    out = []
    for q in qs:
        ot, oroles, _ = compose_logic(bot, q, gate=0.0, entity_boost=0.0, return_meta=True)   # no topic-bind
        nt, nroles, fid = compose_logic(bot, q, return_meta=True)       # + gate + entity-bind (cue roles)
        out += ["Q: " + q,
                "  OLD roles: " + " -> ".join(oroles),
                "  OLD: " + ot[:300],
                "  NEW roles: " + " -> ".join(nroles) + ("   [fidelity=%.3f]" % fid),
                "  NEW: " + nt[:330], ""]
    print("B64CMP:" + base64.b64encode("\n".join(out).encode()).decode())
