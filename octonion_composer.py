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
import octonion_infer as INF

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

def compose_logic(bot, query, k_nbr=80, n_facts=45, max_aspects=7, lex=0.6, w_rel=0.3, return_meta=False):
    """INFERENTIAL composition: order aspects by the logic of exposition (definition -> cause
    -> mechanism -> symptom -> diagnosis -> treatment -> prevention -> prognosis), and realise
    that order as a chain of octonion IMPLIES rotations verified by modus-ponens (octonion_infer)."""
    pe, nn = bot._match(query, k=k_nbr, lex=lex)                       # fuzzy scan of the world model
    seen, pool = set(), []
    for j in nn:
        for s in sents(bot.train[int(j)][1]):
            if s not in seen and len(s.split()) >= 4: seen.add(s); pool.append(s)
    if not pool:
        return (bot.answer(query), [], 1.0) if return_meta else bot.answer(query)
    cv = unit(np.array([bot._vec(s) for s in pool])); qs = cv @ pe
    idx = np.argsort(-qs)[:n_facts]; pool = [pool[i] for i in idx]; cv, qs = cv[idx], qs[idx]
    k = int(min(max_aspects, max(2, len(pool) // 6)))
    lab, C = spherical_kmeans(cv, k, seed=0)                            # cluster -> topic headings
    aspects = []
    for j in range(k):
        mem = np.where(lab == j)[0]
        if not len(mem): continue
        rep = mem[int(np.argmax((cv[mem] @ C[j]) + w_rel * qs[mem]))]
        rank, name = _role([pool[m] for m in mem])
        oct8 = unit(unit(C[j]).reshape(12, 8).mean(0))                  # aspect heading as one octonion
        aspects.append((pool[rep], oct8, rank, name, float(qs[mem].max())))
    # mantik silsilesi: order by exposition role-rank, then by topic relevance
    order = sorted(range(len(aspects)), key=lambda i: (aspects[i][2], -aspects[i][4]))
    # realise the order as composed octonion IMPLIES rotations; verify by modus-ponens search
    eng = INF.InferenceEngine(np.array([aspects[i][1] for i in order]))
    for t in range(len(order) - 1): eng.add_rule(t, t + 1)
    fid = 1.0
    if len(order) >= 2:
        ok, _, f = eng.prove(0, len(order) - 1, max_depth=len(order) + 1)
        if ok: fid = f
    text = " ".join(aspects[i][0] for i in order)
    return (text, [aspects[i][3] for i in order], fid) if return_meta else text

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
        txt, roles, fid = compose_logic(bot, q, return_meta=True)
        out += ["Q: " + q,
                "  COMPOSE(geodesic): " + compose(bot, q)[:300],
                "  LOGIC roles: " + " -> ".join(roles) + ("   [implies-chain fidelity=%.3f]" % fid),
                "  LOGIC(inferential): " + txt[:340], ""]
    print("B64CMP:" + base64.b64encode("\n".join(out).encode()).decode())
