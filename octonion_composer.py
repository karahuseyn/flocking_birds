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
        out += ["Q: " + q,
                "  FLAT    : " + bot.answer(q)[:230],
                "  COMPOSE : " + compose(bot, q)[:340], ""]
    print("B64CMP:" + base64.b64encode("\n".join(out).encode()).decode())
