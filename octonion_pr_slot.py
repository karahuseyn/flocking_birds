# octonion_pr_slot.py -- prompt->response transport on a MEANING-PRESERVING octonion rep.
#
# The long fano-path product destroyed sentence meaning (verified: fano-NN ~ random). Fix,
# staying octonionic + gradient-free + shortest-path: represent a span by its mean PMI-SVD
# embedding = 12 octonion slots (semantics preserved; plain mean-emb retrieval scores 0.842).
# Learn f: prompt-slots -> answer-slots as a per-slot composition of exact S^7 Fano rotations
# (matching pursuit). "Ayristirma" = spherical k-means partitions prompt space; each region
# gets its own per-slot Fano transport, routed at inference. No backprop anywhere.
import json, base64, time, re
import numpy as np
import octonion_gpt as G
from exp_fano_layer import unit
from octonion_transport import FanoTransport, rotate

def toks(text, wi): return [wi[w] for w in re.findall(r"[a-z']+", text.lower()) if w in wi]
def meanemb(id_lists, emb):
    return np.array([emb[t].mean(0) if t else np.zeros(emb.shape[1]) for t in id_lists])
def slots(E):                                  # (N,96) -> (N,12,8) unit per octonion slot
    S = E.reshape(len(E), 12, 8); return S / (np.linalg.norm(S, axis=2, keepdims=True) + 1e-12)
def align_slots(X, T): return float(np.mean(np.sum(unit(X) * unit(T), axis=2)))

def fit_slot_transport(Xs, Ts, steps=16):      # one FanoTransport per octonion slot
    return [FanoTransport().fit(Xs[:, k, :], Ts[:, k, :], steps) for k in range(Xs.shape[1])]
def apply_slot(F, Xs):
    return np.stack([F[k](Xs[:, k, :]) for k in range(len(F))], axis=1)

if __name__ == "__main__":
    t0 = time.time()
    QA = json.load(open("webmdQAs.json"))
    pairs = [(d["question"], d["answer"]) for d in QA
             if d.get("question") and d.get("answer") and 3 <= len(d["question"].split()) <= 60]
    rng = np.random.default_rng(0); perm = rng.permutation(len(pairs)); pairs = [pairs[i] for i in perm]
    train, test = pairs[:16000], pairs[16000:16400]
    M = G.build("\n".join(q + " " + a for q, a in train), vocab_size=9000, verbose=False)
    wi, emb = M["wi"], M["emb"]
    EPr = meanemb([toks(q, wi) for q, _ in train], emb); EAr = meanemb([toks(a, wi) for _, a in train], emb)
    EPe = meanemb([toks(q, wi) for q, _ in test], emb);  EAe = meanemb([toks(a, wi) for _, a in test], emb)
    Xtr, Ttr = slots(EPr), slots(EAr); Xte, Tte = slots(EPe), slots(EAe)
    EAru = unit(EAr); EAeu = unit(EAe)
    print("built %.0fs train=%d test=%d" % (time.time()-t0, len(train), len(test)))
    out = []

    # (A) global per-slot Fano transport: does prompt-slots -> answer-slots generalise?
    out.append("SLOT-TRANSPORT GENERALISATION (held-out slot alignment to true answer-slots):")
    out.append("  no-transport align(Xte,Tte) = %.3f" % align_slots(Xte, Tte))
    F = fit_slot_transport(Xtr, Ttr, steps=16)
    out.append("  global Fano transport       = %.3f" % align_slots(apply_slot(F, Xte), Tte))

    # (B) "ayristirma": cluster prompt space, per-region per-slot transport, route at inference
    def kmeans_cos(V, k, it=15, seed=0):
        r = np.random.default_rng(seed); C = unit(V)[r.choice(len(V), k, replace=False)]
        for _ in range(it):
            lab = (unit(V) @ C.T).argmax(1)
            C = np.stack([unit(V[lab == j].sum(0)) if (lab == j).any() else C[j] for j in range(k)])
        return lab, C
    for K in [8, 32]:
        lab, C = kmeans_cos(EPr, K, seed=1)
        regF = [fit_slot_transport(Xtr[lab == j], Ttr[lab == j], steps=12) if (lab == j).sum() >= 20 else None
                for j in range(K)]
        rte = (unit(EPe) @ C.T).argmax(1); pred = Xte.copy()
        for j in range(K):
            m = rte == j
            if m.any() and regF[j] is not None: pred[m] = apply_slot(regF[j], Xte[m])
        out.append("  %2d-region routed transport  = %.3f" % (K, align_slots(pred, Tte)))
    out.append("")

    # (C) answer relevance: does transporting the prompt into ANSWER space retrieve better?
    def relev(idx): return float(np.mean(np.sum(EAru[idx] * EAeu, axis=1)))
    rnd = rng.integers(0, len(train), len(test))
    b0 = (unit(EPe) @ unit(EPr).T).argmax(1)                          # prompt->prompt retrieval
    predE = unit(apply_slot(F, Xte).reshape(len(test), 96))          # f(prompt) in answer space
    m3 = (predE @ EAru.T).argmax(1)                                  # f(prompt)->answer retrieval
    out.append("ANSWER RELEVANCE to gold (cos mean-emb; random=%.3f):" % relev(rnd))
    out.append("  B0 prompt->prompt retrieval      = %.3f" % relev(b0))
    out.append("  M3 f(prompt)->answer retrieval   = %.3f" % relev(m3))
    print("B64SLOT:" + base64.b64encode("\n".join(out).encode()).decode())
