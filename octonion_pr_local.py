# octonion_pr_local.py -- the decisive test: does the LEARNED Fano transport beat lookup
# when lookup cannot cheat?  webmd is paraphrase-dense, so prompt->prompt retrieval wins by
# finding a near-twin.  Here we (a) build a HARD split (drop train prompts that are near-
# duplicates of any test prompt) so retrieval has no twin, and (b) push the "ayristirma" to
# the local limit: a per-test kNN-local per-slot Fano transport (locally-weighted gradient-
# free regression on S^7).  Reuses octonion_pr_slot machinery.
import json, base64, time
import numpy as np
import octonion_gpt as G
from exp_fano_layer import unit
from octonion_transport import FanoTransport
from octonion_pr_slot import toks, meanemb, slots, align_slots, fit_slot_transport, apply_slot

if __name__ == "__main__":
    t0 = time.time()
    QA = json.load(open("webmdQAs.json"))
    pairs = [(d["question"], d["answer"]) for d in QA
             if d.get("question") and d.get("answer") and 3 <= len(d["question"].split()) <= 60]
    rng = np.random.default_rng(0); perm = rng.permutation(len(pairs)); pairs = [pairs[i] for i in perm]
    train, test = pairs[:16000], pairs[16000:16400]
    M = G.build("\n".join(q + " " + a for q, a in train), vocab_size=9000, verbose=False); wi, emb = M["wi"], M["emb"]
    EPr = meanemb([toks(q, wi) for q, _ in train], emb); EAr = meanemb([toks(a, wi) for _, a in train], emb)
    EPe = meanemb([toks(q, wi) for q, _ in test], emb);  EAe = meanemb([toks(a, wi) for _, a in test], emb)
    Xtr, Ttr, Xte, Tte = slots(EPr), slots(EAr), slots(EPe), slots(EAe)
    EAru, EAeu, EPru, EPeu = unit(EAr), unit(EAe), unit(EPr), unit(EPe)
    def relev(idx): return float(np.mean(np.sum(EAru[idx] * EAeu, axis=1)))
    out = []

    # local kNN transport: per test point, fit a per-slot Fano transport on its k nearest
    # train prompts (weighted by proximity) -> the locally-linear limit of the region field.
    def local_transport(k=60, steps=10):
        sims = EPeu @ EPru.T; nn = np.argsort(-sims, axis=1)[:, :k]; pred = Xte.copy()
        for i in range(len(test)):
            idx = nn[i]; F = fit_slot_transport(Xtr[idx], Ttr[idx], steps=steps)
            pred[i] = apply_slot(F, Xte[i:i+1])[0]
        return pred
    predL = local_transport()
    out.append("ALIGNMENT to true answer-slots (held-out, standard split):")
    out.append("  no-transport            = %.3f" % align_slots(Xte, Tte))
    out.append("  local kNN(60) transport = %.3f" % align_slots(predL, Tte))
    predLe = unit(predL.reshape(len(test), 96))
    out.append("  relevance: B0 prompt->prompt = %.3f | local f(prompt)->answer = %.3f"
               % (relev((EPeu @ EPru.T).argmax(1)), relev((predLe @ EAru.T).argmax(1))))
    out.append("")

    # HARD split: drop train prompts that are near-duplicates of ANY test prompt (cos>thr),
    # so prompt->prompt retrieval can no longer find a twin.
    for thr in [0.85, 0.75]:
        maxsim = (EPru @ EPeu.T).max(1); keep = maxsim <= thr
        kr = np.where(keep)[0]
        EPrk, EArk, Xtrk, Ttrk = EPru[kr], EAru[kr], Xtr[kr], Ttr[kr]
        b0 = (EPeu @ EPrk.T).argmax(1); rb0 = float(np.mean(np.sum(EArk[b0] * EAeu, axis=1)))
        # 32-region routed transport on the deduped train
        def kmeans_cos(V, K, it=12, seed=1):
            r = np.random.default_rng(seed); C = unit(V)[r.choice(len(V), K, replace=False)]
            for _ in range(it):
                lab = (unit(V) @ C.T).argmax(1)
                C = np.stack([unit(V[lab == j].sum(0)) if (lab == j).any() else C[j] for j in range(K)])
            return lab, C
        lab, C = kmeans_cos(EPrk, 32)
        regF = [fit_slot_transport(Xtrk[lab == j], Ttrk[lab == j], steps=12) if (lab == j).sum() >= 20 else None
                for j in range(32)]
        rte = (EPeu @ C.T).argmax(1); pred = Xte.copy()
        for j in range(32):
            m = rte == j
            if m.any() and regF[j] is not None: pred[m] = apply_slot(regF[j], Xte[m])
        prede = unit(pred.reshape(len(test), 96)); m3 = (prede @ EArk.T).argmax(1)
        rm3 = float(np.mean(np.sum(EArk[m3] * EAeu, axis=1)))
        out.append("HARD split (drop train cos>%.2f to any test): kept %d/%d train" % (thr, keep.sum(), len(train)))
        out.append("  B0 prompt->prompt retrieval    = %.3f" % rb0)
        out.append("  M3 f(prompt)->answer retrieval = %.3f" % rm3)
    print("built %.0fs" % (time.time()-t0))
    print("B64LOC:" + base64.b64encode("\n".join(out).encode()).decode())
