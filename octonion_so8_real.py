# octonion_so8_real.py -- does the 28-generator so(8) basis help the REAL prompt->answer slot
# transport (noisy targets, not exact rotations)? Local kNN transport, 7-gen vs 28-gen.
import json, base64, numpy as np
import octonion_transport as T
from exp_fano_layer import unit
from octonion_pr_bot import OctonionPRBot
from octonion_pr_big import load_full
from octonion_pr_slot import toks, slots, fit_slot_transport, apply_slot, align_slots

if __name__ == "__main__":
    pool = load_full(); test, train = pool[:200], pool[200:]
    bot = OctonionPRBot().fit(train, vocab_size=9000, verbose=True)
    vec = bot._vec
    EPr = np.array([vec(q) for q, _ in train]); EAr = np.array([vec(a) for _, a in train])
    EPe = np.array([vec(q) for q, _ in test]);  EAe = np.array([vec(a) for _, a in test])
    Xtr, Ttr, Xte, Tte = slots(EPr), slots(EAr), slots(EPe), slots(EAe)
    EPru, EAru, EAeu = unit(EPr), unit(EAr), unit(EAe)
    nn = np.argsort(-(unit(EPe) @ EPru.T), axis=1)[:, :60]
    out = ["REAL prompt->answer slot transport (held-out, k=60):  align | relevance"]
    for name, active in [("7 single gens ", range(7)), ("28 so(8) basis", range(28))]:
        T.GEN_IDX = list(active); pred = Xte.copy()
        for i in range(len(test)):
            F = fit_slot_transport(Xtr[nn[i]], Ttr[nn[i]], steps=10); pred[i] = apply_slot(F, Xte[i:i+1])[0]
        al = align_slots(pred, Tte)
        rel = float(np.mean(np.sum(EAru[(unit(pred.reshape(len(test), 96)) @ EAru.T).argmax(1)] * EAeu, axis=1)))
        out.append("  %s   %.3f  | %.3f" % (name, al, rel))
    print("B64R:" + base64.b64encode("\n".join(out).encode()).decode())
