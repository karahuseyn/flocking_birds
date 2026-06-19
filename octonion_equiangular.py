# octonion_equiangular.py -- the NO-LEAKAGE / equiangular-projection transport (paper sec 4-5):
# the transition between two octonions x,t is the SO(8) gate V(x,t) = (I + L_x^T L_t)/norm,
# reversible by construction with an intrinsic cos^2 (equiangular) scale -- no matching-pursuit
# fit needed. We build a closed-form transport W = mean_i V(x_i,t_i) over the kNN neighbourhood,
# apply it to the query, and compare held-out slot alignment vs the fitted Fano transport.
import json, base64, numpy as np
import octonion_transport as T
from exp_fano_layer import unit
from octonion_pr_bot import OctonionPRBot
from octonion_pr_big import load_full
from octonion_pr_slot import toks, slots, fit_slot_transport, apply_slot, align_slots

FG = np.stack(T.FANO_GEN)                                  # (8,8,8): E_0..E_7 left-mult matrices
def Lmat(O):                                               # octonions O (M,8) -> left-mult mats (M,8,8)
    return np.einsum('mi,ijk->mjk', O, FG)

def equiangular_predict(Xtr_k, Ttr_k, Xte_k, nn):
    """Per slot: W_i = mean over neighbours of V(x,t)=(I+L_x^T L_t)/2 (unit octonions), pred = W x_q."""
    pred = np.zeros_like(Xte_k); I = np.eye(8)
    for i in range(len(Xte_k)):
        p = nn[i]; Ax = Lmat(Xtr_k[p]); At = Lmat(Ttr_k[p])
        V = (I + np.transpose(Ax, (0, 2, 1)) @ At) / 2.0   # (60,8,8) no-leakage gates
        W = V.mean(0)
        pred[i] = W @ Xte_k[i]
    n = np.linalg.norm(pred, axis=1, keepdims=True)
    return pred / (n + 1e-12)

if __name__ == "__main__":
    pool = load_full(); test, train = pool[:200], pool[200:]
    bot = OctonionPRBot().fit(train, vocab_size=9000, verbose=True); vec = bot._vec
    EPr = np.array([vec(q) for q, _ in train]); EAr = np.array([vec(a) for _, a in train])
    EPe = np.array([vec(q) for q, _ in test]);  EAe = np.array([vec(a) for _, a in test])
    Xtr, Ttr, Xte, Tte = slots(EPr), slots(EAr), slots(EPe), slots(EAe)
    EAru, EAeu = unit(EAr), unit(EAe)
    nn = np.argsort(-(unit(EPe) @ unit(EPr).T), axis=1)[:, :60]
    def relev(predE): return float(np.mean(np.sum(EAru[(unit(predE) @ EAru.T).argmax(1)] * EAeu, axis=1)))
    out = ["TRANSPORT: matching-pursuit (fit) vs equiangular no-leakage (closed form):  align | relevance"]
    # baseline: 28-gen matching pursuit
    T.GEN_IDX = list(range(28)); predMP = Xte.copy()
    for i in range(len(test)):
        F = fit_slot_transport(Xtr[nn[i]], Ttr[nn[i]], steps=10); predMP[i] = apply_slot(F, Xte[i:i+1])[0]
    out.append("  matching pursuit (28)   %.3f  | %.3f" % (align_slots(predMP, Tte), relev(predMP.reshape(len(test), 96))))
    # equiangular no-leakage transport, per slot
    K = Xte.shape[1]; predEA = np.stack([equiangular_predict(Xtr[:, k], Ttr[:, k], Xte[:, k], nn) for k in range(K)], axis=1)
    out.append("  equiangular no-leakage  %.3f  | %.3f" % (align_slots(predEA, Tte), relev(predEA.reshape(len(test), 96))))
    print("B64EA:" + base64.b64encode("\n".join(out).encode()).decode())
