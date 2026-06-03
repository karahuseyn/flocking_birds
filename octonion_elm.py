# octonion_elm.py -- gradient-free Hypercomplex Extreme Learning Machine (MethodsX 2025,
# octonion/quaternion ELM). Random hidden layer + closed-form (ridge pseudo-inverse) output,
# NO backprop. Octonion-ELM uses octonion-structured random hidden weights (block left-mult
# matrices L_w); Real-ELM uses plain Gaussian weights. Compared to matching-pursuit transport
# on the prompt->answer slot mapping (all three gradient-free).
import json, base64, numpy as np
import octonion_transport as T
from exp_fano_layer import unit
from octonion_pr_bot import OctonionPRBot
from octonion_pr_big import load_full
from octonion_pr_slot import slots, fit_slot_transport, apply_slot, align_slots
from octonion_pr_extract import rouge1
from octonion_pr_slot import toks

FG = np.stack(T.FANO_GEN)                                  # (8,8,8) E_0..E_7
def Lw(w):                                                 # octonion w(8,) -> 8x8 left-mult matrix
    return np.einsum('i,ijk->jk', w, FG)

def elm_hidden_matrix(d_oct, L, octonion, seed=0):
    """Build the (8L x 96) random hidden projection. octonion=True -> each (slot,unit) block is
    a left-mult matrix L_w of a random octonion; else plain Gaussian."""
    rng = np.random.default_rng(seed); M = np.zeros((8 * L, 8 * d_oct))
    for l in range(L):
        for i in range(d_oct):
            blk = Lw(rng.standard_normal(8)) if octonion else rng.standard_normal((8, 8))
            M[8*l:8*l+8, 8*i:8*i+8] = blk
    return M / np.sqrt(d_oct)

def elm_fit_predict(Xtr, Ttr, Xte, L=200, octonion=True, lam=1e-2, seed=0):
    d_oct = Xtr.shape[1] // 8
    M = elm_hidden_matrix(d_oct, L, octonion, seed)
    Htr = np.tanh(Xtr @ M.T); Hte = np.tanh(Xte @ M.T)
    Htr = np.hstack([Htr, np.ones((len(Htr), 1))]); Hte = np.hstack([Hte, np.ones((len(Hte), 1))])
    beta = np.linalg.solve(Htr.T @ Htr + lam * np.eye(Htr.shape[1]), Htr.T @ Ttr)   # closed form
    return Hte @ beta

if __name__ == "__main__":
    pool = load_full(); test, train = pool[:200], pool[6200:6200+6000]   # disjoint train/test
    bot = OctonionPRBot().fit(train, vocab_size=12000, verbose=True); vec = bot._vec
    EPr = np.array([vec(q) for q, _ in train]); EAr = np.array([vec(a) for _, a in train])
    EPe = np.array([vec(q) for q, _ in test]);  EAe = np.array([vec(a) for _, a in test])
    EAru, EAeu = unit(EAr), unit(EAe)
    def relev(Y): return float(np.mean(np.sum(EAru[(unit(Y) @ EAru.T).argmax(1)] * EAeu, axis=1)))
    out = ["GRADIENT-FREE transport learners (held-out):  slot-align | relevance"]
    # matching pursuit (local kNN, 28-gen)
    Xtr, Ttr, Xte, Tte = slots(EPr), slots(EAr), slots(EPe), slots(EAe)
    nn = np.argsort(-(unit(EPe) @ unit(EPr).T), axis=1)[:, :60]; T.GEN_IDX = list(range(28))
    predMP = Xte.copy()
    for i in range(len(test)):
        F = fit_slot_transport(Xtr[nn[i]], Ttr[nn[i]], steps=10); predMP[i] = apply_slot(F, Xte[i:i+1])[0]
    out.append("  matching pursuit (28)    %.3f  | %.3f" % (align_slots(predMP, Tte), relev(predMP.reshape(len(test), 96))))
    # ELMs
    for name, octo in [("Real-ELM      ", False), ("Octonion-ELM  ", True)]:
        Y = elm_fit_predict(EPr, EAr, EPe, L=300, octonion=octo)
        out.append("  %s   %.3f  | %.3f" % (name, align_slots(slots(Y), Tte), relev(Y)))
    print("B64ELM:" + base64.b64encode("\n".join(out).encode()).decode())
