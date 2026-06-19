# exp_octopath.py -- foundation of the long-fano-path idea (remove K=12, keep the trajectory).
# A composition is a LONG chain P_k = P_{k-1} (x) o(token_k) on the unit octonions S^7.
# Two provable properties of octonions as a COMPOSITION algebra (||ab||=||a|| ||b||):
#   (1) non-decaying memory: flipping the token at position p changes the FINAL state by
#       exactly ||o_p - o_p'||, independent of how far back p is (EMA forgets exponentially).
#   (2) exact recovery: o_k = P_{k-1}^{-1} (x) P_k recovers every token from the kept
#       trajectory, losslessly, at any length.
# Demonstrated here vs an EMA state (what the current generator uses). base64-verified.
import base64, time
import numpy as np
import exp_fano_layer as X

def tok_octonion(M):
    """One unit octonion per token (first 8 SVD dims). (A variable-size SET per token,
    the chosen design, removes the collisions this 8-number code still has -- see demo 2.)"""
    o = M["emb"][:, :8].copy()
    return o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)

if __name__ == "__main__":
    t0 = time.time()
    TEXT = open("corpus_books.txt", encoding="utf-8", errors="ignore").read()[:5_000_000]
    M = X.build(TEXT, vocab_size=6000); W = M["W"]; emb = M["emb"]
    O = tok_octonion(M); rng = np.random.default_rng(0)
    print("built %.0fs" % (time.time()-t0))
    ID = np.zeros(8); ID[0] = 1.0

    # ---- DEMO 1: non-decaying long-range memory (chain) vs EMA ----------------
    n = 60; trials = 400; r = 0.18
    # buckets by distance-from-end (n-p): recent / mid / far
    buck = {"recent (0-9)": [], "mid (25-34)": [], "far (50-59)": []}
    bemax = {"recent (0-9)": [], "mid (25-34)": [], "far (50-59)": []}
    def which(dp):
        return "recent (0-9)" if dp < 10 else "mid (25-34)" if 25 <= dp < 35 else "far (50-59)" if 50 <= dp < 60 else None
    for _ in range(trials):
        seq = rng.integers(W, size=n)
        # chain trajectory
        P = ID.copy()
        for w in seq: P = X.octo_mul(P, O[w])
        # EMA state
        S = np.zeros(emb.shape[1])
        for w in seq: S = (1-r)*S + r*emb[w]
        Pf, Sf = P, S
        for p in range(n):
            dp = n-1-p; b = which(dp)
            if b is None: continue
            w2 = int(rng.integers(W))
            if w2 == seq[p]: continue
            # chain final after flipping position p
            P2 = ID.copy()
            for i, w in enumerate(seq):
                P2 = X.octo_mul(P2, O[w2] if i == p else O[w])
            buck[b].append(np.linalg.norm(P2 - Pf))
            # EMA final after flip
            S2 = np.zeros(emb.shape[1])
            for i, w in enumerate(seq):
                S2 = (1-r)*S2 + r*(emb[w2] if i == p else emb[w])
            bemax[b].append(np.linalg.norm(S2 - Sf) / (np.linalg.norm(Sf)+1e-12))
    print("\nDEMO 1 -- final-state change when ONE token is flipped at distance (n-p) from the end:")
    rows = []
    for b in ["recent (0-9)", "mid (25-34)", "far (50-59)"]:
        rows.append("  %-13s  chain=%.3f   EMA(rel)=%.3f" % (b, np.mean(buck[b]), np.mean(bemax[b])))

    # ---- DEMO 2: exact recovery of the whole sequence from the trajectory -----
    rec_ok = 0; rec_tot = 0
    for L in (20, 60, 120, 240):                       # length-independence
        seq = rng.integers(W, size=L); traj = [ID.copy()]
        for w in seq: traj.append(X.octo_mul(traj[-1], O[w]))
        good = 0
        for k in range(L):
            ok = X.octo_mul(X._inv(traj[k]), traj[k+1])          # o_k = P_{k-1}^-1 (x) P_k
            pred = int((O @ (ok/ (np.linalg.norm(ok)+1e-12))).argmax())
            good += (pred == seq[k])
        rows.append("  recover L=%-4d  exact-token=%.3f" % (L, good/L))
        rec_ok += good; rec_tot += L
    out = "DEMO 1 (non-decay) + DEMO 2 (recovery)\n" + "\n".join(rows)
    out += "\n  overall recovery=%.3f  (misses are first-8-dim COLLISIONS, not length)" % (rec_ok/rec_tot)
    print("B64OP:" + base64.b64encode(out.encode()).decode())
