# octonion_so8.py -- does the full 28-generator so(8) basis (paper Thm 5.1) learn GENERAL
# SO(8) transports better than the 7 single Fano generators, at a fixed matching-pursuit budget?
import base64, numpy as np
import octonion_transport as T
from octonion_transport import FanoTransport, rotate, alignment, unit

def hidden_SO8(N, K, seed):
    r = np.random.default_rng(seed); Xs = unit(r.standard_normal((N, 8))); Xt = Xs.copy()
    for _ in range(K):                                   # general SO(8) target: full 28-basis
        g = int(r.integers(0, 28)); th = float(r.uniform(-1.2, 1.2)); Xt = rotate(Xt, g, th)
    return Xs, Xt

def learn(active, Xs, Xt, steps):
    T.GEN_IDX = list(active)
    return FanoTransport().fit(Xs[:320], Xt[:320], steps)

if __name__ == "__main__":
    out = ["GENERAL SO(8) TARGET -- held-out alignment  (7 single gens vs full 28 so(8) basis):",
           "  K(hops) steps |  7-gen   28-gen"]
    for K, steps in [(3, 12), (5, 20), (8, 28)]:
        a7 = a28 = 0.0
        for s in range(4):
            Xs, Xt = hidden_SO8(400, K, s); te = slice(320, None)
            f7 = learn(range(7), Xs, Xt, steps); a7 += alignment(f7(Xs[te]), Xt[te])
            f28 = learn(range(28), Xs, Xt, steps); a28 += alignment(f28(Xs[te]), Xt[te])
        out.append("  %5d  %4d  |  %.3f   %.3f" % (K, steps, a7/4, a28/4))
    print("B64SO8:" + base64.b64encode("\n".join(out).encode()).decode())
