# octonion_grad.py -- LEVER 3, gradient stage with OCTONION NATURAL ALIGNMENT.
#
# TRM brings gradients in only at the final stage; here the final stage is a learned octonionic
# operator that ALIGNS the input field to the output field. The key: octonion multiplication F (x) g
# is LINEAR in the multiplier g (via the left-multiplication matrix L(a), a (x) b = L(a) b). So the
# best octonion-valued 3x3 convolution -- each neighbour tap applies its own learned octonion
# multiplier, summed -- is the gradient OPTIMUM of the field-alignment loss, and that optimum is
# reached in closed form by least squares instead of iterating. This is the "natural alignment":
# the gradient-zero solution is solvable directly because the algebra is linear in the parameters.
# It is the continuous, generalising cousin of the discrete CA rule. Honesty gate is kept: the
# learned operator is accepted only if, after snapping to nearest colour, it reproduces every train
# pair EXACTLY -- so it generalises rather than memorises.
import json, time
import numpy as np
import exp_fano_layer as XF
from octonion_arc import A, eq, bg_color
from octonion_arc import COLOR_OCTON

# octonion left-multiplication structure: L(a) = sum_i a_i M[i],  (a (x) b)[o] = sum_{i,j} a_i M[i,o,j] b_j
_E = np.eye(8)
_M = np.stack([np.stack([XF.octo_mul(_E[i], _E[j]) for j in range(8)], axis=1) for i in range(8)])  # (i,o,j)
def _L(field):                                                    # field (...,8) -> (...,8,8)
    return np.einsum('...i,ioj->...oj', field, _M)

def _encode(g):                                                  # grid -> octonion field (H,W,8)
    return COLOR_OCTON[A(g)]
def _decode(F):                                                  # field -> nearest-colour grid
    d = ((F[..., None, :] - COLOR_OCTON) ** 2).sum(-1); return d.argmin(-1)

def _taps(F):
    """3x3 neighbour shifts (zero-pad) + a constant identity tap (bias). Returns list of (H,W,8)."""
    H, W, _ = F.shape; P = np.zeros((H + 2, W + 2, 8)); P[1:-1, 1:-1] = F
    out = [P[1 + dr:1 + dr + H, 1 + dc:1 + dc + W] for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    bias = np.zeros((H, W, 8)); bias[..., 0] = 1.0                # e0 -> additive octonion constant
    out.append(bias); return out

def _fit(pairs):
    """Solve the octonion-conv multipliers g (n_taps x 8) by least squares over all train cells."""
    N = 10; ATA = np.zeros((8 * N, 8 * N)); ATy = np.zeros(8 * N)
    for i, o in pairs:
        if i.shape != o.shape: return None
        F = _encode(i); Y = _encode(o)
        X = np.concatenate([_L(t) for t in _taps(F)], axis=-1)    # (H,W,8, 8N)
        X = X.reshape(-1, 8, 8 * N); Yf = Y.reshape(-1, 8)
        ATA += np.einsum('coi,coj->ij', X, X); ATy += np.einsum('coi,co->i', X, Yf)
    g = np.linalg.solve(ATA + 1e-6 * np.eye(8 * N), ATy)
    def fn(grid):
        F = _encode(grid); X = np.concatenate([_L(t) for t in _taps(F)], axis=-1)
        pred = (X.reshape(-1, 8, 8 * N) @ g).reshape(F.shape[0], F.shape[1], 8)
        return _decode(pred)
    return fn

def op_octo_align(pairs):
    try: fn = _fit(pairs)
    except Exception: return None
    if fn is None: return None
    return fn if all(eq(A(fn(i)), o) for i, o in pairs) else None

def solve(task):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    fn = op_octo_align(pairs)
    if fn is None: return None
    try: return [A(fn(A(tp["input"]))) for tp in task["test"]]
    except Exception: return None


if __name__ == "__main__":
    import sys
    DIR = "arc_data/"; split = sys.argv[1] if len(sys.argv) > 1 else "training"
    ch = json.load(open(DIR + "arc-agi_%s_challenges.json" % split))
    sol = json.load(open(DIR + "arc-agi_%s_solutions.json" % split))
    t0 = time.time(); solved = []
    for tid, task in ch.items():
        try: preds = solve(task)
        except Exception: preds = None
        if preds is None: continue
        if all(eq(preds[i], A(g)) for i, g in enumerate(sol[tid])): solved.append(tid)
    print("OCTONION-ALIGNMENT gradient operator (closed-form optimum): %d / %d  %s  (%.0fs)"
          % (len(solved), len(ch), split, time.time() - t0))
    print("  ", " ".join(solved[:30]))
