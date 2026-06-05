# octonion_ttt.py -- corpus/augmentation-regularised BACKPROP learner with test-time training.
# Per task: build a small octonion-embedded conv net, train it by backprop (our numpy autodiff) on
# the task's example pairs with dihedral + translation augmentation (the regulariser that forces a
# generalising rule), then predict the test input with test-time-augmentation voting. This is the
# deliberate move past the gradient-free ceiling -- a learned model, the way TRM attacks ARC.
import json, time, sys
import numpy as np
from octonion_arc import A, eq, COLOR_OCTON
from octonion_learn import T, add, matmul, relu, reshape, conv3, softmax_ce, Adam

S = 30; PAD = 10; K = 11                                          # canvas, pad class, num classes
EMB = np.concatenate([COLOR_OCTON, np.zeros((1, 8))])            # (11,8): colour octons + pad->0
DIH = [lambda g: g, np.fliplr, np.flipud, lambda g: g.T,
       lambda g: np.rot90(g, 1), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 3),
       lambda g: np.rot90(np.fliplr(g))]
INV = [0, 1, 2, 3, 6, 5, 4, 7]                                    # inverse dihedral index

def canvas(g):
    g = A(g); c = np.full((S, S), PAD, int); c[:g.shape[0], :g.shape[1]] = g; return c

def he(shape, fan): return T(np.random.default_rng().standard_normal(shape) * np.sqrt(2.0 / fan))

def model_params(C=32, L=3):
    P = {"w0": he((9 * 8, C), 9 * 8), "b0": T(np.zeros(C))}
    for i in range(L): P["w%d" % (i + 1)] = he((9 * C, C), 9 * C); P["b%d" % (i + 1)] = T(np.zeros(C))
    P["wo"] = he((9 * C, K), 9 * C); P["bo"] = T(np.zeros(K)); P["_L"] = L; return P

def forward(P, class_canvas):                                    # (N,S,S) ints -> logits (N*S*S,K)
    N = class_canvas.shape[0]; x = T(EMB[class_canvas])           # (N,S,S,8)
    h = relu(conv3(x, P["w0"], P["b0"]))
    for i in range(1, P["_L"] + 1):                              # residual conv blocks
        h = relu(add(conv3(h, P["w%d" % i], P["b%d" % i]), h))
    o = conv3(h, P["wo"], P["bo"]); return reshape(o, (N * S * S, K))

def augment(pairs):
    ins, outs = [], []
    for gi, go in pairs:
        for d, fn in enumerate(DIH):
            ins.append(canvas(fn(gi))); outs.append(canvas(fn(go)))
    return np.array(ins), np.array(outs)

def train_task(pairs, steps=250, C=32, L=3, lr=3e-3, seed=0):
    np.random.seed(seed)
    P = model_params(C, L); ps = [v for k, v in P.items() if isinstance(v, T)]
    opt = Adam(ps, lr); Xin, Xout = augment(pairs); tgt = Xout.reshape(-1); mask = np.ones_like(tgt, float)
    for _ in range(steps):
        opt.zero(); logits = forward(P, Xin); loss = softmax_ce(logits, tgt, mask); loss.backward(); opt.step()
    return P

def predict(P, g):                                              # test-time augmentation voting
    H, W = A(g).shape; votes = np.zeros((H, W, K))
    for d, fn in enumerate(DIH):
        gi = fn(A(g)); cc = canvas(gi)[None]
        lg = forward(P, cc).data.reshape(S, S, K)
        pred = lg.argmax(-1)[:gi.shape[0], :gi.shape[1]]
        back = DIH[INV[d]](pred)
        if back.shape == (H, W):
            for r in range(H):
                for c in range(W):
                    votes[r, c, back[r, c]] += 1
    out = votes.argmax(-1)
    return out

def solve(task, steps=250):
    pairs = [(A(p["input"]), A(p["output"])) for p in task["train"]]
    if any(max(g.shape) > S for pr in pairs for g in pr): return None
    P = train_task(pairs, steps=steps)
    preds = []
    for tp in task["test"]:
        if max(A(tp["input"]).shape) > S: return None
        preds.append(A(predict(P, tp["input"])))
    return preds


if __name__ == "__main__":
    DIR = "arc_data/"; split = sys.argv[1] if len(sys.argv) > 1 else "evaluation"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ch = json.load(open(DIR + "arc-agi_%s_challenges.json" % split))
    sol = json.load(open(DIR + "arc-agi_%s_solutions.json" % split))
    items = list(ch.items());
    if n: items = items[:n]
    t0 = time.time(); solved = []
    for tid, task in items:
        try: preds = solve(task)
        except Exception as e: preds = None
        if preds is None: continue
        if all(eq(preds[i], A(g)) for i, g in enumerate(sol[tid])): solved.append(tid)
    print("TTT backprop learner: %d / %d  %s  (%.0fs)  %s"
          % (len(solved), len(items), split, time.time() - t0, " ".join(solved[:20])))
