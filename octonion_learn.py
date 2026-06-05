# octonion_learn.py -- LEVER 4: corpus-scale, BACKPROP learning with our own numpy reverse-mode
# autodiff (no torch/jax available). This is the deliberate break from the gradient-free identity:
# we attack the ARC-2 wall the way TRM actually does -- a learned model trained by backprop, here
# regularised by heavy augmentation (the augmentation set is the "corpus" that forces a generalising
# rule rather than per-task memorisation, which we showed the closed-form octonion fit could not do).
# Octonion colour encoding stays as the fixed input embedding (project identity); learned 3x3
# convolutions reason on top. Honest: numpy backprop is slow; the model is small.
import numpy as np

# ============================ minimal reverse-mode autodiff (array-valued) ============================
class T:
    def __init__(self, data, parents=()):
        self.data = np.asarray(data, np.float64); self.grad = np.zeros_like(self.data)
        self._back = lambda: None; self._prev = parents
    def backward(self):
        topo, seen = [], set()
        def build(v):
            if id(v) in seen: return
            seen.add(id(v))
            for p in v._prev: build(p)
            topo.append(v)
        build(self); self.grad = np.ones_like(self.data)
        for v in reversed(topo): v._back()

def _unb(g, shape):                                              # reduce broadcasted grad to shape
    while g.ndim > len(shape): g = g.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1: g = g.sum(i, keepdims=True)
    return g

def add(a, b):
    out = T(a.data + b.data, (a, b))
    def back(): a.grad += _unb(out.grad, a.data.shape); b.grad += _unb(out.grad, b.data.shape)
    out._back = back; return out

def matmul(a, b):
    out = T(a.data @ b.data, (a, b))
    def back():
        a.grad += out.grad @ b.data.swapaxes(-1, -2); b.grad += a.data.swapaxes(-1, -2) @ out.grad
    out._back = back; return out

def relu(a):
    out = T(np.maximum(a.data, 0), (a,))
    def back(): a.grad += (a.data > 0) * out.grad
    out._back = back; return out

def reshape(a, shape):
    out = T(a.data.reshape(shape), (a,))
    def back(): a.grad += out.grad.reshape(a.data.shape)
    out._back = back; return out

# ---- 3x3 conv, NHWC, stride1 pad1, via im2col/col2im ----
def _im2col(x):                                                  # x (N,H,W,C) -> (N,H,W,9C)
    N, H, W, C = x.shape; P = np.zeros((N, H + 2, W + 2, C)); P[:, 1:-1, 1:-1] = x
    cols = [P[:, dr:dr + H, dc:dc + W, :] for dr in range(3) for dc in range(3)]
    return np.concatenate(cols, axis=-1)

def _col2im(g, H, W, C):                                         # g (N,H,W,9C) -> dx (N,H,W,C)
    N = g.shape[0]; P = np.zeros((N, H + 2, W + 2, C)); k = 0
    for dr in range(3):
        for dc in range(3):
            P[:, dr:dr + H, dc:dc + W, :] += g[..., k * C:(k + 1) * C]; k += 1
    return P[:, 1:-1, 1:-1, :]

def conv3(x, Wt, bt):                                            # x (N,H,W,Cin), Wt (9Cin,Cout), bt (Cout,)
    N, H, W, Cin = x.data.shape; cols = _im2col(x.data).reshape(N * H * W, 9 * Cin)
    out_data = (cols @ Wt.data + bt.data).reshape(N, H, W, -1)
    out = T(out_data, (x, Wt, bt))
    def back():
        go = out.grad.reshape(N * H * W, -1)
        Wt.grad += cols.T @ go; bt.grad += go.sum(0)
        dcols = (go @ Wt.data.T).reshape(N, H, W, 9 * Cin)
        x.grad += _col2im(dcols, H, W, Cin)
    out._back = back; return out

def softmax_ce(logits, targets, mask):                          # logits (M,K), targets (M,), mask (M,)
    z = logits.data - logits.data.max(1, keepdims=True); e = np.exp(z); p = e / e.sum(1, keepdims=True)
    M = z.shape[0]; idx = np.arange(M)
    loss = -(np.log(p[idx, targets] + 1e-12) * mask).sum() / max(mask.sum(), 1)
    out = T(loss, (logits,))
    def back():
        g = p.copy(); g[idx, targets] -= 1; g *= (mask / max(mask.sum(), 1))[:, None]
        logits.grad += g * out.grad
    out._back = back; return out

# ============================ Adam ============================
class Adam:
    def __init__(self, params, lr=3e-3):
        self.p = params; self.lr = lr; self.t = 0
        self.m = [np.zeros_like(x.data) for x in params]; self.v = [np.zeros_like(x.data) for x in params]
    def zero(self):
        for x in self.p: x.grad = np.zeros_like(x.data)
    def step(self):
        self.t += 1
        for i, x in enumerate(self.p):
            g = x.grad; self.m[i] = .9 * self.m[i] + .1 * g; self.v[i] = .999 * self.v[i] + .001 * g * g
            mh = self.m[i] / (1 - .9 ** self.t); vh = self.v[i] / (1 - .999 ** self.t)
            x.data -= self.lr * mh / (np.sqrt(vh) + 1e-8)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] and sys.argv[1] == "check":
        rng = np.random.default_rng(0)
        x = T(rng.standard_normal((2, 5, 5, 3))); Wt = T(rng.standard_normal((27, 4)) * .1); bt = T(np.zeros(4))
        tgt = rng.integers(0, 4, 2 * 5 * 5); mask = np.ones(2 * 5 * 5)
        def loss_of():
            o = conv3(x, Wt, bt); return softmax_ce(reshape(o, (2 * 5 * 5, 4)), tgt, mask)
        L = loss_of(); L.backward(); ana = Wt.grad.copy()
        num = np.zeros_like(Wt.data); eps = 1e-5
        for i in range(Wt.data.shape[0]):
            for j in range(Wt.data.shape[1]):
                Wt.data[i, j] += eps; lp = loss_of().data; Wt.data[i, j] -= 2 * eps; lm = loss_of().data
                Wt.data[i, j] += eps; num[i, j] = (lp - lm) / (2 * eps)
        print("grad check conv->ce  max abs err:", np.abs(ana - num).max())
        # check input grad too
        L = loss_of(); L.backward(); axg = x.grad.copy(); numx = np.zeros_like(x.data)
        for idx in [(0,0,0,0),(1,2,3,2),(0,4,4,1)]:
            x.data[idx]+=eps; lp=loss_of().data; x.data[idx]-=2*eps; lm=loss_of().data; x.data[idx]+=eps
            numx[idx]=(lp-lm)/(2*eps)
        print("input grad sample err:", max(abs(axg[i]-numx[i]) for i in [(0,0,0,0),(1,2,3,2),(0,4,4,1)]))
