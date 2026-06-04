# octonion_universe.py -- the octonionic universe for ARC patterns.
#
# Everything is an octonion. A grid is sent to a single CARRIER octonion in S^7, a transformation
# is a single RELATION octonion R, and the whole training corpus becomes a CARRIER SET in which a
# new transformation is detected INSTANTLY (sip diye) by nearest-neighbour in S^7. No gradients,
# no backprop -- only octonion algebra, fano paths and the 7-rule.
#
#   grid -> objects (connected components); each object -> a unit octonion (its physical state:
#           colour, size, shape, position) ......................................... obj_octon
#   objects -> TSP-optimal tour (greedy nearest-neighbour over centroids) ........... tsp_order
#   tour -> a HOLOGRAPHIC superposition: object k is bound to fano-cycle ROLE octon (k mod 7) by
#           the fano-path product, then summed -- a non-diffusing carrier (the 7-rule binds the
#           tour positions) ............................................................ carrier
#   transformation = relation octonion R = carrier_out (x) carrier_in^-1 ........... relation
#   CARRIER SET = {task -> its mean R}; detect(task) = nearest R in S^7 ............ Universe
#
# Why holographic, not a plain fold-product: a left-fold octo_mul DIFFUSES (each new factor rotates
# the whole accumulator, so the carrier forgets early objects and within-task R is unstable). Binding
# each tour position to one of the 7 fano-cycle roles and SUPERPOSING keeps every object legible, so
# the same transformation lands on the same R. Measured within-task R cosine: fold 0.419 -> holo 0.828.
import json, time, zlib, struct, numpy as np
import exp_fano_layer as XF
from octonion_arc import A, bg_color, objects

_rng = np.random.default_rng(7)
COLOR_OCTON = XF.unit(np.concatenate([np.eye(8), _rng.standard_normal((2, 8))]))   # 10 colours -> S^7
# 7 fano-cycle ROLE octonions: tour position k binds to imaginary axis e_{(k mod 7)+1}. The 7-rule
# (one role per fano-plane point) makes the superposition order-aware without diffusing.
ROLE = XF.unit(np.stack([np.roll(np.eye(8)[1], k) for k in range(7)]))

def obj_octon(o, shape):
    """An object -> a unit octonion encoding its physical state (colour, size, aspect, position, fill)."""
    H, W = shape; r0, c0, r1, c1 = o["bbox"]
    cy = (r0 + r1) / 2 / max(1, H - 1); cx = (c0 + c1) / 2 / max(1, W - 1)
    f = np.array([(o["color"] or 0) / 9.0, o["size"] / (H * W), o["h"] / H, o["w"] / W, cy, cx,
                  o["size"] / (o["h"] * o["w"] + 1e-9), o["ncolors"] / 9.0], float)
    return XF.unit(f)

def tsp_order(pts):
    """Greedy nearest-neighbour tour from the canonical top-left-most object (order-stable traversal)."""
    pts = np.array(pts, float); n = len(pts)
    if n <= 2: return list(range(n))
    start = int(np.argmin(pts[:, 0] * 1e4 + pts[:, 1]))
    order = [start]; used = {start}
    for _ in range(n - 1):
        d = np.linalg.norm(pts - pts[order[-1]], axis=1)
        for u in used: d[u] = 1e18
        nxt = int(d.argmin()); order.append(nxt); used.add(nxt)
    return order

def carrier(grid, max_obj=49):
    """Send a grid to its carrier octonion in S^7 (holographic, ROLE-bound along the TSP tour)."""
    g = A(grid); bg = bg_color(g); objs = objects(g, bg, True, False)
    if not objs: return COLOR_OCTON[bg]
    objs = objs[:max_obj]
    cents = [((o["bbox"][0] + o["bbox"][2]) / 2, (o["bbox"][1] + o["bbox"][3]) / 2) for o in objs]
    order = tsp_order(cents)
    acc = np.zeros(8)
    for k, i in enumerate(order):
        acc = acc + XF.octo_mul(ROLE[k % 7], obj_octon(objs[i], g.shape))   # fano-path bind role(x)object
    return XF.unit(acc)                                                     # superpose -> the carrier

def relation(cin, cout):
    """The transformation octonion R that carries c_in to c_out: R = c_out (x) c_in^-1."""
    return XF.unit(XF.octo_mul(cout, XF._inv(cin)))


class Universe:
    """The carrier set: every training task contributes one transformation octonion R. A new task's
    transformation is detected instantly as the nearest R in S^7 -- sip diye tespit."""
    def __init__(self): self.ids = []; self.R = None

    def fit(self, challenges):
        ids, Rs = [], []
        for tid, task in challenges.items():
            rs = [relation(carrier(p["input"]), carrier(p["output"])) for p in task["train"]]
            if len(rs) >= 2:
                ids.append(tid); Rs.append(XF.unit(np.mean(rs, 0)))   # task's mean transformation octon
        self.ids = ids; self.R = np.array(Rs); return self

    def detect(self, task, k=5):
        """Map a task's transformation to the carrier set: return the k nearest known R (instant)."""
        rs = [relation(carrier(p["input"]), carrier(p["output"])) for p in task["train"]]
        q = XF.unit(np.mean(rs, 0)); sim = self.R @ q
        idx = np.argsort(-sim)[:k]
        return [(self.ids[i], float(sim[i])) for i in idx]

    def neighbours(self):
        """For every task, the nearest OTHER task in the carrier set (self excluded). O(n^2) once."""
        S = self.R @ self.R.T; np.fill_diagonal(S, -2.0)
        nn = S.argmax(1)
        return {self.ids[i]: (self.ids[nn[i]], float(S[i, nn[i]])) for i in range(len(self.ids))}


def _family(task):
    """A coarse, octonion-free transformation label (ground truth proxy) to score retrieval against:
    how the output shape relates to the input, and whether it is a pure in-place recolour."""
    i, o = A(task["train"][0]["input"]), A(task["train"][0]["output"])
    if i.shape == o.shape:
        recolour = bool(((i != 0) == (o != 0)).all())          # same occupancy mask -> recolour-only
        return "recolour" if recolour else "same-shape"
    rh = o.shape[0] / i.shape[0]; rw = o.shape[1] / i.shape[1]
    if rh == rw and rh == int(rh): return "scale-%d" % int(rh)
    if o.shape[0] * o.shape[1] < i.shape[0] * i.shape[1]: return "shrink"
    return "grow"


def _heldout_consistency(challenges):
    """Honest validation: for each >=3-train task, predict the held-out pair's R from the OTHERS and
    measure cosine. High = the carrier-set transformation is a stable, reusable octonion (not noise)."""
    cos = []
    for task in challenges.values():
        rs = [relation(carrier(p["input"]), carrier(p["output"])) for p in task["train"]]
        if len(rs) < 3: continue
        rs = np.array(rs)
        for j in range(len(rs)):
            ref = XF.unit(np.delete(rs, j, 0).mean(0))
            cos.append(float(rs[j] @ ref))
    return np.array(cos)


# ---- visualise the octonionic universe as a PNG: carriers projected to 2D, transformations as arrows ----
def _save_png(rgb, path):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    def chunk(typ, data):
        return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", zlib.crc32(typ + data) & 0xffffffff)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))

def _disc(img, y, x, r, col):
    H, W, _ = img.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy*dy + dx*dx <= r*r:
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W: img[yy, xx] = col

def _line(img, y0, x0, y1, x1, col):
    H, W, _ = img.shape; n = int(max(abs(y1-y0), abs(x1-x0)) + 1)
    for t in np.linspace(0, 1, n):
        yy, xx = int(round(y0 + t*(y1-y0))), int(round(x0 + t*(x1-x0)))
        if 0 <= yy < H and 0 <= xx < W: img[yy, xx] = col

def visualise(challenges, path, ntasks=40):
    """A 2D map of the universe: every grid is a carrier point, every train pair an arrow in->out.
    Carriers (8D, S^7) are projected to the plane by their top-2 principal directions (an SVD basis,
    gradient-free). Tasks are coloured distinctly so each transformation reads as a coloured arrow field."""
    pal = np.array([(0,116,217),(255,65,54),(46,204,64),(255,220,0),(240,18,190),(255,133,27),
                    (127,219,255),(135,12,37),(177,13,201),(0,204,163),(255,255,255),(160,160,160)], np.uint8)
    pts_in, pts_out, cols = [], [], []
    for ti, (tid, task) in enumerate(list(challenges.items())[:ntasks]):
        for p in task["train"]:
            try: ci, co = carrier(p["input"]), carrier(p["output"])
            except Exception: continue
            pts_in.append(ci); pts_out.append(co); cols.append(ti % len(pal))
    Cin = np.array(pts_in); Cout = np.array(pts_out)
    allc = np.vstack([Cin, Cout]); mu = allc.mean(0)
    U, S, Vt = np.linalg.svd(allc - mu, full_matrices=False)   # principal directions of the carrier cloud
    B = Vt[:2]                                                  # top-2 -> the universe's 2D map
    Pin = (Cin - mu) @ B.T; Pout = (Cout - mu) @ B.T
    allp = np.vstack([Pin, Pout])
    lo, hi = allp.min(0), allp.max(0); span = np.maximum(hi - lo, 1e-9)
    Wd = Ht = 900; pad = 40
    def to_px(p):
        q = (p - lo) / span
        return int(pad + q[1]*(Ht-2*pad)), int(pad + q[0]*(Wd-2*pad))
    img = np.full((Ht, Wd, 3), 16, np.uint8)
    for i in range(len(Pin)):
        y0, x0 = to_px(Pin[i]); y1, x1 = to_px(Pout[i]); c = pal[cols[i]]
        _line(img, y0, x0, y1, x1, (c.astype(int)*0.5).astype(np.uint8))   # transformation arrow (in->out)
        _disc(img, y0, x0, 4, c); _disc(img, y1, x1, 6, c)                 # in (small) -> out (large)
    _save_png(img, path); return path, len(Pin)


if __name__ == "__main__":
    D = "arc_data/"
    tr = json.load(open(D + "arc-agi_training_challenges.json"))
    t0 = time.time()

    # 1) within-task transformation stability (holographic carrier)
    within = []
    for task in tr.values():
        rs = [relation(carrier(p["input"]), carrier(p["output"])) for p in task["train"]]
        if len(rs) >= 2:
            R = np.array(rs); within.append(float((R @ R.T)[np.triu_indices(len(R), 1)].mean()))
    within = np.array(within)
    print("OCTONIONIC UNIVERSE  (%d tasks, %.0fs)" % (len(within), time.time() - t0))
    print("  within-task transformation cosine (holographic carrier): mean %.3f  median %.3f"
          % (within.mean(), np.median(within)))
    print("  tasks whose transformation is stably one octon (>0.8): %d / %d (%.0f%%)"
          % ((within > 0.8).sum(), len(within), 100 * (within > 0.8).mean()))

    # 2) held-out consistency: is the carrier-set R a reusable transformation, not noise?
    ho = _heldout_consistency(tr)
    print("  held-out pair R cosine (predict a pair from the others): mean %.3f  (>0 random=0)" % ho.mean())

    # 3) the carrier set + instant detection
    uni = Universe().fit(tr)
    print("  carrier set built: %d task transformation-octons in S^7" % len(uni.ids))
    probe = list(tr.items())[0]
    hits = uni.detect(probe[1], k=3)
    print("  detect(%s) sip-diye -> nearest transformations:" % probe[0],
          ", ".join("%s %.2f" % (i, s) for i, s in hits))

    # honest limitation: the carrier is a LOSSY holographic summary, so nearest-R across the whole
    # corpus only weakly recovers a gross SHAPE-family label -- R encodes object-state transport, not
    # output dimensions. Strength is WITHIN a transformation (above), not coarse cross-corpus typing.
    fam = {tid: _family(task) for tid, task in tr.items() if tid in uni.ids}
    nn = uni.neighbours()
    from collections import Counter
    base = max(Counter(fam.values()).values()) / len(fam)
    print("  [limit] nearest-R shares a coarse shape-family only %.0f%% (majority %.0f%%): the carrier"
          % (100 * np.mean([fam[t] == fam[nn[t][0]] for t in fam]), 100 * base))
    print("          summarises object STATE, so detection is strong within a transformation, weak as gross typing")

    # 4) visualise the universe
    p, n = visualise(tr, "octonion_universe.png")
    print("  wrote %s : %d carrier points (grids), arrows = transformations" % (p, n))
