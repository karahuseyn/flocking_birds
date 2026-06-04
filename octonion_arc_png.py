# octonion_arc_png.py -- render ARC tasks to real PNG images (no matplotlib): train pairs +
# test input / our prediction (red frame) / gold (green frame), ARC palette. Pure stdlib PNG.
import json, zlib, struct
import numpy as np
import octonion_arc as ARC

PAL = np.array([(0,0,0),(0,116,217),(255,65,54),(46,204,64),(255,220,0),
                (170,170,170),(240,18,190),(255,133,27),(127,219,255),(135,12,37)], np.uint8)

def save_png(rgb, path):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    def chunk(typ, data):
        return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", zlib.crc32(typ + data) & 0xffffffff)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    open(path, "wb").write(sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))

def grid_rgb(g, cell):
    g = np.array(g); h, w = g.shape
    img = PAL[g].repeat(cell, 0).repeat(cell, 1)                      # colour each cell
    img[::cell] = 60; img[:, ::cell] = 60                             # thin gridlines
    return img

def place(canvas, img, y, x, frame=None):
    if frame is not None:
        canvas[y-2:y+img.shape[0]+2, x-2:x+img.shape[1]+2] = frame
    canvas[y:y+img.shape[0], x:x+img.shape[1]] = img

def render_task(tid, task, gold, preds, cell=12, path=None):
    BG = np.array([24, 24, 24], np.uint8)
    imgs_top = []                                                     # train pairs: in,out,in,out,...
    for p in task["train"]:
        imgs_top += [grid_rgb(p["input"], cell), grid_rgb(p["output"], cell)]
    imgs_bot = []                                                     # test: in, pred, gold (per test input)
    frames_bot = []
    for i, tp in enumerate(task["test"]):
        ok = any(ARC.eq(pp, ARC.A(gold[i])) for pp in preds[i])
        imgs_bot += [grid_rgb(tp["input"], cell), grid_rgb(preds[i][0], cell), grid_rgb(gold[i], cell)]
        frames_bot += [(255,255,255), (60,255,90) if ok else (255,60,60), (90,160,255)]
    gap = 14; pad = 16
    def rowdim(imgs): return (max((im.shape[0] for im in imgs), default=0),
                              sum(im.shape[1] for im in imgs) + gap*len(imgs))
    h1, w1 = rowdim(imgs_top); h2, w2 = rowdim(imgs_bot)
    Wd = max(w1, w2) + 2*pad; Ht = h1 + h2 + 3*pad + 4
    canvas = np.zeros((Ht, Wd, 3), np.uint8); canvas[:] = BG
    x = pad; y = pad
    for im in imgs_top: place(canvas, im, y, x); x += im.shape[1] + gap
    canvas[y+h1+pad-2:y+h1+pad] = (80, 80, 80)                        # separator
    x = pad; y2 = y + h1 + 2*pad
    for im, fr in zip(imgs_bot, frames_bot): place(canvas, im, y2, x, fr); x += im.shape[1] + gap
    if path: save_png(canvas, path)
    return path

if __name__ == "__main__":
    D = "arc_data/"
    ev = json.load(open(D + "arc-agi_evaluation_challenges.json")); evs = json.load(open(D + "arc-agi_evaluation_solutions.json"))
    ids = list(ev)[:4]
    out = []
    for tid in ids:
        preds, _ = ARC.solve_task(ev[tid])
        p = render_task(tid, ev[tid], evs[tid], preds, path="arc_eval_%s.png" % tid)
        out.append(p)
    print("wrote:", out)
