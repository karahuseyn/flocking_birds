# octonion_arc_viz.py -- render ARC tasks (train pairs + test input / prediction / gold) as a
# single self-contained SVG using the official ARC colour palette.
import json, numpy as np
import octonion_arc as ARC

PAL = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
       "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"]

def grid_svg(g, x0, y0, cell):
    g = np.array(g); h, w = g.shape; out = []
    out.append('<rect x="%d" y="%d" width="%d" height="%d" fill="#222"/>' % (x0-1, y0-1, w*cell+2, h*cell+2))
    for r in range(h):
        for c in range(w):
            out.append('<rect x="%d" y="%d" width="%d" height="%d" fill="%s" stroke="#222" stroke-width="0.5"/>'
                       % (x0+c*cell, y0+r*cell, cell, cell, PAL[int(g[r, c])]))
    return "".join(out), w*cell, h*cell

def task_panel(tid, task, gold, preds, cell=12):
    """One horizontal strip: train pairs (in->out), then each test (in -> pred / gold)."""
    parts = []; x = 10; y = 30; rowh = 0; W = 0
    def put(g, label, color="#ccc"):
        nonlocal x, rowh
        s, gw, gh = grid_svg(g, x, y, cell)
        parts.append('<text x="%d" y="%d" font-size="11" fill="%s">%s</text>' % (x, y-4, color, label))
        parts.append(s); x += gw + 26; rowh = max(rowh, gh)
    for i, p in enumerate(task["train"]):
        put(p["input"], "train%d in" % i); put(p["output"], "out")
        x += 14
    for i, tp in enumerate(task["test"]):
        put(tp["input"], "TEST%d in" % i, "#fff")
        ok = any(ARC.eq(pp, ARC.A(gold[i])) for pp in preds[i])
        put(preds[i][0], "pred %s" % ("OK" if ok else "X"), "#3f6" if ok else "#f55")
        put(gold[i], "GOLD", "#6cf")
    W = x; H = y + rowh + 16
    head = '<text x="10" y="18" font-size="13" fill="#fff" font-weight="bold">task %s</text>' % tid
    return head + "".join(parts), W, H

def render(tasks, gold_map, pred_map, path):
    panels = []; Y = 0; maxW = 0
    for tid, task in tasks:
        svg, w, h = task_panel(tid, task, gold_map[tid], pred_map[tid])
        panels.append('<g transform="translate(0,%d)">%s</g>' % (Y, svg)); Y += h + 24; maxW = max(maxW, w)
    body = ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" font-family="monospace">'
            '<rect width="100%%" height="100%%" fill="#111"/>%s</svg>' % (maxW + 20, Y + 10, "".join(panels)))
    open(path, "w").write(body); return path

if __name__ == "__main__":
    D = "arc_data/"
    tr = json.load(open(D + "arc-agi_training_challenges.json")); trs = json.load(open(D + "arc-agi_training_solutions.json"))
    ev = json.load(open(D + "arc-agi_evaluation_challenges.json")); evs = json.load(open(D + "arc-agi_evaluation_solutions.json"))
    # find a few SOLVED training tasks (proof the gradient-free solver works) + a few eval failures
    solved = []
    for tid, task in tr.items():
        preds, _ = ARC.solve_task(task); g = trs[tid]
        if all(any(ARC.eq(p, ARC.A(gg)) for p in preds[i]) for i, gg in enumerate(g)):
            solved.append(tid)
        if len(solved) >= 4: break
    chosen = []; gold_map = {}; pred_map = {}
    for tid in solved[:4]:
        chosen.append((tid, tr[tid])); gold_map[tid] = trs[tid]; pred_map[tid] = ARC.solve_task(tr[tid])[0]
    for tid in list(ev)[:3]:
        chosen.append((tid, ev[tid])); gold_map[tid] = evs[tid]; pred_map[tid] = ARC.solve_task(ev[tid])[0]
    p = render(chosen, gold_map, pred_map, "arc_viz.svg")
    print("wrote", p, "with", len(chosen), "tasks (4 solved training + 3 eval)")
