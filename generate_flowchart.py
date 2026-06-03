# generate_flowchart.py -- emit a self-contained SVG flowchart of the octonion pipeline,
# with the actual computations shown in each stage. No external deps.
def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

STAGES = [
 ("1. BUILD  -  octonion world model M   (offline, gradient-free)", "#1f4e79", "#dce6f2", [
    "PMI-SVD embeddings:        emb in R^(W x 96)",
    "token octonion:            o(w) = unit(emb[w][1:9])  in S^7",
    "IDF weight:                idf[w] = log((N+1)/(df[w]+1)) + 1",
    "inverted indices:          post[t] -> docs ;  bpost[(a,b)] -> docs   (fano-grams)",
 ]),
 ("2. ENCODE  -  text -> representations", "#2e6b34", "#d9ead3", [
    "IDF mean-emb:              v(x) = unit( SUM_{t in x} idf[t] . emb[t] )    (96-d)",
    "octonion slots:            X = reshape(v, (12, 8))      (12 octonions)",
    "fano-gram (bigram):        fg(a,b) = o(a) (x) o(b)      ((x) = octonion product, order-sensitive)",
 ]),
 ("3. MATCH  -  prompt P' -> k nearest neighbours", "#8a5a00", "#fce5cd", [
    "dense:     d_i   = v(P') . v(P_i)",
    "lexical:   L_i   = SUM_{t in P'} idf[t] . 1[t in P_i]            (entity-weighted overlap)",
    "fano-gram: B_i   = | bigrams(P') ∩ bigrams(P_i) |               (order-sensitive)",
    "score_i = d_i + 0.6 . L^_i + 3.0 . B^_i      ->   nn = argtop-k(score)",
 ]),
 ("4. TRANSPORT  -  local Fano path  ->  answer-region anchor", "#7030a0", "#e6d6f2", [
    "Fano rotation (exact S^7):  R_g(theta) = cos(theta).I + sin(theta).L_g ,   L_g x = e_g (x) x",
    "matching pursuit (no grad): A=SUM<x_i,t_i>, B=SUM<L_g x_i, t_i>,  theta* = atan2(B, A)",
    "fit F on { slots(P_i) -> slots(A_i) }_{i in nn}   (per-slot, greedy compose)",
    "anchor:    g = unit( F( slots(P') ) )       (predicted answer region)",
 ]),
 ("5. COMPOSE  -  compositional construction  (mantik silsilesi)", "#9c2b2b", "#f4cccc", [
    "fuzzy scan:   facts = sents( answers[nn] ) ,  drop _listy (nav + research jargon)",
    "topic bind:   keep facts containing entity ;  qs += 0.5 . |bigrams ∩|   (fano-gram)",
    "cluster:      spherical k-means -> C_1..C_k ,  k <= 7        (RULE OF 7 / ayristirma)",
    "role:         r_j = argmax( cue-count | prototype . C_j )   {def->cause->...->treatment}",
    "order:        sort aspects by (role_rank, -relevance)       (exposition logic chain)",
    "verify:       IMPLIES  R_t = C_{t+1} (x) C_t^-1 ;  prove(0,k-1) -> fidelity (=1.000)",
    "realize:      stitch each aspect's representative sentence in chain order",
 ]),
 ("6. DECODE (flat)  -  extractive answer + honesty signal", "#444444", "#e8e8e8", [
    "score(s) = 0.6(c.g) + 0.2(c.cen) + 0.2(c.v_q) + 0.15.nbr      (c = sentence vec)",
    "topic gate: drop s if (c . v_q) < tau . max ;   MMR select m sentences",
    "confidence: conf = v(P') . v(P_nn1)    ->   'I don't know' if conf < threshold",
 ]),
]

LINE_H, PAD, TITLE_H, BOX_W, X0, GAP = 19, 10, 28, 880, 40, 34
boxes, y = [], 50
for title, hc, bc, lines in STAGES:
    h = TITLE_H + PAD + len(lines) * LINE_H + PAD
    boxes.append((title, hc, bc, lines, y, h)); y += h + GAP
H = y + 30; W = BOX_W + 2 * X0

svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" font-family="monospace">' % (W, H)]
svg.append('<defs><marker id="a" markerWidth="10" markerHeight="10" refX="6" refY="3" orient="auto">'
           '<path d="M0,0 L6,3 L0,6 Z" fill="#333"/></marker></defs>')
svg.append('<rect width="%d" height="%d" fill="white"/>' % (W, H))
svg.append('<text x="%d" y="28" font-size="20" font-weight="bold" fill="#111">Octonionic gradient-free pipeline &#8212; prompt &#8594; composed answer</text>' % X0)
for title, hc, bc, lines, by, h in boxes:
    svg.append('<rect x="%d" y="%d" width="%d" height="%d" rx="8" fill="%s" stroke="%s" stroke-width="1.5"/>' % (X0, by, BOX_W, h, bc, hc))
    svg.append('<rect x="%d" y="%d" width="%d" height="%d" rx="8" fill="%s"/>' % (X0, by, BOX_W, TITLE_H, hc))
    svg.append('<text x="%d" y="%d" font-size="14" font-weight="bold" fill="white">%s</text>' % (X0 + 12, by + 19, esc(title)))
    ly = by + TITLE_H + PAD + 13
    for ln in lines:
        svg.append('<text x="%d" y="%d" font-size="12.5" fill="#1a1a1a">%s</text>' % (X0 + 16, ly, esc(ln))); ly += LINE_H
for i in range(len(boxes) - 1):
    y1 = boxes[i][4] + boxes[i][5]; y2 = boxes[i + 1][4]; cx = X0 + BOX_W // 2
    svg.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333" stroke-width="2" marker-end="url(#a)"/>' % (cx, y1, cx, y2 - 4))
svg.append('</svg>')
open("octonion_pipeline.svg", "w").write("\n".join(svg))
print("wrote octonion_pipeline.svg  (%d x %d)" % (W, H))
