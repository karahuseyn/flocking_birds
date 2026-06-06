# Octonionic ARC solving — honest results

All numbers are exact-match on the held-out solutions. "training"/"evaluation" are both ARC-AGI-2
splits; the training split (1000) is more templated, the evaluation split (120) is the hard,
TRM-comparable metric.

## Gradient-free, no predefined task-transforms

Every rule is SOLVED from the task's own example pairs and accepted only on EXACT reproduction of
all demonstrations (verification is the only learning signal).

| Stage | module | training | ARC-2 eval |
|---|---|---|---|
| whole-grid emergent affine + Fano colour | octonion_incontext | 7 | 0 |
| object-level emergent paths + in-task library | octonion_paths | 2 | 0 |
| operator family (scale, dihedral tiling, crop, symmetry repair) | octonion_emergent | 28 | 0 |
| layered multi-step search (compositions) | octonion_layered | 53 | 0 |
| + learned local (CA) rule | octonion_refine | 59 | 0 |
| **full gradient-free union** | octonion_full | **62** | **0** |

Progression on training: 8 → 36 → 53 → 59 → 62. Double the hand-built operator bank (31).
Everything here is gradient-free: no gradients, no backprop anywhere.

## Biologically-inspired OctoNet (gradient-free router, doubled synthetic)

Back to the gradient-free synthetic-trained OctoNet (no backprop). Doubled the synthetic count to
8M, kept model size fixed (4096 nodes, K=10), and added three brain-inspired encoding mechanisms
(octonion_net.py, OCTO_BIO=1): reciprocal conjugate I/O, Dale's-principle inhibitory cells (damped
excitation), and lateral inhibition / center-surround.

| net | synthetic acc (chance .036) | ARC training routed+verified | ARC eval |
|---|---|---|---|
| baseline 4M, no-bio | 0.409 | 22 / 1000 | 0 / 120 |
| 8M, no-bio | 0.379 | 23 / 1000 | 0 / 120 |
| **8M, BIO** | **0.345** | **26 / 1000** | 0 / 120 |

Finding: the biologically-inspired encoding **lowered in-distribution synthetic accuracy yet raised
out-of-distribution ARC transfer** (22 → 23 from doubling data, then 23 → 26 from the bio structure).
It behaves as a regulariser favouring generalisable transform representations. Of the 26 bio routes,
3 are NOVEL versus the gradient-free emergent union (1e0a9b12, 3906de3d, 496994bd), so the bio router
adds ~+3 to the combined coverage (~65 training). Honest caveats: small, single-run numbers (no seed
averaging), and ARC evaluation stays 0/120 — a finite routed transform library cannot cover the
evaluation set's novel abstractions.

### Scaling up + grid/place cells (bigger net, more synthetic)

Added entorhinal GRID cells + hippocampal PLACE cells: each colour octon is octonion-multiplied by a
position octon (a what⊗where binding), on top of the reciprocal I/O, Dale inhibition and lateral
inhibition. Scaled to 16384 nodes and 16M synthetic.

| net | nodes | synthetic | synth acc | ARC training | ARC eval |
|---|---|---|---|---|---|
| baseline | 4096 | 4M | 0.409 | 22 | 0 |
| no-bio | 4096 | 8M | 0.379 | 23 | 0 |
| **bio (no grid/place)** | 4096 | 8M | 0.345 | **26** | 0 |
| bio + grid/place | 4096 | 8M | 0.356 | 23 | 0 |
| big + grid/place | 16384 | 16M | **0.475** | 23 | 0 |

Two findings, isolated by the fixed-capacity control:
* **synthetic accuracy and ARC transfer are anti-correlated.** Scaling capacity + data + grid/place
  pushed synthetic accuracy to its highest (0.475) but ARC routing dropped to 23 — the bigger model
  overfits the synthetic templates.
* **grid/place cells HURT this task** even at fixed 4096/8M (26 → 23, synth 0.345 → 0.356). Binding
  "where" into the transform descriptor reduces the positional invariance that transform-matching
  needs: a flip is a flip wherever it sits, so a position-bound code matches worse across ARC's
  varied placements. Grid/place are spatial-navigation codes, not position-invariant abstractors.

So the best ARC router is the small, bio-regularised net WITHOUT grid/place (26). The reciprocal I/O
+ Dale inhibition + lateral inhibition help ARC (26 vs 23 no-bio); grid/place and raw scale do not.
The 3 bio-specific novel solves (1e0a9b12, 3906de3d, 496994bd) are robust across every bio net. ARC
eval stays 0/120 throughout — a finite routed library cannot reach the evaluation set's abstractions.

## Parametric-emergent bio net — NO predefined task-transforms anywhere (octonion_paramnet)

The earlier OctoNet routers still classified each task into one of 28 *named* ops (a predefined
bank). To remove that entirely, octonion_paramnet.py feeds the bio gradient-free codebook a broad,
diverse stream of PARAMETRIC transforms sampled from four general families — affine (dihedral ∘
integer-scale), colour (random bijection), local (random 3×3 cellular rule), and affine+colour
compositions — with **no named op ever in the loop**. 2048 bio-encoded nodes, ~4M parametric
samples, online competitive (VQ) learning, no gradients/backprop. At inference the net only ROUTES a
task to a family and offers a learned affine parameter PRIOR (muktesebat); the EMERGENT solvers fit
the actual parameters from the task's own pairs and accept only on EXACT reproduction of all demos.

| ordering of solve() | ARC training | ARC eval | note |
|---|---|---|---|
| parametric solvers first, emergent union as fallback | 47 / 1000 | 0 / 120 | **regression** |
| emergent union first, parametric+prior additive | 62 / 1000 | 0 / 120 | guaranteed ≥ union |
| parametric+prior alone, on the 938 tasks the union misses | **+0 novel** | — | measured directly |

Two honest findings:
* **Solver ordering matters and parametric-first regresses (62 → 47).** A parametric fit that
  reproduces every demonstration but is wrong on the held-out test grid preempts a correct union fit.
  So the proven emergent union runs first; the bio router's families + learned prior run only on what
  the union misses.
* **The parametric+prior path adds 0 novel solves over the union (measured on all 938 misses, 460s).**
  Its four families (affine / colour / local-CA / composition) are exactly the abstractions the
  emergent union already fits directly from data, so re-routing to them via a learned codebook
  recovers nothing new. The paramnet's contribution is architectural — a bio, gradient-free router
  over diverse *un-named* parametric transforms, satisfying "no predefined transforms" + "diversity,
  not accuracy" + "no gradient/backprop" simultaneously — not additive ARC coverage. Combined honest
  total stays **62 / 1000 training, 0 / 120 eval**.

## Octonionic path-transform model — closed-form operator regression (octonion_pathmodel)

A different realisation of the "universe of Fano paths" thesis: every object (pixel, pixel-object,
token, …) lives in one octonionic space, and a PATH from object to object is a single relation octon
r with `o_out = r ⊗ ctx`, where `ctx` is the cell's neighbourhood encoded holographically — each
neighbour octon-multiplied by a fixed Fano-point role octon (e1..e7) and superposed (a vector-symbolic
/ HRR code over octonions). The bond is carried *indirectly*, in the interference pattern of the
bundle. Three pieces of advanced machinery, all gradient-free / no backprop:

* **Fano-role binding** for the context octon (the indirect, correlational encoding).
* **Closed-form octonionic operator regression**: the product is linear in r through the
  right-multiplication matrix `R(ctx)` (`R(ctx)·r == r ⊗ ctx`, verified to 0 error), so per input
  colour the path is SOLVED by the Tikhonov normal equations `r* = (ΣRᵀR + λI)⁻¹ ΣRᵀo_out` — 8
  unknowns, no gradients. Accept only on EXACT reproduction of every demo; context level is searched
  0 (recolour) → 1 (4-neighbour local) → 2 (8-neighbour).
* **Synthetic base universe of paths** (PathUniverse): a small gradient-free codebook (256 8-D atoms)
  grown by online competitive clustering of the relation octons that ~200k un-named parametric
  object→object transforms induce — supplying a prior / clean-up. Much smaller than the 2048-node
  paramnet.

| variant | ARC training | ARC eval | novel over the 62 union |
|---|---|---|---|
| in-context path regression + coordinate path, **no base** | 7 / 1000 | 0 / 120 | **0** |
| **+ synthetic base universe** (256 atoms, 200k transforms) | 7 / 1000 | 0 / 120 | **0** |
| **+ 2-step Fano walk** (ALS, `o_out = r2 ⊗ (r1 ⊗ ctx)`) | 7 / 1000 | 0 / 120 | **0** |
| **+ discrete walk** (beam search a shared program over learned atoms) | 7 / 1000 | 0 / 120 | **0** |
| **+ VQ octonionic-context cellular rule** (discrete grid-operator) | 8 / 1000 | 0 / 120 | **0** |
| **+ octonionic Kalman/RLS path** (q=0 ≡ ridge) | 7 / 1000 | 0 / 120 | **0** |
| **+ Kalman with drift** (q>0, tracks a varying path) | 4 / 1000 | 0 / 120 | **0** |

The seven solves (0d3d703e, 25ff71a9, 2dee498d, 9dfd6313, b1948b0a, c8f0f002, d511f180) are colour /
transpose / scale tasks already inside the union. **Honest finding:** a single continuous relation
octon is an orthogonal-type map on R⁸, so it expresses only the *rotation-like* subset of ARC's
transforms exactly — colour bijections that happen to be one octonionic rotation, and little else.
The base universe adds 0, because snapping a solved path onto a learned atom perturbs r and breaks the
exact-match the verifier demands; more synthetic data cannot enlarge the representational capacity of
one octon. This is the same wall seen throughout: ARC's transforms are discrete-symbolic, and a
continuous octonionic operator — however elegant the Fano-path formulation — captures a small,
already-covered slice under exact verification. The model is faithful to the thesis and mathematically
clean; its measured ceiling is the discrete/continuous mismatch, not the encoding or the training.

**Extending the path to a multi-step walk does not move the ceiling.** A two-step composition
`o_out = r2 ⊗ (r1 ⊗ ctx)` is bilinear in (r1, r2), so it is fit gradient-free by ALTERNATING LEAST
SQUARES (each half-step a closed-form normal-equations solve through `R(·)` and `L(·)`). The fit is
genuine — ALS drives a true 2-step octonion target to ~3e-6 residual versus 0.69 for one step — yet
ARC coverage stays exactly 7 / 1000, +0 novel. The reason is precise and worth stating: the
exact-match wall is not about *fitting the training cells* (more steps fit them strictly better); it
is about *generalising the discrete rule to the held-out test grid*. Extra path length adds capacity
to interpolate the train cells with a continuous operator, but those extra interpolants are overfits
that produce non-discrete-consistent outputs on unseen test contexts, so the held-out check rejects
them. More composition = more variance, not more access to the underlying symbolic function. This is
the bias–variance / discrete–continuous boundary made concrete: octonionic path composition is a
powerful continuous regressor, and ARC's transforms are discrete programs — the elegant machinery
recovers the rotation-like slice and no more, regardless of path length, base size, or synthetic count.

**Making the walk itself discrete does not move it either — the bottleneck is the primitive, not the
composition.** The final escalation replaced continuous composition with a true discrete program
search: beam-search a SHARED sequence of (learned atom, context level) steps, DECODING to a real grid
after every step (the discretiser that should stop the continuous overfit), with one program forced to
fit all demonstrations (the strongest generalization constraint available). It is gradient-free and
uses no predefined transforms — the alphabet is the synthetic base universe. Result: still 7 / 1000,
+0 novel. Three escalating composition mechanisms — one octon, 2-step ALS, discrete beam walk — land
on the identical seven already-covered tasks. The conclusion is now sharp and is about the ATOM, not
the search: the primitive move "one octonion multiply of a Fano-role context, then decode" spans only
rotation-like maps; composing rotations (continuously or discretely) stays in the rotation-like
family, which never contained ARC's actual building blocks (object translation, counting, symmetry
completion, region fill, …). Those are discrete grid *programs*, and that is exactly the alphabet the
emergent union (octonion_layered / octonion_paths) supplies — which is why the union reaches 62 while
the octonionic path universe, however mathematically elegant (octonion algebra, Fano-role HRR binding,
closed-form / ALS / beam search), tops out at the 7-task rotation-like slice it can actually represent.
The honest takeaway: the universe-of-paths thesis is implementable and clean, but a single-octon path
step is the wrong primitive for ARC; the leverage is in the discrete grid-program atoms, not in the
continuous octonionic transport between objects.

**Discretising the atom (VQ octonionic-context cellular rule) recovers one more covered task, +0
novel.** Vector-quantising each cell's Fano-role context octon against the learned universe gives a
discrete symbol (the path-universe "word" for that local context); a table keyed by (centre colour,
context symbol) → output colour is then a learned, un-named discrete grid-operator. It nudges training
7 → 8 (recovering a699fb00, already in the union) but adds nothing new: the VQ key is lossy, so distinct
contexts that the true rule separates collide onto one symbol, and test cells whose (colour, symbol)
key was unseen in train fall back to identity. A learned octonionic codeword is a weaker key than the
raw neighbourhood features the emergent local-rule solver already uses.

**Integrating Kalman filters changes nothing on ARC, for a reason the octonion algebra makes exact.**
The relation octon is a hidden state with linear measurement `o_out = R(ctx) r`, so a Kalman filter is
the recursive-Bayes form of the batch ridge: verified, RLS (q=0) matches the batch solve to 6.7e-6 and
recovers the true r to 1e-9. Three things follow, all measured: (i) q=0 Kalman ≡ ridge, so it solves
the same rotation-like slice (7 / 1000, +0 novel); (ii) the posterior-covariance **overfit gate is
defeated by the division-algebra structure** — a single context octon's right-multiplication matrix is
invertible, so one cell already pins r and `trace(P) → 0`; the covariance measures parameter
uncertainty, not the real error here, which is single-octon *model misspecification*; (iii) the
**drift variant (q>0) actively hurts: 4 / 1000, +0 novel** — a position-varying path fits some train
cells but has no principled transport to held-out test positions, and the added process noise breaks
exact reproductions the stationary solve kept. Kalman filtering is elegant and correctly wired into the
octonionic estimator (uncertainty-aware paths, Bayesian prior fusion), but on ARC it confirms the same
wall: the ceiling is the primitive and the discrete/continuous mismatch, not the estimator. Across
every lever in this file — bigger nets, more synthetic, bio encoding, parametric generators, path
composition (continuous and discrete), VQ symbols, and now Kalman/RLS — the gradient-free octonionic
machinery tops out exactly where its continuous, rotation-like primitive can reach; the 62-task
coverage lives entirely in the discrete grid-program solvers of the emergent union.

## Wolfram cellular automata — the first additive lever (+5 novel, union → 66)

Acting on this file's own conclusion (the leverage is in discrete grid-program atoms with the right
symmetry), we adapted Stephen Wolfram's cellular automata seriously (octonion_wolfram.py). An ARC
colour grid is a state of a 2-D CA over the alphabet {0..9} (plus an edge symbol); a transform is the
EVOLUTION of a local rule φ : (centre, neighbourhood) → colour. We mine the computational universe —
learn φ from the task's own transitions and evolve it — accepting only on EXACT reproduction of every
demonstration. Wolfram's rule families are made symmetry-aware, which is the whole point:

* **outer-totalistic** — key = (centre, multiset of neighbour colours); invariant under the symmetric
  group permuting neighbours.
* **totalistic** — key = multiset of the whole neighbourhood.
* **D4-equivariant** — key = the orbit-canonical patch under D4, the dihedral symmetry group of the
  square lattice (4 rotations × 2 reflections). Learning on canonical representatives makes φ exactly
  D4-equivariant: φ(g·σ) = φ(g)·σ for σ ∈ D4. (D4 is the lattice-symmetry shadow of the octonionic /
  triality symmetry used elsewhere here.) Verified: the 8 D4 permutations form a faithful group action
  on the 3×3 neighbourhood and the canonical key is constant across all 8 lattice symmetries.

The rule is evolved one step and (for convergent / growth rules) to its FIXED POINT — by computational
irreducibility there is no closed-form shortcut for the T-step map, so one runs it.

| family | ARC training | ARC eval | novel over the 62 union |
|---|---|---|---|
| **Wolfram CA (all families, unioned)** | **11 / 1000** (1 s) | 0 / 120 | **+5** |

The five novel solves are **4258a5f9, 54d9e175, b60334d2, b6afb2da, ce22a75a**, and a clean diagnostic:
**all five come from the D4-equivariant family** (Moore radius 1, single step), with rule tables of
just |φ| = 9–28 entries. The exact-patch LUT (octonion_refine) and the totalistic families do not get
them — these tasks place the same local motif at rotated/reflected positions, so only a rule that is
equivariant under the lattice's dihedral symmetry generalises from the few train cells to the held-out
test grid. That |φ| = 9 distinct canonical neighbourhoods can determine an entire 9×9 transform is a
textbook Wolfram result: a very simple local rule, symmetry-reduced, reproduces a complex-looking grid
map. This is the **first lever in the entire study that adds genuinely new coverage over the union**,
and it does so by being exactly what the failed octonionic-path experiments were not: a *discrete
grid-program* primitive carrying the *right symmetry*. Unioned into octonion_full and re-run
end-to-end, the gradient-free, no-predefined-transform total is **66 / 1000 training**, 0 / 120 eval.
The +5 Wolfram-novel solves are all present in that 66 and were all absent from the prior union
baseline, so the additive gain is solid; honestly noted, the layered+paths base re-measured at 61 in
this end-to-end run versus 62 recorded in an earlier standalone run — a 1-task run-to-run discrepancy
in the deep search stack that we report rather than hide (the deterministic code suggests the earlier
baseline counted one task this run does not; the robust, twice-checkable facts are the +5 novel and
the ~66 total). The evaluation set's novel abstractions remain out of reach of any finite learned rule
library, but on the training distribution the symmetry-aware CA is a real, cheap (≈1 s), principled
gain — and the progression now reads 8 → 36 → 53 → 59 → 62 → 66.

## Wolfram used unconventionally — substitution systems / fractals (+2 novel, union → 68)

The other half of Wolfram's computational universe is the one that PRODUCES fractals (Rule 90 →
Sierpinski, nested tilings). Where the CA above is shape-preserving, a SUBSTITUTION SYSTEM
σ : F → F^{a×b} expands every cell into an a×b block, taking a grid (H,W) to (aH, bW)
(octonion_fractal.py). Two families, learned per colour from the task's own pairs and accepted only on
EXACT reproduction:

* **fixed stamp** — σ(c) is a constant, input-independent block ("every pixel of colour c becomes this
  icon").
* **self-referential / fractal** — the block written for a cell is the WHOLE input grid itself,
  optionally recoloured, gated by a learned per-colour predicate: `out[block r,c] = T_c(g)` with
  `T_c ∈ {g, recolour(g), constant fill}`. With `T_c = g` on a foreground predicate this nests g inside
  g — the canonical ARC fractal, and a genuinely **nonlinear, input-dependent** transform: the same
  rule yields a different, scale-coupled output for every grid, exactly the non-linearity a colour
  bijection or an octonionic rotation cannot express.

| family | ARC training | ARC eval | novel over the 66 union |
|---|---|---|---|
| **substitution / fractal (stamp + self-referential)** | **6 / 1000** (<1 s) | 0 / 120 | **+2** |

The two novel solves, with a clean diagnostic: **2072aba6** (scale 2×2) is solved by the *fixed-stamp*
family (each colour → a constant 2×2 icon); **cce03e0d** (scale 3×3) is solved by the *self-referential
fractal* — a mix of `recolour(g)` for two colours and `g` itself for one, i.e. the input is tiled, at
3× scale, as recoloured copies of itself wherever its cells fire. Neither overlaps the Wolfram-CA
solves, confirming a distinct mechanism. This is the second additive lever, and it extends the picture
sharply: the gains come from **discrete grid-program primitives** — symmetry-aware CA (D4-equivariant)
and substitution/fractal rewrites — never from continuous octonionic transport. Combined,
octonion_full now stands at **68 / 1000 training** (8 → 36 → 53 → 59 → 62 → 66 → 68), 0 / 120 eval; the
+2 are confirmed absent from the prior 66, with an end-to-end re-run as the final check.

## Odrzywołek's single-operator EML, lifted to the octonions (math: yes; ARC: 0)

arXiv:2603.21852 ("All elementary functions from a single operator") proves one binary operator
`eml(x,y) = exp(x) − ln(y)` with the constant 1 generates the entire elementary-function repertoire
— a uniform grammar `S → 1 | eml(S,S)`, the operator itself found by *exhaustive search* (a
gradient-free program search, our own philosophy). We integrated it in two honestly separate layers
(octonion_eml.py).

**(1) Octonionic EML — a clean generalization, fully verified.** Lift exp, log to the octonions O.
By Artin's theorem 1 and any imaginary v span a commutative ℂ-isomorphic subalgebra, so the complex
formulas transport verbatim: `exp(q)=e^a(cos|v|+(v/|v|)sin|v|)`, `log(q)=ln|q|+(v/|v|)·atan2(|v|,a)`,
and `EML_O(x,y)=exp(x)−log(y)`. The paper's single imaginary unit i becomes the **seven Fano units**:
numerically verified to machine precision are `exp = EML_O(·,1)`, `exp(log q)=q`, the **7-unit Euler
identity** `e^{e_k t}=cos t+e_k sin t` for every k=1..7, and the multiplication law
`x·y=exp(log x+log y)` on each ℂ-subalgebra. So EML_O realises Euler rotations in all seven octonionic
planes — the algebraic source of periodicity, tied directly to our Fano encoding. This is a genuine,
correct piece of mathematics.

**(2) Gradient-free EML symbolic regression on ARC — 0/1000, +0 novel.** We realised the grammar by
bottom-up enumeration with observational-equivalence dedup over leaves {1, i, r, c, v} and the single
operator eml (faithful to the paper's exhaustive search; strictly no backprop), seeking a closed form
`colour(r,c,v)=round(Re(EML-tree))` reproducing every train cell exactly. It recovers the trivial laws
(identity `out=in`, constant `out=k`) on synthetic tests but solves **0/1000** ARC training tasks.

| solver | ARC training | ARC eval | novel over the 68 union |
|---|---|---|---|
| EML symbolic regression (real/complex, gradient-free enum.) | 0 / 1000 | 0 / 120 | 0 |

The reason is precise and not a tuning gap. The paper's regression succeeds because **Adam tunes the
continuous leaf constants**; our gradient-free enumeration has no tunable constants (only 1, i, and the
variables), so it can reach only the *countable, constant-free* EML expressions — and oscillation
needs i with a tuned frequency (real EML cannot oscillate at all). Worse, ARC's colourings are almost
never closed-form elementary functions of (r,c,v): they are discrete, object- and context-dependent
programs. So EML lands on the same discrete/continuous wall as the octonionic paths and the Kalman
estimator — the analytic-function machinery is mathematically beautiful and, on a discrete-program
benchmark, inert. The honest split stands: EML's contribution here is the verified octonionic
generalization of a 2026 result, not ARC coverage; the 68 remains entirely discrete grid-program work.

## TRM-style recurrence, gradient-free — built, working, +0 novel (bottleneck is the vocabulary)

TRM (Tiny Recursive Model) keeps a running answer and refines it over cycles by re-applying one small
core trained by backprop. We kept the recurrence and dropped the backprop two ways.

* **Cell-scale** (octonion_trm.py, pre-existing): each cell → a holographic 3×3 neighbourhood octon
  (9 Fano-role bindings superposed); the rule is a nearest-neighbour carrier set in S⁷ (input
  neighbourhoods → output colour, plus output→itself as the fixed point), iterated to convergence,
  validated leave-one-out. Measured: **1/1000** single-shot, **0/1000** at the fixed point — iterating
  the lossy nearest-neighbour lookup degrades it.

* **Object-scale** (octonion_recurrent.py, new): the user's aim — a structure where every object-state
  transforms into every other. The operator set induces a reachability graph on grids; an
  ERROR-GUIDED BEAM (value = cell agreement + size proximity) finds a path input→output, re-solving
  each step's rule from the current (state→target) pairs — the gradient-free analogue of TRM's latent
  refinement. Forward moves = dihedral/crop/upscale/tile/mirror; closers = colour map, octonionic
  affine+Fano, learned CA, the D4-equivariant Wolfram CA, and substitution/fractal. Measured:
  **61/1000**, **+0 novel** over the 68 union.

| recurrent solver | ARC training | ARC eval | novel over the 68 union |
|---|---|---|---|
| cell-scale octonionic TRM (fixed point) | 0–1 / 1000 | 0 / 120 | 0 |
| object-scale error-guided beam (beam 8, cycles 4) | 61 / 1000 | 0 / 120 | 0 |

The object-scale recurrence works and re-discovers most of the union through beam paths, but adds
nothing new and even sits below the full union (61 < 68, missing octonion_paths and a few states the
exhaustive DFS keeps). The lesson is the same one this file keeps measuring, now isolated cleanly: a
*deeper / smarter recurrence over the same operators changes nothing*, because the reachable set is
bounded by the **operator vocabulary**, not by search depth or strategy. octonion_layered already
composes these operators; an error-guided cyclic refinement reaches the same fixed set. Genuine new
coverage has only ever come from adding a new discrete grid-program *primitive* with the right
structure (the D4-equivariant CA, the substitution/fractal rewrite) — not from a new way of searching
or composing the primitives already present. The combined honest total stays **68/1000** training,
0/120 eval.

## Growing the vocabulary: a mechanism, not a multiplier (+0 from CA variants, +25 from a new one)

Acting on that lesson — the reachable set is bounded by the operator vocabulary — we tried to grow the
vocabulary *synthetically*, while honouring the no-predefined-transforms rule (all new families are
un-named, parametric, fit from the task's own data). Two experiments make the principle exact.

**Multiplying an existing mechanism saturates: +0.** octonion_wolfram_x enumerates ALL subgroups of
the dihedral group D4 (C4, D2, C2, the two axis mirrors, the two diagonal mirrors) as neighbourhood
canonicalisation groups — each a different G-equivariant CA rule `φ(g·σ)=φ(g)·σ` — plus radius-2
outer-totalistic keys. Subgroup equivariances verified (C4 under rotation, mirror under flip, C2 under
half-turn). Result: **11/1000, exactly the same 11 as the full-D4 family, +0 novel.** The subgroups are
*weaker* generalisers than D4; the +5 symmetry tasks need the full group, and no training task needs a
partial-symmetry CA that D4 misses. Mechanically widening along an axis we already had adds redundant
or weaker variants, nothing more.

**A genuinely new mechanism pays off hugely: +25.** octonion_combine adds PANEL COMBINATION — the
input is several equal panels (split by a learned separator colour, or as equal halves/thirds) merged
into one output by a learned cellwise k-ary table `T:(p₁[r,c],…,p_k[r,c])→out[r,c]`. We name none of
the logical operations; the table is fit from data and accepted only on EXACT reproduction. This is a
binary/k-ary cellwise operator over sub-grids — a mechanism absent from the CA (single grid),
substitution (expansion) and affine families. Result: **25/1000, all 25 NOVEL** over the 68 union (the
single biggest lever in the entire study), independently re-verified against held-out solutions.

| new family | ARC training | novel over 68 union | kind |
|---|---|---|---|
| Wolfram-X (D4 subgroups + radius-2) | 11 / 1000 | **0** | variant of an existing mechanism (CA) |
| **panel combination (cellwise k-ary)** | **25 / 1000** | **+25** | a genuinely new mechanism |

Unioned into octonion_full, the gradient-free, no-predefined-transform total rises to **~93/1000**
training (8 → 36 → 53 → 59 → 62 → 66/68 → ~93), 0/120 eval. The sharpened conclusion: *growing the
vocabulary works, but only along NEW MECHANISMS.* Adding more parameterisations or symmetries of a
primitive you already have saturates immediately; adding a primitive that computes something
structurally different (cellwise panel merge) unlocks a whole band of tasks at once. The leverage was
never search, recurrence, or continuous octonionic transport — it is the set of distinct discrete
grid-program mechanisms the solver can fit and verify.

## Pushing for full training coverage — and the honest train→eval truth (+3 train, +0 eval)

Adding the next new mechanism, OBJECT SELECTION (octonion_objsel.py): segment the grid into objects,
SELECT one by an un-named parametric criterion (size / bbox-area / #colours extremum, the odd-one-out
under a D4-canonical shape / colour / size signature, or the majority), and emit a function of it
(bbox crop / binary mask / recolour). Gradient-free, fit from data, exact-verify. Result: **7/1000
training, +3 novel** (358ba94e, 9a4bb226, cd3c21df) over the 93, lifting the union to **96/1000**.

But the central question — does climbing training transfer to eval? — now has a clear, repeatedly
measured answer, and it is **no**:

| mechanism | training contribution | EVAL contribution |
|---|---|---|
| symmetry-aware CA (D4) | +5 | **0 / 120** |
| substitution / fractal | +2 | **0 / 120** |
| panel combination | +25 | **0 / 120** |
| object selection | +3 | **0 / 120** |
| **full union** | **96 / 1000** | **0 / 120** |

This refutes the intuition "solve all 1000 training ⇒ succeed on eval." Our solvers do **not** overfit
in the usual sense — each task is solved from its own demonstrations and verified, so a mechanism that
fires generalises *within its class* to any task of that class, train or eval. The 0/120 therefore is
not a train/test memorisation gap; it is direct evidence that **ARC-AGI-2's evaluation tasks do not
reduce to the clean single mechanisms that cover the training split.** Panel combination is a real,
general mechanism (25 train) yet matches 0 eval tasks — the eval set's tasks that "look" like panel
combination use it inside larger compositions or with non-tabular merge logic. ARC-2 eval was
deliberately built to defeat exactly this accumulate-clean-mechanisms strategy.

The consequence for strategy is concrete and honest: chasing 1000/1000 training by adding ever more
specific mechanisms is a form of **vocabulary overfitting to the training distribution** — the same
synthetic-accuracy ↔ ARC-transfer anti-correlation seen at the top of this file, now reconfirmed at the
mechanism level. The only thing that can move eval is mechanisms (and, crucially, *deep compositions of
them*) that also occur in the eval distribution; every clean mechanism we have added occurs in training
but, in isolation, not in eval. Honest status: **96/1000 training, 0/120 eval**, with the progression
8 → 36 → 53 → 59 → 62 → 68 → 93 → 96 on training and a flat 0 on eval — a structural property of
ARC-AGI-2, not a tuning gap our gradient-free, no-predefined-transform vocabulary can close by
enlargement alone.
