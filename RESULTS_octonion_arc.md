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

## Wolfram cellular automata — the first additive lever (+5 novel, 62 → 67)

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
grid-program* primitive carrying the *right symmetry*. Unioned into octonion_full, the gradient-free,
no-predefined-transform total is **67 / 1000 training** (8 → 36 → 53 → 59 → 62 → 67), 0 / 120 eval —
the evaluation set's novel abstractions remain out of reach of any finite learned rule library, but on
the training distribution the symmetry-aware CA is a real, cheap (1 s), and principled gain.
