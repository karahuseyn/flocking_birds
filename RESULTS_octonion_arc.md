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
