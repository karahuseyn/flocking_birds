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
| + octonion natural-alignment gradient (closed-form optimum) | octonion_grad | — | 0 |
| **full gradient-free union** | octonion_full | **62** | **0** |

Progression on training: 8 → 36 → 53 → 59 → 62. Double the hand-built operator bank (31).

## Lever 4 — real backprop (our own numpy reverse-mode autodiff)

Deliberate break from the gradient-free identity. octonion_learn.py implements a minimal array
autodiff (gradient-checked vs finite differences to ~1e-11); octonion_ttt.py does per-task
test-time training of a small octonion-embedded residual conv net with dihedral augmentation.

| metric | result |
|---|---|
| ARC-2 evaluation (TRM-comparable) | **0 / 120** |
| ARC-2 training (200-task sample) | 4 / 200 (~2%) |
| of those 4, NOVEL vs the gradient-free union | **3** (0ca9ddb6, 1c0d0a4b, 32597951) |

So backprop is **complementary**, not redundant: it generalises on regular tasks the closed-form
operators miss, and the combined system covers strictly more of the training split than either
alone. But on the hard ARC-2 evaluation it scores 0, the same as every gradient-free mechanism.

## Honest conclusion

* The octonionic machinery genuinely works in pieces — emergent operators solve real tasks with
  zero predefined transforms, and our hand-written backprop learner trains and generalises.
* The ARC-2 **evaluation wall (0/120) is structural**, not a tuning gap. It held against closed-form
  operators, their compositions, learned local rules, an octonion gradient optimum, AND real
  backprop with augmentation.
* Why: every method here learns **per task** from 2–4 examples. TRM's ~5% on ARC-2 eval comes from
  training a high-capacity reasoner **across the whole corpus** (learning which abstractions are
  plausible) and then adapting at test. Per-task-only learning generalises on simple regular tasks
  (hence the ARC-2-training solves) but cannot invent the novel abstractions the evaluation set
  demands.
* The only realistic path to move the eval number is corpus-scale pretraining of one model with an
  in-context mechanism — a much larger build, and matching TRM's numbers in numpy-only is ambitious.

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
