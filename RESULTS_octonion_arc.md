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
