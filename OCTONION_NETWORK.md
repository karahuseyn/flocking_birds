# OKTONYON NETWORK — implementation of the handwritten design

Faithful, end-to-end implementation (`octonion_network.py`) of the five mechanisms in the
hand-drawn spec.

## The design (from the notes)

1. **Octonion neuron.** Each neuron is an octonion carrying a **value in its real component e0**
   and an **identity tag e_k** (k = 1..7): `neuron = value·e0 + e_k`.
2. **Relational, Fano-routed connections.** A connection octon `a` entering a neuron of identity
   `e_b` leaves as `a ⊗ e_b` — the octonion / Fano-plane product routes the signal.
3. **Sign gate.** The octonion product is signed; a connection propagates "straight" on positive
   orientation and "conjugate" (negative) otherwise — the sign the connection carries.
4. **Redundant matrix representation.** A matrix/tensor is laid into the **e0 channel**; the identity
   tag is free, so the *same* matrix has *many* equivalent octonionic encodings (incl. "Okt9"-type
   tags whose e0 is also 0).
5. **Choose / use / propagate.** Operations route the identity tags through the Fano product while the
   e0 value channel stays invariant.

## Verified end-to-end (run `python octonion_network.py`)

- **Fano propagation matches the notes:** `e1 ⊗ e2 = +e3` (the worked path Okt1→Okt2→Okt3), and
  `e1 ⊗ e5 = −e4` — the negative sign is exactly the "conjugate" propagation gate the notes describe.
  A multi-hop walk carries the accumulating sign, e.g. `e1` through `[2,4,1]` → `e3, e7, −e6`.
- **Redundant representation:** the 2×2 identity matrix is encoded four different ways (identity tags
  `[[4,4],[6,7]]`, `[[1,2],[6,7]]`, …) and **all decode to the same matrix** — the value channel is
  invariant to the tag choice, the "rich, many-agent representation" the notes build toward.
- **Operate by propagation:** routing every tag by `e3` permutes the identities
  (`[[4,4],[6,7]] → [[7,7],[5,4]]`) while `decode(before) == decode(after)` — routing the identities
  does not disturb the data.
- **ARC-style grid:** a colour grid encoded two different ways both decode exactly, and tag-routing by
  `e5` then decoding preserves the grid.

All exact octonion algebra; no gradients, no learning — this is the network's mechanism, implemented
and checked. The redundant-representation property is the octonionic realisation of the EML paper's
"one object, many trees", and the Fano-routed sign-gated propagation is the connection rule from the
notes.

## Applied to ARC (octonion_network_arc.py)

The network is applied to ARC end to end through its two channels:
* **e0 value channel = colour** — colour transforms are e0 maps (recolour table).
* **identity/tag channel = geometry** — the dihedral group acts by Fano tag-routing, permuting
  positions while the e0 value channel stays invariant (colours move, unchanged).

A transform = geometry-routing then e0 value-map, solved from the task's pairs and accepted only on
EXACT reproduction; the redundant representations are emitted as ARC's two attempts (pass@2).

**Measured: training pass@1 11/1000, pass@2 11/1000, evaluation 0/120** (<1 s, gradient-free). The
eleven are the dihedral + recolour tasks the two-channel substrate expresses natively (all inside the
129 mechanism union). pass@2 equals pass@1 here because, for this small native operator set, the
redundant representations that fit the demonstrations also agree on the test grid. The octonion network
thus *natively* covers the geometry+colour slice; to reach the full 129 it serves as the SUBSTRATE that
hosts the discrete grid-program mechanisms (octonion_full) — values in e0, routing and selection by the
Fano tags. This is the honest end-to-end application: the network's own mechanism solves the
geometry/colour band, and the broader coverage lives in the repertoire it carries.

## Cellular, recursive ARC solver (octonion_cells.py)

Each base mini-network is a representation CELL; the solver holds THOUSANDS of cells and evolves them
in a cyclic, recursive loop — seed with the input's redundant octon-network representations (dihedral
images), expand by Fano-routed geometry + structural moves to a population cap (~3000 cells), select
the cells closest to the target (gradient-free survival), close the fittest with the mechanism
repertoire; a cell reproducing every demonstration exactly is a solution. Surviving distinct solution
cells give ARC's two attempts.

**Measured: training pass@1 104/1000, pass@2 105/1000, eval 0/120** (~46 min, gradient-free). 104 are
inside the 129 mechanism union; **one is NOVEL — ce039d91 — and it is solved only at pass@2**: its two
candidate cells disagree on the test grid, candidate 0 (pass@1) is wrong and candidate 1 (a redundant
octon-network representation) is correct. So the network's redundant-representation property — the
hand-drawn design's "one object, many encodings" — delivers a genuine new ARC solve through the second
attempt, lifting the combined union to 130. The cellular recursion reaches a strong single-pipeline
score (104 committed) and the redundant cells turn ARC's two-attempt budget into real coverage.
