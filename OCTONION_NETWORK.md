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
