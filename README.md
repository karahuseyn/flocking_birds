# Flocking Birds – An Octonionic Simulation

This project is an experimental flocking simulation inspired by collective motion models and octonionic representations of alignment.

Birds move inside a 3D bounded space and update their velocities based on:
- Local neighbor alignment (encoded via octonions)
- Cohesion and separation
- Boundary avoidance
- Small stochastic steering

The simulation is based on ideas discussed in:
**arXiv:0709.1916** — Collective motion and alignment dynamics.

## Files
- `flocking_birds.html` — Interactive 3D simulation (Three.js, single-file)
- `flocking_birds.py` — Original Python prototype
- `octonion_tokenizer.html` — Octonionic lossy compression demo (single-file)
- `README.md` — Project description

## Usage
Open `flocking_birds.html` directly in a modern web browser.  
No server or build step is required.

Use the on-screen controls to select the number of birds and start or reset the simulation.

## Octonionic Tokenizer

`octonion_tokenizer.html` reuses the same `Octonion3D` algebra to build an
experimental **lossy compressor** — a kind of tokenizer that turns a byte stream
into octonion tokens.

Pipeline:
1. Bytes are centred (`−128`) and folded 8 at a time into an octonion `O`.
2. `O` is rotated into a transform basis by a unit octonion key `K` derived from a
   seed: `P = O ⊗ K`. Because `|K| = 1`, this rotation preserves the norm.
3. The 8 components of `P` are uniformly quantized to *N* bits over a shared range —
   the lossy step. Each quantized octonion is one **token**.

Decoding dequantizes the components and applies the exact octonion inverse
`P ⊗ K⁻¹ = O` (valid by Artin's theorem, since `O` and `K` generate an associative
subalgebra), then un-centres the bytes.

A slider trades **compression ratio** (≈ `8 / N`) against **fidelity** (byte RMSE).
Because the octonion rotation mixes all 8 bytes of a block together, quality is best
read from the RMSE rather than exact-character accuracy. Open the file in a browser —
no server or build step required.