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
- `README.md` — Project description

## Usage
Open `flocking_birds.html` directly in a modern web browser.  
No server or build step is required.

Use the on-screen controls to select the number of birds and start or reset the simulation.