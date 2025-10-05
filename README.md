[README.md](https://github.com/user-attachments/files/22706422/README.md)

# Agape Cosmos V4.2 — Theory-Faithful Core

This package provides **theory-faithful** engines required by Agape Cosmos V4.2:
- A **Mutual-Information estimator** for J4 (Empowerment):
  - `InfoNCEEstimator` (contrastive)
  - `BarberAgakovEstimator` (variational with Gaussian q)
- A **Hamiltonian Neural-SDE** with a **symplectic (leapfrog/BAOAB) integrator**.

## Quickstart
```bash
pip install torch
pytest -q
```
The tests include:
- Synthetic MI with known ground truth (relative error < 20%)
- Energy conservation drift of harmonic oscillator (< 1e-3 per 100 steps)

## Integration Notes (V4.1 -> V4.2)
- Replace your J4 proxy with one of the estimators in `agape.mi`.
- Split hidden state `h` into `(q, p)` to step with `HamiltonianSDE`.
- Use `SymplecticIntegrator.step(q, p, dt)` in your inner loop.
