
import torch
from agape.hamiltonian_sde import HamiltonianSDE, SymplecticIntegrator
from agape.utils import set_seed

class Harmonic(torch.nn.Module):
    def forward(self, q):  # V(q) = 0.5 * ||q||^2
        return 0.5 * (q**2).sum(dim=-1)

def test_energy_drift_small():
    set_seed(0)
    q_dim = 4
    sde = HamiltonianSDE(q_dim=q_dim, mass=1.0, sigma=0.0, potential=Harmonic())
    integ = SymplecticIntegrator(sde)
    B = 32
    q = torch.randn(B, q_dim)
    p = torch.randn(B, q_dim)
    H0 = sde.hamiltonian(q, p).mean().item()
    steps = 2000
    dt = 1e-2
    for _ in range(steps):
        q, p = integ.step(q, p, dt)
    H1 = sde.hamiltonian(q, p).mean().item()
    drift_per_100 = abs(H1 - H0) / max(1e-9, abs(H0)) / (steps/100)
    assert drift_per_100 < 1e-3
