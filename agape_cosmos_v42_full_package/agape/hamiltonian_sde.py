
from typing import Callable, Tuple, Optional
import torch
import torch.nn as nn

Tensor = torch.Tensor

class NeuralPotential(nn.Module):
    """V(q; phi) neural potential."""
    def __init__(self, q_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(q_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, q: Tensor) -> Tensor:
        return self.net(q).squeeze(-1)

class HamiltonianSDE(nn.Module):
    """Hamiltonian SDE under Stratonovich:
        dq =  ∂H/∂p dt
        dp = -∂H/∂q dt + Σ(q) ∘ dW_t
      with H(q,p) = 1/2 * p^T M^{-1} p + V(q)
      Noise acts on momentum only (Langevin-type)."""
    def __init__(self, q_dim: int, mass: float = 1.0, sigma: float = 0.0, potential: Optional[nn.Module]=None):
        super().__init__()
        self.q_dim = q_dim
        self.mass = mass
        self.inv_mass = 1.0 / mass
        self.sigma = sigma
        self.V = potential if potential is not None else NeuralPotential(q_dim)

    def hamiltonian(self, q: Tensor, p: Tensor) -> Tensor:
        kinetic = 0.5 * (p**2).sum(dim=-1) * self.inv_mass
        potential = self.V(q)
        return kinetic + potential

    def drift(self, q: Tensor, p: Tensor) -> Tuple[Tensor, Tensor]:
        q = q.requires_grad_(True)
        V = self.V(q)
        gradV = torch.autograd.grad(V.sum(), q, create_graph=True)[0]
        dqdt = p * self.inv_mass
        dpdt = -gradV
        return dqdt, dpdt

class SymplecticIntegrator:
    """Stochastic symplectic BAOAB/Leapfrog integrator for Hamiltonian SDE.
    When sigma=0, this reduces to standard leapfrog (symplectic, energy-conserving).
    For sigma>0, we add Stratonovich noise in p with midpoint scheme.
    """
    def __init__(self, sde: HamiltonianSDE):
        self.sde = sde

    @torch.no_grad()
    def step(self, q: Tensor, p: Tensor, dt: float) -> Tuple[Tensor, Tensor]:
        sde = self.sde
        # Half-kick (A): p_{n+1/2} = p_n - (dt/2) * dV/dq(q_n)
        q.requires_grad_(True)
        V = sde.V(q).sum()
        gradV = torch.autograd.grad(V, q)[0]
        p = p - 0.5*dt*gradV

        # Drift (B): q_{n+1} = q_n + dt * M^{-1} p_{n+1/2}
        q = q + dt * p * sde.inv_mass

        # Second half-kick (A): p_{n+1} preliminary
        q.requires_grad_(True)
        V2 = sde.V(q).sum()
        gradV2 = torch.autograd.grad(V2, q)[0]
        p = p - 0.5*dt*gradV2

        # Noise (O): Stratonovich noise on momentum via midpoint / additive approx
        if sde.sigma > 0.0 and dt > 0.0:
            # dW ~ N(0, dt); Stratonovich midpoint equals Ito with 0.5 correction for additive
            xi = torch.randn_like(p)
            p = p + sde.sigma * math_sqrt(dt) * xi
        return q, p

def math_sqrt(x: float) -> float:
    return float(torch.sqrt(torch.tensor(x)))
