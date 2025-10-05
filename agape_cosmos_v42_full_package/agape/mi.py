
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class MutualInformationEstimator(nn.Module):
    """Abstract base class for MI estimators I(A; S' | S)."""
    def forward(self, a, s_next, s_cond):
        raise NotImplementedError

class InfoNCEEstimator(MutualInformationEstimator):
    """InfoNCE lower bound with a bilinear critic f(a, s_next | s_cond).
    Returns: estimated MI (in nats).
    """
    def __init__(self, a_dim: int, s_dim: int, hidden: int = 128):
        super().__init__()
        # Build critic g([a, s_cond], s_next)
        self.enc_ctx = nn.Sequential(
            nn.Linear(a_dim + s_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.enc_next = nn.Sequential(
            nn.Linear(s_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.score = nn.Linear(hidden, 1, bias=False)  # dot after hadamard

    def forward(self, a: torch.Tensor, s_next: torch.Tensor, s_cond: torch.Tensor) -> torch.Tensor:
        # a: [B, A], s_next: [B, S], s_cond: [B, S]
        B = a.size(0)
        ctx = torch.cat([a, s_cond], dim=-1)  # [B, A+S]
        h_ctx = self.enc_ctx(ctx)             # [B, H]
        h_next = self.enc_next(s_next)        # [B, H]

        # pairwise scores: f_ij = <h_ctx_i, W h_next_j>
        # compute via outer product trick
        # First project h_next via score.weight^T to get [B, H]
        proj_next = F.linear(h_next, self.score.weight)  # [B, H]
        # similarity matrix [B, B]
        logits = h_ctx @ proj_next.t()
        # InfoNCE with temperature = 1.0
        labels = torch.arange(B, device=logits.device)
        loss = F.cross_entropy(logits, labels, reduction='none')  # NLL
        # MI lower bound = log(B) - NCE loss
        mi = math_log(B) - loss
        return mi.mean()

def math_log(x: int) -> float:
    return float(torch.log(torch.tensor(x, dtype=torch.float32)))

class BarberAgakovEstimator(MutualInformationEstimator):
    """Barber-Agakov variational bound:
        I(A; S' | S) >= E_{p(a, s', s)} [ log q(a | s', s) ] + H(A | S).
    We model q(a|s',s) as diagonal Gaussian with NN-predicted mean and logvar.
    We require either:
      - known conditional entropy H(A|S) (e.g., Gaussian policy with known std),
      - or an estimate passed in as `H_a_given_s`.
    """
    def __init__(self, a_dim: int, s_dim: int, hidden: int = 128, fixed_log_std: Optional[float]=None):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(2*s_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, a_dim)
        self.logvar = nn.Linear(hidden, a_dim)
        self.fixed_log_std = fixed_log_std  # if policy std known: H = 0.5*sum(log(2πe σ^2))

    def forward(self, a: torch.Tensor, s_next: torch.Tensor, s_cond: torch.Tensor, H_a_given_s: Optional[torch.Tensor]=None) -> torch.Tensor:
        h = self.decoder(torch.cat([s_next, s_cond], dim=-1))
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(min=-8.0, max=8.0)
        # log q(a|s',s) for diagonal Gaussian
        log_q = -0.5 * ( (a - mu)**2 / logvar.exp() + logvar + math_log_2pi() )
        log_q = log_q.sum(dim=-1)  # [B]
        if H_a_given_s is None:
            if self.fixed_log_std is None:
                # fallback: estimate entropy from batch std (adds slight bias)
                log_std = 0.5 * torch.log(a.var(dim=0, unbiased=False) + 1e-8)
                H = 0.5 * torch.sum(math_log_2pi() + 2*log_std)
            else:
                H = 0.5 * a.size(-1) * (math_log_2pi() + 2*self.fixed_log_std)
        else:
            H = H_a_given_s
        return (log_q + H).mean()

def math_log_2pi():
    return float(torch.log(torch.tensor(2.0*3.141592653589793)))
