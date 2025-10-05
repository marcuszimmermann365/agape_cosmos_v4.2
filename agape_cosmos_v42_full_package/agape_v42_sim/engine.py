
import math, random
from typing import Dict, Any
import numpy as np
import torch
from agape.mi import InfoNCEEstimator
from agape.hamiltonian_sde import HamiltonianSDE, SymplecticIntegrator

def _set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def _init_jivas(N:int, dim:int=8):
    H = np.random.randn(N, dim).astype(float)  # hidden per jiva
    return H

def _ahimsa_penalty(H_i, H_j, enabled=True):
    delta = np.mean((H_i - H_j)**2)
    return 0.0 if enabled else delta*0.2

def _teleology_weights(mode:str, dim:int):
    if mode == "fixed_uniform":
        return np.ones(dim, dtype=float) / dim
    w = np.ones(dim, dtype=float) / dim
    w[: dim//2] *= 1.6
    w = w / w.sum()
    return w

def _step_sde(H, use_sde: bool, static_reality: bool):
    if static_reality or not use_sde:
        return H
    q = torch.tensor(H[:, :4], dtype=torch.float32)
    p = torch.tensor(H[:, 4:8], dtype=torch.float32)
    sde = HamiltonianSDE(q_dim=4, sigma=0.0)
    integ = SymplecticIntegrator(sde)
    q, p = integ.step(q, p, 1e-2)
    H2 = H.copy()
    H2[:, :4] = q.detach().numpy()
    H2[:, 4:8] = p.detach().numpy()
    return H2

def _compute_empowerment(H_prev, H_next):
    A = torch.tensor(H_next - H_prev, dtype=torch.float32)
    S_next = torch.tensor(H_next, dtype=torch.float32)
    S_cond = torch.tensor(H_prev, dtype=torch.float32)
    est = InfoNCEEstimator(a_dim=A.shape[1], s_dim=S_next.shape[1], hidden=64)
    opt = torch.optim.Adam(est.parameters(), lr=3e-3)
    for _ in range(40):
        mi = est(A, S_next, S_cond)
        loss = -mi
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return float(est(A, S_next, S_cond).item())

def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).flatten()
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(np.abs(x))
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2*(cum.sum()/cum[-1]))/n)

def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    N = int(config.get("N_jivas", 16))
    outer = int(config.get("outer_steps", 100))
    inner = int(config.get("inner_steps", 50))
    use_sde = bool(config.get("use_hamiltonian_sde", True))
    use_mi = bool(config.get("use_mi_empowerment", True))
    teleology_mode = config.get("teleology_mode", "learned")
    disable_ahimsa = bool(config.get("disable_ahimsa", False))
    static_reality = bool(config.get("static_reality", False))
    seed = int(config.get("seed", 0))
    _set_seed(seed)

    H = _init_jivas(N, 8)
    W = _teleology_weights(teleology_mode, H.shape[1])
    last_emp = 0.0
    suffering_series = []

    for _ in range(outer):
        for _ in range(max(1, inner//10)):
            H_next = _step_sde(H, use_sde=use_sde, static_reality=static_reality)
            emp = _compute_empowerment(H, H_next) if use_mi else math.log(np.var(H_next) + 1e-8)
            last_emp = emp
            proj = H_next @ W
            mean_proj = proj.mean()
            harm = np.mean((proj - mean_proj)**2)
            ah = 0.0
            for i in range(min(4, N-1)):
                ah += _ahimsa_penalty(H_next[i], H_next[(i+1)%N], enabled=not disable_ahimsa)
            suffering = harm + ah / max(1, N)
            suffering_series.append(suffering)
            H = H_next

    L_pp_final = float(np.mean(suffering_series[-max(10, len(suffering_series)//5):])) if suffering_series else 0.0
    final_proj = (H @ W)
    per_jiva_suffer = (final_proj - final_proj.mean())**2
    Gini_suffering = _gini(per_jiva_suffer)
    J4_final = float(last_emp)
    dists = []
    for i in range(N):
        for j in range(i+1, N):
            dists.append(np.mean((H[i]-H[j])**2))
    SCM_final = float(1.0/(1.0 + (np.mean(dists) if dists else 0.0)))

    return {"L_pp_final": L_pp_final, "Gini_suffering": Gini_suffering, "J4_final": J4_final, "SCM_final": SCM_final}
