
import torch, math
from agape.mi import InfoNCEEstimator, BarberAgakovEstimator
from agape.utils import set_seed

def analytic_mi_gaussian(beta: float, snr: float) -> float:
    # For S' = S + beta*A + ε, with ε ~ N(0, σ^2 I), A ~ N(0, I) independent
    # I(A; S' | S) = 0.5 * log det(I + beta^2 / σ^2 I) = 0.5 * d * log(1 + beta^2 * snr)
    d = 2
    return 0.5 * d * math.log(1.0 + (beta*beta) * snr)

def make_batch(B=512, d=2, beta=0.8, sigma=0.5):
    set_seed(0)
    S = torch.randn(B, d)
    A = torch.randn(B, d)
    noise = sigma * torch.randn(B, d)
    S_next = S + beta*A + noise
    return A, S_next, S, beta, sigma

def test_infonce_tracks_mi():
    A, S_next, S, beta, sigma = make_batch()
    snr = 1.0/(sigma*sigma)
    true_mi = analytic_mi_gaussian(beta, snr)
    est = InfoNCEEstimator(a_dim=2, s_dim=2, hidden=64)
    opt = torch.optim.Adam(est.parameters(), lr=3e-3)
    for _ in range(400):
        mi = est(A, S_next, S)
        loss = -mi
        opt.zero_grad(); loss.backward(); opt.step()
    mi_val = est(A, S_next, S).item()
    assert abs(mi_val - true_mi)/max(1e-6, true_mi) < 0.2  # <20% rel error

def test_barber_agakov_tracks_mi():
    A, S_next, S, beta, sigma = make_batch()
    snr = 1.0/(sigma*sigma)
    true_mi = analytic_mi_gaussian(beta, snr)
    est = BarberAgakovEstimator(a_dim=2, s_dim=2, hidden=64, fixed_log_std=0.0)
    opt = torch.optim.Adam(est.parameters(), lr=3e-3)
    H = 0.5 * A.size(1) * math.log(2*math.pi*math.e)  # std=1 -> entropy
    for _ in range(600):
        mi = est(A, S_next, S, H_a_given_s=H)
        loss = -mi
        opt.zero_grad(); loss.backward(); opt.step()
    mi_val = est(A, S_next, S, H_a_given_s=H).item()
    assert abs(mi_val - true_mi)/max(1e-6, true_mi) < 0.2
