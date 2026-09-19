"""Transmission experiment: plug-in race vs tilted race vs Bayesian race (posterior draws of Sigma).
Metric: squared distance to the race under the true covariance, averaged over random block markets."""
import numpy as np
from allocation._thurstone.ability import base_density
from allocation._thurstone.calibrate import calibrate_diagonal
from allocation._thurstone.diagonal import diagonal_portfolio
from allocation._thurstone.transport import blend_correlation, transport_weights, nearest_correlation
from allocation._thurstone.covariance import cov_to_corr
rng = np.random.default_rng(0)
n, K, M = 12, 3, 4
lab = np.repeat(np.arange(K), M)
same = lab[:, None] == lab[None, :]
def market():
    F = np.eye(K); g = 0.5 + 0.3*rng.random(K)
    for a in range(K):
        for b in range(a): F[a,b] = F[b,a] = g[a]*g[b]*(0.5+0.3*rng.random())
    vol = 0.12+0.28*rng.random(n); load = 0.4+0.4*rng.random(n)
    C = np.outer(load,load)*F[np.ix_(lab,lab)]; np.fill_diagonal(C, 1.0)
    return C*np.outer(vol,vol)
base = base_density(); seeds = np.random.default_rng(1).standard_normal((4096, n))
_cache = {}
def abilities(tgt):
    k = tuple(np.round(tgt, 12))
    if k not in _cache: _cache[k] = calibrate_diagonal(tgt, base=base)
    return _cache[k]
def race(tgt, Ccorr):
    return transport_weights(abilities(tgt), Ccorr, seeds)
def inv_wishart_draw(nu, Psi, r):
    # Sigma ~ IW(nu, Psi):  Sigma = inv(W), W ~ Wishart(nu, inv(Psi))
    L = np.linalg.cholesky(np.linalg.inv(Psi))
    X = L @ r.standard_normal((n, nu))
    return np.linalg.inv(X @ X.T)
def log_marginal(S, T, nu0, Sigma0):
    # marginal likelihood of the data under N(0, Sigma), Sigma ~ IW(nu0, nu0*Sigma0), up to constants common across nu0
    from scipy.special import multigammaln
    Psi0 = nu0*Sigma0; PsiT = Psi0 + T*S; nuT = nu0 + T
    return (nu0/2)*np.linalg.slogdet(Psi0)[1] - (nuT/2)*np.linalg.slogdet(PsiT)[1] + multigammaln(nuT/2, n) - multigammaln(nu0/2, n)
def bayes_race(S, T, prior, nu0, ndraw=32, r=None):
    Sigma0 = np.where(same, S, 0.0) if prior == 'block' else np.diag(np.diag(S))
    Psi = nu0*Sigma0 + T*S; nu = nu0 + T
    tgt = diagonal_portfolio(Psi/(nu - n - 1))          # abilities from the posterior mean
    w = np.zeros(n)
    for _ in range(ndraw):
        Sk = inv_wishart_draw(nu, Psi, r)
        w += race(tgt, nearest_correlation(cov_to_corr(Sk)))
    return w/ndraw
def pm_race(S, T, prior, nu0):
    Sigma0 = np.where(same, S, 0.0) if prior == 'block' else np.diag(np.diag(S))
    Psi = nu0*Sigma0 + T*S; Sm = Psi/(nu0 + T - n - 1)
    return race(diagonal_portfolio(Sm), nearest_correlation(cov_to_corr(Sm)))
nu_grid = [n+2, 20, 40, 80, 160, 320]
for T in (60, 250):
    res = {}
    nseed = 20
    for s in range(nseed):
        Sig = market(); w_true = race(diagonal_portfolio(Sig), cov_to_corr(Sig))
        X = rng.multivariate_normal(np.zeros(n), Sig, size=T); S = np.cov(X, rowvar=False)
        Ch = cov_to_corr(S); tgt = diagonal_portfolio(S)
        def add(k, w): res.setdefault(k, []).append(np.sum((w - w_true)**2))
        add('plug-in race (phi=1)', race(tgt, Ch))
        add('independent race (phi=0)', race(tgt, np.eye(n)))
        add('block taper s=0', race(tgt, nearest_correlation(np.where(same, Ch, 0.0))))
        best = min((np.sum((race(tgt, blend_correlation(np.eye(n), S, p)) - w_true)**2), p) for p in np.linspace(0,1,11))
        add('tilt, best phi per market (oracle)', race(tgt, blend_correlation(np.eye(n), S, best[1])))
        rr = np.random.default_rng(100+s)
        for prior in ('block', 'diag'):
            # empirical Bayes nu0 by marginal likelihood
            nu_eb = max(nu_grid, key=lambda v: log_marginal(S, T, v, np.where(same, S, 0.0) if prior=='block' else np.diag(np.diag(S))))
            add(f'posterior-mean race, {prior} prior, EB nu0', pm_race(S, T, prior, nu_eb))
            add(f'Bayesian race (32 draws), {prior} prior, EB nu0', bayes_race(S, T, prior, nu_eb, r=rr))
    print(f"\nT={T} observations, {nseed} markets, squared distance to the true race x1e4")
    for k, v in res.items(): print(f"  {k:48s} {np.mean(v)*1e4:6.2f}  (median {np.median(v)*1e4:6.2f})")
