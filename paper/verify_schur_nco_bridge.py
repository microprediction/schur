"""Companion verification script for schur-nco-bridge.tex (a Schur bridge from NCO to unconstrained minimum variance).

Each check_* function returns a dict of measured errors. `main` asserts them.
Run with `python verify_schur_nco_bridge.py`; it needs only NumPy and the standard library.

  check_sufficiency      Prop 1: complement against other knots = against everything
  check_bridge_right     Prop 2(2,3): gamma = 1 stacks to Sigma^{-1} u, unnormalized
                         and through the two-tier recipe
  check_bridge_left      Prop 2(1): gamma = 0 equals an independent NCO
  check_violation        all of the above fail when the gateway model is violated
  check_diagnostic       regressing on other knots false-accepts; R_j does not
  check_change_of_vars   covariance-only rules: apply to M, map back by x = y / b
  check_proxy            Prop 3: V' Sigma V = K Sigma_PP K + D
  check_symmetric        R_j(c) = (1 - rho_c) lambda_O Cov(f_j, r_{-j})
  check_quotient         complements (and companion vectors) compose, so tree = flat at gamma = 1
  check_loss             Delta_i is PSD, zero under the model, and not tracked by the raw size of R
  check_precision_sparsity  Gaussian factorization: zero precision between J_i and assets outside I_i
  check_unnormalized     w ∝ D(D'SD)^{-1}D'u equals the two-tier form; survives a zero cluster total
  check_interior_example      exact rationals: unique interior optimum near 0.5587 (3/1210, 5/1452)
  check_exact_jets       exact G'(1), V0'(1), V0''(1) for both examples by bivariate jets
  check_full_coupling_example exact-rational spot check on a grid (not a proof of global optimality)
  check_endpoint_shift   gamma~(tau) = 1 - G'(1)/V0''(1) tau^2
  check_incremental_cost Phi = gamma tau^2 H with H bounded
  check_zero_direction   a vanishing block of Sigma^{-1}u at gamma = 1 still yields the optimum
  check_interior_zero_direction  the continuous extension at an interior vanishing (issue 11)
  check_scale_invariance rescaling Sigma or u leaves the weights unchanged (issue 12)
  check_sharpe_sign      fully invested Sigma^{-1}mu is max Sharpe only when its total is positive
  check_only_knots_move  member part of d_i is constant in gamma; knot exposure in closed form
  check_equicorrelated_knots  closed-form kappa(gamma) for equicorrelated knots
  check_degenerate_partitions singletons or one cluster: the bridge is the GMV for every gamma
  check_identical_clusters    knot share t/(t+delta) and the quadratic V(x)-V(x*), exact
  check_two_thirds       gamma* = 2/3 with gaps 1/576 and 1/900; the general closed form, exact
  check_sign_threshold   (8 - delta)/(1 + 13 delta/4) for k = 10, c = 1/4, by exact jets
  check_compression      a cluster is its knot plus one independent asset of variance 1/delta_i
  check_unit_betas       unit betas make the portfolio independent of gamma
  check_effective_damping kappa_i = (1-lambda_i)/s_i + lambda_i h_i with lambda_i = gamma r_i/(1-gamma+gamma r_i)
  check_lost_precision   V0 - V* = L/(Z(Z-L)); kappa - h in closed form; curvature exactly, via jets
  check_gain_order       the gain from tuning gamma is G'(1)^2/(2 V0''(1)) tau^4 to leading order
  check_rank_deficiency  duplicated knots (3/8, 1/2, 1/3), the exposure path, non-commuting limits,
                         and noise that must preserve the null space
"""
import numpy as np

TOL = 1e-10


def random_spd(rng, n, scale=1.0):
    a = rng.standard_normal((n, n))
    return scale * (a @ a.T / n + np.eye(n))


def clusters(sizes):
    idx, pos = [], 0
    for m in sizes:
        idx.append(list(range(pos, pos + 1 + m)))
        pos += 1 + m
    return idx, [I[0] for I in idx]


def gateway_model_cov(rng, sizes, violation=0.0):
    """Index 0 of each cluster is its knot; sizes[j] remaining members follow.

    violation > 0 correlates member residuals across clusters while keeping
    them orthogonal to every knot.
    """
    k = len(sizes)
    idx, knots = clusters(sizes)
    n = idx[-1][-1] + 1
    scc = random_spd(rng, k)
    betas = [rng.standard_normal(m) * 0.8 + 1.0 for m in sizes]
    eps = [random_spd(rng, m, 0.3) for m in sizes]
    sigma = np.zeros((n, n))
    for j, Ij in enumerate(idx):
        Oj = Ij[1:]
        for l, Il in enumerate(idx):
            Ol = Il[1:]
            s = scc[j, l]
            sigma[knots[j], knots[l]] = s
            sigma[np.ix_(Oj, [knots[l]])] = (betas[j] * s)[:, None]
            sigma[np.ix_([knots[j]], Ol)] = (betas[l] * s)[None, :]
            sigma[np.ix_(Oj, Ol)] = np.outer(betas[j], betas[l]) * s
        sigma[np.ix_(Oj, Oj)] += eps[j]
    if violation > 0:
        allO = [i for I in idx for i in I[1:]]
        sigma[np.ix_(allO, allO)] += violation * random_spd(rng, len(allO))
    assert np.all(np.linalg.eigvalsh(sigma) > 0)
    return sigma, idx, knots


def full_pair(sigma, I, u):
    rest = [i for i in range(sigma.shape[0]) if i not in I]
    coef = sigma[np.ix_(I, rest)] @ np.linalg.inv(sigma[np.ix_(rest, rest)])
    return sigma[np.ix_(I, I)] - coef @ sigma[np.ix_(rest, I)], u[I] - coef @ u[rest]


def cheap_pair(sigma, I, other_knots, gamma, u):
    S_ic = sigma[np.ix_(I, other_knots)]
    coef = gamma * S_ic @ np.linalg.inv(sigma[np.ix_(other_knots, other_knots)])
    return sigma[np.ix_(I, I)] - coef @ S_ic.T, u[I] - coef @ u[other_knots]


def directions(sigma, idx, knots, gamma, u):
    n = sigma.shape[0]
    d = np.zeros(n)
    for i, I in enumerate(idx):
        other = [c for j, c in enumerate(knots) if j != i]
        Q, b = cheap_pair(sigma, I, other, gamma, u)
        d[I] = np.linalg.solve(Q, b)
    return d


def bridge(sigma, idx, knots, gamma, u):
    n = sigma.shape[0]
    d = directions(sigma, idx, knots, gamma, u)
    V = np.zeros((n, len(idx)))
    for i, I in enumerate(idx):
        V[I, i] = d[I] / d[I].sum()
    a = np.linalg.solve(V.T @ sigma @ V, V.T @ u)
    return V @ a, V


def nco(sigma, idx):
    """Independent NCO: min-var inside each block, min-var across cluster portfolios."""
    n = sigma.shape[0]
    V = np.zeros((n, len(idx)))
    for i, I in enumerate(idx):
        x = np.linalg.solve(sigma[np.ix_(I, I)], np.ones(len(I)))
        V[I, i] = x / x.sum()
    a = np.linalg.solve(V.T @ sigma @ V, np.ones(len(idx)))
    return V @ (a / a.sum())


def _random_problem(rng, violation=0.0):
    sizes = list(rng.integers(1, 6, size=rng.integers(2, 7)))
    sigma, idx, knots = gateway_model_cov(rng, sizes, violation)
    mu = rng.standard_normal(sigma.shape[0]) + 3.0
    return sigma, idx, knots, mu


def check_sufficiency(rng, trials=100, violation=0.0):
    err = 0.0
    for _ in range(trials):
        sigma, idx, knots, mu = _random_problem(rng, violation)
        for u in (np.ones(len(mu)), mu):
            for i, I in enumerate(idx):
                other = [c for j, c in enumerate(knots) if j != i]
                Qf, bf = full_pair(sigma, I, u)
                Qc, bc = cheap_pair(sigma, I, other, 1.0, u)
                err = max(err, np.abs(Qf - Qc).max(), np.abs(bf - bc).max())
    return {"pair": err}


def check_bridge_right(rng, trials=100, violation=0.0):
    stacked = recipe = 0.0
    for _ in range(trials):
        sigma, idx, knots, mu = _random_problem(rng, violation)
        for u in (np.ones(len(mu)), mu):
            target = np.linalg.solve(sigma, u)
            stacked = max(stacked, np.abs(directions(sigma, idx, knots, 1.0, u) - target).max())
            w, _ = bridge(sigma, idx, knots, 1.0, u)
            recipe = max(recipe, np.abs(w / w.sum() - target / target.sum()).max())
    return {"stacked": stacked, "recipe": recipe}


def check_bridge_left(rng, trials=100):
    err = 0.0
    for _ in range(trials):
        sigma, idx, knots, _ = _random_problem(rng, violation=0.3)  # no model needed at gamma = 0
        w, _ = bridge(sigma, idx, knots, 0.0, np.ones(sigma.shape[0]))
        err = max(err, np.abs(w / w.sum() - nco(sigma, idx)).max())
    return {"nco": err}


def check_violation(rng, trials=20):
    s = [check_sufficiency(rng, 1, violation=0.5)["pair"] for _ in range(trials)]
    r = [check_bridge_right(rng, 1, violation=0.5) for _ in range(trials)]
    return {"pair": min(s), "stacked": min(x["stacked"] for x in r),
            "recipe": min(x["recipe"] for x in r)}


def residual_R(sigma, I, c):
    O = [m for m in I if m != c]
    rest = [i for i in range(sigma.shape[0]) if i not in I]
    return sigma[np.ix_(O, rest)] - np.outer(sigma[O, c] / sigma[c, c], sigma[c, rest])


def check_diagnostic(rng):
    out = {}
    for name, violation in (("model", 0.0), ("violated", 0.5)):
        sigma, idx, knots = gateway_model_cov(rng, [3, 2, 4], violation)
        coef = R = 0.0
        for j, I in enumerate(idx):
            c, O = I[0], I[1:]
            reg = [c] + [x for l, x in enumerate(knots) if l != j]
            B = sigma[np.ix_(O, reg)] @ np.linalg.inv(sigma[np.ix_(reg, reg)])
            coef = max(coef, np.abs(B[:, 1:]).max())
            R = max(R, np.abs(residual_R(sigma, I, c)).max())
        out[name] = {"other_knot_coef": coef, "R": R}
    return out


def check_change_of_vars(rng, trials=100):
    back = 0.0
    for _ in range(trials):
        m = rng.integers(2, 6)
        Q = random_spd(rng, m)
        b = rng.uniform(0.5, 2.0, size=m) * rng.choice([-1, 1], size=m)
        M = np.diag(1 / b) @ Q @ np.diag(1 / b)
        assert np.allclose(M, np.linalg.inv(np.linalg.inv(Q) * np.outer(b, b)))
        y = np.linalg.solve(M, np.ones(m))
        x = y / b
        t = np.linalg.solve(Q, b)
        back = max(back, np.abs(x / x.sum() - t / t.sum()).max())
    # Q = I, b = (1, 2): minimum variance on M without mapping back gives (1/5, 4/5), not (1/3, 2/3)
    y = np.linalg.solve(np.diag([1.0, 0.25]), np.ones(2))
    return {"mapped_back": back, "unmapped_example": y / y.sum()}


def check_proxy(rng, trials=100):
    err = 0.0
    for _ in range(trials):
        sigma, idx, knots, _ = _random_problem(rng)
        n, k = sigma.shape[0], len(idx)
        V = np.zeros((n, k)); kappa = np.zeros(k); D = np.zeros(k)
        for i, I in enumerate(idx):
            c, O = I[0], I[1:]
            V[I, i] = rng.standard_normal(len(I))  # any portfolio on the cluster
            beta = sigma[O, c] / sigma[c, c]
            E = sigma[np.ix_(O, O)] - np.outer(beta, beta) * sigma[c, c]
            kappa[i] = V[c, i] + beta @ V[O, i]
            D[i] = V[O, i] @ E @ V[O, i]
        rhs = np.diag(kappa) @ sigma[np.ix_(knots, knots)] @ np.diag(kappa) + np.diag(D)
        err = max(err, np.abs(V.T @ sigma @ V - rhs).max())
    return {"identity": err}


def check_symmetric(rng, trials=50):
    err = 0.0
    for _ in range(trials):
        sizes = list(rng.integers(2, 5, size=3))
        idx, _ = clusters([s - 1 for s in sizes])
        n, k = idx[-1][-1] + 1, len(idx)
        F = random_spd(rng, k)
        lam = rng.uniform(0.5, 1.5, size=n)
        L = np.zeros((n, k))
        for j, I in enumerate(idx):
            L[I, j] = lam[I]
        sigma = L @ F @ L.T + np.diag(rng.uniform(0.1, 0.5, size=n))
        for j, I in enumerate(idx):
            rest = [i for i in range(n) if i not in I]
            cov_f_rest = (F[j] @ L[rest].T)
            for c in I:
                O = [m for m in I if m != c]
                rho = lam[c] ** 2 * F[j, j] / sigma[c, c]
                pred = (1 - rho) * np.outer(lam[O], cov_f_rest)
                err = max(err, np.abs(residual_R(sigma, I, c) - pred).max())
    return {"formula": err}


def check_quotient(rng, trials=50):
    """Complementing against D then against A2 equals complementing against A2 u D."""
    err = 0.0
    for _ in range(trials):
        n1, n2, n3 = rng.integers(1, 5, size=3)
        n = n1 + n2 + n3
        sigma = random_spd(rng, n)
        u = rng.standard_normal(n)
        A1 = list(range(n1)); A = list(range(n1 + n2))
        QA, bA = full_pair(sigma, A, u)                 # step 1: A against D
        loc1 = list(range(n1))
        Q2, b2 = full_pair(QA, loc1, bA)                # step 2: A1 against A2, inside the pair
        Q1, b1 = full_pair(sigma, A1, u)                # one step: A1 against everything
        err = max(err, np.abs(Q2 - Q1).max(), np.abs(b2 - b1).max())
    return {"compose": err}


def check_loss(rng, trials=50):
    """Delta_i = (full correction) - (knot-only correction): PSD, zero under the model,
    and not tracked by the raw size of R (delta -> 0 example)."""
    zero = 0.0
    min_eig = np.inf
    for _ in range(trials):
        for violation, sink in ((0.0, "zero"), (0.5, "eig")):
            sigma, idx, knots, _ = _random_problem(rng, violation)
            u = np.ones(sigma.shape[0])
            for i, I in enumerate(idx):
                other = [k for j, k in enumerate(knots) if j != i]
                Qf, _ = full_pair(sigma, I, u)
                Qc, _ = cheap_pair(sigma, I, other, 1.0, u)
                delta = Qc - Qf
                if violation == 0.0:
                    zero = max(zero, np.abs(delta).max())
                else:
                    min_eig = min(min_eig, np.linalg.eigvalsh(delta).min())
    # assets (x, p, m) with m = p + e, Var(e) = d^2, Cov(x, e) = c d, x uncorrelated with p
    cc, example = 0.6, []
    for d in (1e-1, 1e-2, 1e-3):
        sigma = np.array([[1.0, 0.0, cc * d], [0.0, 1.0, 1.0], [cc * d, 1.0, 1.0 + d * d]])
        R = abs(residual_R(sigma, [1, 2], 1)).max()
        Qf, _ = full_pair(sigma, [0], np.ones(3))
        Qc, _ = cheap_pair(sigma, [0], [1], 1.0, np.ones(3))
        example.append((R, float((Qc - Qf)[0, 0])))
    return {"zero_under_model": zero, "min_eig_violated": min_eig, "R_vs_loss": example, "c2": cc ** 2}


def check_precision_sparsity(rng, trials=50):
    """Gaussian factorization: precision is zero between J_i and everything outside I_i."""
    out = {}
    for name, violation in (("model", 0.0), ("violated", 0.5)):
        worst = 0.0 if name == "model" else np.inf
        for _ in range(trials):
            sigma, idx, knots, _ = _random_problem(rng, violation)
            prec = np.linalg.inv(sigma)
            m = 0.0
            for I in idx:
                J = I[1:]
                rest = [a for a in range(sigma.shape[0]) if a not in I]
                if J:
                    m = max(m, np.abs(prec[np.ix_(J, rest)]).max())
            worst = max(worst, m) if name == "model" else min(worst, m)
        out[name] = worst
    return out


# ---------------------------------------------------------------------------
# Where to sit on the bridge: unnormalized form, exact examples, endpoint shift
# ---------------------------------------------------------------------------
from fractions import Fraction as Fr


def cluster_direction(sigma_hat, I, other, gamma, u, rel=1e-10):
    """d_i = Q_i^{-1} b_i, with the continuous extension when b_i vanishes.

    b_i = u_I - gamma a_i with a_i = S_{I,P} S_PP^{-1} u_P. The test for b_i = 0 is relative to
    the two terms that cancel, so it is invariant to rescaling Sigma or u. When b_i vanishes
    at gamma > 0 the shape of d_i has the limit Q_i^{-1} d b_i / d gamma = -Q_i^{-1} a_i, and
    that limiting direction is used, which keeps the span continuous in gamma. A cluster is
    dropped only if u_I itself is zero. Returns None for a dropped cluster.
    """
    Q, b = cheap_pair(sigma_hat, I, other, gamma, u)
    a = (u[I] - b) / gamma if gamma > 0 else np.zeros(len(I))
    scale = max(np.abs(u[I]).max(), np.abs(gamma * a).max())
    if np.abs(b).max() <= rel * scale:
        if gamma > 0 and np.abs(a).max() > 0:
            b = -a
        else:
            return None
    d = np.linalg.solve(Q, b)
    return d / np.abs(d).max()          # scale is immaterial to the span; unit columns condition the Gram matrix


def bridge_unnormalized(sigma_hat, idx, knots, gamma, u=None):
    """w ∝ D (D' S D)^{-1} D' u with every quantity read from sigma_hat."""
    n = sigma_hat.shape[0]
    u = np.ones(n) if u is None else u
    cols = []
    for i, I in enumerate(idx):
        other = [k for j, k in enumerate(knots) if j != i]
        d = cluster_direction(sigma_hat, I, other, gamma, u)
        if d is not None:
            c = np.zeros(n); c[I] = d; cols.append(c)
    D = np.array(cols).T
    return D @ np.linalg.solve(D.T @ sigma_hat @ D, D.T @ u)


def check_unnormalized(rng, trials=50):
    """Equals the two-tier form when totals are nonzero; survives a zero cluster total."""
    same = zero_total = 0.0
    for _ in range(trials):
        sigma, idx, knots, mu = _random_problem(rng)
        for gam in (0.0, 0.4, 1.0):
            a = bridge_unnormalized(sigma, idx, knots, gam); a = a / a.sum()
            b, _ = bridge(sigma, idx, knots, gam, np.ones(sigma.shape[0])); b = b / b.sum()
            same = max(same, np.abs(a - b).max())
        # choose u so that Sigma^{-1}u = z has zero total on cluster 0 (needs >= 2 members)
        if len(idx[0]) >= 2:
            z = rng.standard_normal(sigma.shape[0])
            z[idx[0]] -= z[idx[0]].mean()
            w = bridge_unnormalized(sigma, idx, knots, 1.0, sigma @ z)
            zero_total = max(zero_total, np.abs(w - z).max())
    return {"matches_two_tier": same, "zero_total_cluster": zero_total}


def _fr_solve(A, b):
    """Gaussian elimination over Fractions. A is a list of rows, b a list."""
    n = len(A)
    M = [list(map(Fr, A[i])) + [Fr(b[i])] for i in range(n)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        M[c] = [v / M[c][c] for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                M[r] = [vr - M[r][c] * vc for vr, vc in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]


def _sub(S, rows, cols):
    return [[S[r][c] for c in cols] for r in rows]


def bridge_exact(S_hat, idx, knots, gamma):
    """The bridge portfolio in exact rational arithmetic, normalized to sum to one, u = 1."""
    return bridge_generic(S_hat, idx, knots, Fr(gamma), Fr, lambda z: z != 0)


def _var(w, S):
    n = len(w)
    return sum(w[a] * Fr(S[a][b]) * w[b] for a in range(n) for b in range(n))


def check_interior_example():
    """Two clusters, cross-knot estimate Z in {0,1}: unique interior optimum near 0.5587."""
    h = Fr(1, 2)
    S = [[1, 0, h, 0], [0, 1, 0, 0], [h, 0, 4, 0], [0, 0, 0, 1]]
    idx, knots = [[0, 1], [2, 3]], [0, 2]
    def est(Z):
        E = [row[:] for row in S]; E[0][2] = E[2][0] = Z; return E
    def F(gam):
        return (_var(bridge_exact(est(0), idx, knots, gam), S) + _var(bridge_exact(est(1), idx, knots, gam), S)) / 2
    v0 = _var(bridge_exact(est(0), idx, knots, Fr(1, 3)), S)
    gapa, gapb = F(0) - F(h), F(1) - F(h)
    P = lambda x: 49 * x**4 + 84 * x**3 - 21 * x**2 + 48 * x - 6
    dP = lambda x: 196 * x**3 + 252 * x**2 - 42 * x + 48
    lo, hi = Fr(0), Fr(1, 4)
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if P(mid) < 0 else (lo, mid)
    xs = float(lo); gstar = (1 - 4 * xs) / (1 - xs)
    grid = [Fr(j, 40) for j in range(41)]
    gs = Fr(gstar).limit_denominator(10**6)
    return {"Z0_variance": v0, "F0_minus_Fhalf": gapa, "F1_minus_Fhalf": gapb,
            "P0": P(Fr(0)), "Pquarter": P(Fr(1, 4)),
            "dP_min_on_grid": min(dP(Fr(j, 400)) for j in range(101)),
            "gamma_star": gstar, "F_star_is_grid_min": all(F(gs) <= F(gq) for gq in grid)}


def check_full_coupling_example():
    """Three-cluster example, spot check only: F(gamma, tau) > F(1, tau) on a grid, in exact
    rationals. This does not prove global optimality. The exact sign of G'(1) is in check_exact_jets."""
    Sk = [[5, 6, 5], [6, Fr(33, 2), 12], [5, 12, Fr(25, 2)]]
    N = [[0, -1, 0], [-1, 0, 2], [0, 2, 0]]
    resid = [1, 4, 8]
    idx, knots = [[0, 1], [2, 3], [4, 5]], [0, 2, 4]
    def full(K):
        M = [[Fr(0)] * 6 for _ in range(6)]
        for i in range(3):
            for j in range(3):
                M[2 * i][2 * j] = Fr(K[i][j])
            M[2 * i + 1][2 * i + 1] = Fr(resid[i])
        return M
    Sig = full(Sk)
    def V(gam, tau):
        K = [[Fr(Sk[i][j]) + Fr(tau) * N[i][j] for j in range(3)] for i in range(3)]
        return _var(bridge_exact(full(K), idx, knots, gam), Sig)
    def F(gam, tau):
        return (V(gam, tau) + V(gam, -tau)) / 2
    margins = {}
    for tau in (Fr(1, 10), Fr(1, 100)):
        base = F(Fr(1), tau)
        margins[str(tau)] = min(F(gq, tau) - base for gq in (Fr(999, 1000), Fr(99, 100), Fr(9, 10), Fr(1, 2), Fr(0)))
    return {"min_margin": margins}


class Jet:
    """Truncated bivariate Taylor jet over Fractions: sum c[i][j] eg^i et^j, i <= NG, j <= NT.

    Arithmetic is exact, so a coefficient is an exact mixed partial derivative
    divided by i! j!. Same idea as the first-order dual numbers in
    neither_end_certificate.py, carried to order (2, 2) in two variables.
    """
    NG, NT = 2, 2

    def __init__(self, c=None, const=None):
        if c is None:
            c = [[Fr(0)] * (Jet.NT + 1) for _ in range(Jet.NG + 1)]
            if const is not None:
                c[0][0] = Fr(const)
        self.c = c

    @staticmethod
    def of(x):
        return x if isinstance(x, Jet) else Jet(const=x)

    @staticmethod
    def var(value, which):
        j = Jet(const=value)
        if which == "gamma":
            j.c[1][0] = Fr(1)
        else:
            j.c[0][1] = Fr(1)
        return j

    def __add__(s, o):
        o = Jet.of(o)
        return Jet([[a + b for a, b in zip(ra, rb)] for ra, rb in zip(s.c, o.c)])
    __radd__ = __add__

    def __neg__(s):
        return Jet([[-a for a in r] for r in s.c])

    def __sub__(s, o):
        return s + (-Jet.of(o))

    def __rsub__(s, o):
        return Jet.of(o) - s

    def __mul__(s, o):
        o = Jet.of(o)
        out = Jet()
        for i in range(Jet.NG + 1):
            for j in range(Jet.NT + 1):
                if s.c[i][j] == 0:
                    continue
                for p in range(Jet.NG + 1 - i):
                    for q in range(Jet.NT + 1 - j):
                        out.c[i + p][j + q] += s.c[i][j] * o.c[p][q]
        return out
    __rmul__ = __mul__

    def inv(s):
        a00 = s.c[0][0]
        assert a00 != 0, "jet not invertible at the base point"
        b = Jet()
        for i in range(Jet.NG + 1):
            for j in range(Jet.NT + 1):
                acc = Fr(1) if (i, j) == (0, 0) else Fr(0)
                for p in range(i + 1):
                    for q in range(j + 1):
                        if (p, q) != (0, 0):
                            acc -= s.c[p][q] * b.c[i - p][j - q]
                b.c[i][j] = acc / a00
        return b

    def __truediv__(s, o):
        return s * Jet.of(o).inv()

    def __rtruediv__(s, o):
        return Jet.of(o) * s.inv()


def _solve_generic(A, b, lift, nonzero):
    """Gaussian elimination over any exact field-like type."""
    n = len(A)
    M = [[lift(v) for v in A[i]] + [lift(b[i])] for i in range(n)]
    for c in range(n):
        piv = next(r for r in range(c, n) if nonzero(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        M[c] = [v / M[c][c] for v in M[c]]
        for r in range(n):
            if r != c:
                f = M[r][c]
                M[r] = [vr - f * vc for vr, vc in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]


def bridge_generic(S_hat, idx, knots, gamma, lift, nonzero, is_zero=None):
    """The bridge portfolio w ∝ D(D'SD)^{-1}D'1, normalized to sum to one, over an exact type.

    A direction that vanishes identically is dropped, as in the manuscript. With jets, a
    direction that vanishes only at the base point is a degenerate point for differentiation
    and is not supported.
    """
    is_zero = is_zero or (lambda z: not nonzero(z))
    n, k = len(S_hat), len(idx)
    solve = lambda A, b: _solve_generic(A, b, lift, nonzero)
    zero = lift(0)
    D = [[zero] * k for _ in range(n)]
    for i, I in enumerate(idx):
        P = [q for j, q in enumerate(knots) if j != i]
        Spp = [[S_hat[a][b] for b in P] for a in P]
        ones_sol = solve(Spp, [1] * len(P))
        Q = [[lift(S_hat[a][b]) for b in I] for a in I]
        bvec = [lift(1)] * len(I)
        for r, m in enumerate(I):
            col = [lift(S_hat[q][m]) for q in P]
            for c2, m2 in enumerate(I):
                sol = solve(Spp, [S_hat[q][m2] for q in P])
                Q[r][c2] = Q[r][c2] - gamma * sum((x * y for x, y in zip(col, sol)), zero)
            bvec[r] = bvec[r] - gamma * sum((x * y for x, y in zip(col, ones_sol)), zero)
        if all(is_zero(v) for v in bvec):                       # b_i vanishes exactly
            a_vec = [(lift(u_val) - bv) for u_val, bv in zip([1] * len(I), bvec)]   # gamma * a_i
            if not all(is_zero(v) for v in a_vec):
                bvec = [-v for v in a_vec]                       # limiting direction Q^{-1}(-a_i), scale immaterial
        d = solve(Q, bvec)
        for r, m in enumerate(I):
            D[m][i] = d[r]
    keep = [i for i in range(k) if not all(is_zero(D[m][i]) for m in range(n))]
    D = [[D[m][i] for i in keep] for m in range(n)]
    k = len(keep)
    SD = [[sum((lift(S_hat[a][b]) * D[b][j] for b in range(n)), zero) for j in range(k)] for a in range(n)]
    G = [[sum((D[a][i] * SD[a][j] for a in range(n)), zero) for j in range(k)] for i in range(k)]
    rhs = [sum((D[a][i] for a in range(n)), zero) for i in range(k)]
    a = solve(G, rhs)
    w = [sum((D[m][i] * a[i] for i in range(k)), zero) for m in range(n)]
    tot = sum(w, zero)
    return [x / tot for x in w]


def _jet_variance(S_hat_jet, Sig, idx, knots):
    """Population variance of the bridge portfolio as a jet in (gamma - 1, tau)."""
    g = Jet.var(1, "gamma")
    w = bridge_generic(S_hat_jet, idx, knots, g, Jet.of, lambda z: z.c[0][0] != 0,
                       is_zero=lambda z: all(v == 0 for row in z.c for v in row))
    n = len(w)
    out = Jet()
    for a in range(n):
        for b in range(n):
            if Sig[a][b] != 0:
                out = out + w[a] * w[b] * Fr(Sig[a][b])
    return out


def check_exact_jets():
    """Exact G'(1) and V0''(1) for both examples, by jets of order (2, 2) at (gamma, tau) = (1, 0)."""
    tau = Jet.var(0, "tau")
    # three-cluster example
    Sk = [[5, 6, 5], [6, Fr(33, 2), 12], [5, 12, Fr(25, 2)]]
    N = [[0, -1, 0], [-1, 0, 2], [0, 2, 0]]
    resid = [1, 4, 8]
    Sig = [[Fr(0)] * 6 for _ in range(6)]
    Sh = [[Jet() for _ in range(6)] for _ in range(6)]
    for i in range(3):
        for j in range(3):
            Sig[2 * i][2 * j] = Fr(Sk[i][j])
            Sh[2 * i][2 * j] = Jet.of(Sk[i][j]) + tau * N[i][j]
        Sig[2 * i + 1][2 * i + 1] = Fr(resid[i])
        Sh[2 * i + 1][2 * i + 1] = Jet.of(resid[i])
    V3 = _jet_variance(Sh, Sig, [[0, 1], [2, 3], [4, 5]], [0, 2, 4])
    # two-cluster example with Z = 1/2 + tau
    h = Fr(1, 2)
    S2 = [[1, 0, h, 0], [0, 1, 0, 0], [h, 0, 4, 0], [0, 0, 0, 1]]
    Sh2 = [[Jet.of(v) for v in row] for row in S2]
    Sh2[0][2] = Sh2[2][0] = Jet.of(h) + tau
    V2 = _jet_variance(Sh2, [[Fr(v) for v in row] for row in S2], [[0, 1], [2, 3]], [0, 2])
    out = {}
    for name, V in (("three_cluster", V3), ("two_cluster", V2)):
        out[name] = {"V0_prime_1": V.c[1][0], "V0_pp_1": 2 * V.c[2][0], "G_prime_1": V.c[1][2],
                     "odd_in_tau": (V.c[0][1], V.c[1][1])}
    return out


def _two_cluster_F(gam, tau):
    S = np.array([[1, 0, .5, 0], [0, 1, 0, 0], [.5, 0, 4, 0], [0, 0, 0, 1.0]])
    idx, knots = [[0, 1], [2, 3]], [0, 2]
    out = 0.0
    for sgn in (1, -1):
        E = S.copy(); E[0, 2] = E[2, 0] = 0.5 + sgn * tau
        w = bridge_unnormalized(E, idx, knots, gam); w = w / w.sum()
        out += 0.5 * w @ S @ w
    return out


def check_endpoint_shift():
    """gamma~(tau) = 1 - G'(1)/V0''(1) tau^2 on the two-cluster family with Z = 1/2 +- tau.

    G'(1) and V0''(1) are exact (jets); the optimum at each tau is located numerically.
    """
    j = check_exact_jets()["two_cluster"]
    dG1, V0pp = float(j["G_prime_1"]), float(j["V0_pp_1"])
    out = {"G_prime_1": dG1, "V0_pp_1": V0pp, "rows": []}
    for tau in (0.05, 0.1):
        coarse = np.linspace(0.5, 1.0, 1001)
        g0 = coarse[np.argmin([_two_cluster_F(g, tau) for g in coarse])]
        fine = np.linspace(max(0.5, g0 - 0.001), min(1.0, g0 + 0.001), 2001)
        gstar = fine[np.argmin([_two_cluster_F(g, tau) for g in fine])]
        out["rows"].append((tau, float(gstar), min(1.0, 1 - dG1 / V0pp * tau**2)))
    return out


def check_incremental_cost():
    """Phi(gamma, tau) / (gamma tau^2) stays bounded as gamma, tau -> 0."""
    F = _two_cluster_F
    ratios = []
    for gam in (0.2, 0.05):
        for tau in (0.2, 0.05):
            phi = F(gam, tau) - F(0.0, tau) - (F(gam, 0.0) - F(0.0, 0.0))
            ratios.append(phi / (gam * tau**2))
    return {"H_values": ratios}


def check_zero_direction():
    """A direction that vanishes entirely is dropped; the optimum puts nothing on that cluster."""
    cases = [([[2, 1], [1, 1]], [[0], [1]], [0, 1], [0, 1]),
             ([[2, 2, 1], [2, 3, 1], [1, 1, 1]], [[0, 1], [2]], [0, 2], [0, 0, 1])]
    err, exact_ok = 0.0, True
    for S, idx, knots, want in cases:
        w = bridge_unnormalized(np.array(S, float), idx, knots, 1.0)
        err = max(err, np.abs(w / w.sum() - np.array(want, float)).max())
        exact_ok = exact_ok and bridge_exact(S, idx, knots, 1) == [Fr(v) for v in want]
    return {"numpy_error": err, "exact_matches": exact_ok}


def check_sharpe_sign():
    """Sigma = I, mu = (1, -2): the fully invested multiple of Sigma^{-1} mu has Sharpe -sqrt(5)."""
    mu = np.array([1.0, -2.0])
    sharpe = lambda w: w @ mu / np.sqrt(w @ w)
    d = mu.copy()
    return {"total": d.sum(), "sharpe_fully_invested": sharpe(d / d.sum()),
            "sharpe_of_1_0": sharpe(np.array([1.0, 0.0])), "max_sharpe": float(np.sqrt(mu @ mu))}


def check_only_knots_move(rng, trials=60):
    """Under the gateway model the member part of d_i(gamma) is E^{-1}(u_J - beta u_p) for every
    gamma, and the knot exposure is (u_p - gamma a)/(sigma_p^2 - gamma c)."""
    member = exposure = 0.0
    for _ in range(trials):
        sigma, idx, knots, mu = _random_problem(rng)
        for u in (np.ones(len(mu)), mu):
            for i, I in enumerate(idx):
                p, J = I[0], I[1:]
                other = [k for j, k in enumerate(knots) if j != i]
                beta = sigma[J, p] / sigma[p, p]
                E = sigma[np.ix_(J, J)] - np.outer(beta, beta) * sigma[p, p]
                Spp, sp = sigma[np.ix_(other, other)], sigma[p, other]
                a, cc = sp @ np.linalg.solve(Spp, u[other]), sp @ np.linalg.solve(Spp, sp)
                xJ = np.linalg.solve(E, u[J] - beta * u[p]) if J else np.zeros(0)
                for g in (0.0, 0.3, 0.7, 1.0):
                    Q, b = cheap_pair(sigma, I, other, g, u)
                    d = np.linalg.solve(Q, b)
                    if J:
                        member = max(member, np.abs(d[1:] - xJ).max())
                    exposure = max(exposure, abs(d[0] + beta @ d[1:] - (u[p] - g * a) / (sigma[p, p] - g * cc)))
    return {"member_part": member, "knot_exposure": exposure}


def check_equicorrelated_knots():
    """Closed form kappa(gamma) for k unit-variance knots with common correlation rho."""
    err = 0.0
    for k in (2, 3, 5, 8):
        for rho in (-0.1, 0.2, 0.6):
            S = (1 - rho) * np.eye(k) + rho * np.ones((k, k))
            other = list(range(1, k))
            a = S[0, other] @ np.linalg.solve(S[np.ix_(other, other)], np.ones(k - 1))
            cc = S[0, other] @ np.linalg.solve(S[np.ix_(other, other)], S[0, other])
            err = max(err, abs(a - rho * (k - 1) / (1 + (k - 2) * rho)), abs(cc - rho * a))
            for g in (0.0, 0.5, 1.0):
                closed = (1 + (k - 2) * rho - g * rho * (k - 1)) / (1 + (k - 2) * rho - g * rho**2 * (k - 1))
                err = max(err, abs((1 - g * a) / (1 - g * cc) - closed))
            err = max(err, abs((1 - a) / (1 - cc) - 1 / (1 + (k - 1) * rho)))
    return {"closed_form": err}


def check_degenerate_partitions(rng, trials=30):
    """Singleton clusters, or one cluster: the bridge is the minimum-variance portfolio for every
    gamma, on any SPD covariance (no gateway model)."""
    err = 0.0
    for _ in range(trials):
        n = int(rng.integers(3, 8))
        S = random_spd(rng, n)
        gmv = np.linalg.solve(S, np.ones(n)); gmv /= gmv.sum()
        for idx, knots in (([[i] for i in range(n)], list(range(n))), ([list(range(n))], [0])):
            for g in (0.0, 0.37, 1.0):
                w = bridge_unnormalized(S, idx, knots, g)
                err = max(err, np.abs(w / w.sum() - gmv).max())
    return {"bridge_minus_gmv": err}


# ---------------------------------------------------------------------------
# Identical clusters: the one-dimensional picture
# ---------------------------------------------------------------------------
def _t(gam, z, k):
    return (1 + (k - 2) * z - gam * (k - 1) * z) / (1 + (k - 2) * z - gam * (k - 1) * z * z)


def _identical_clusters(k, z, delta):
    """k clusters of (knot, independent asset of variance 1/delta); knot correlation z."""
    n = 2 * k
    S = [[Fr(0)] * n for _ in range(n)]
    for i in range(k):
        for j in range(k):
            S[2 * i][2 * j] = Fr(1) if i == j else Fr(z)
        S[2 * i + 1][2 * i + 1] = 1 / Fr(delta)
    return S, [[2 * i, 2 * i + 1] for i in range(k)], [2 * i for i in range(k)]


def check_identical_clusters():
    """The knot share of the bridge portfolio is t/(t + delta); V(x) - V(x*) is the stated quadratic."""
    share = quad = Fr(0)
    for k, c, z, delta, gam in ((2, Fr(1, 4), Fr(1, 2), 1, Fr(1, 3)), (5, Fr(1, 10), Fr(3, 10), Fr(5, 2), Fr(4, 5)),
                                (4, Fr(1, 5), Fr(-1, 5), Fr(7, 10), Fr(1, 2)), (10, Fr(1, 4), Fr(1, 5), 12, 1)):
        S_hat, idx, knots = _identical_clusters(k, z, delta)
        Sig, _, _ = _identical_clusters(k, c, delta)
        w = bridge_exact(S_hat, idx, knots, gam)
        x = sum(w[2 * i] for i in range(k))
        t = _t(Fr(gam), z, k)
        share = max(share, abs(x - t / (t + Fr(delta))))
        L = 1 + (k - 1) * c
        xs = 1 / (1 + Fr(delta) * L)
        quad = max(quad, abs(_var(w, Sig) - (L * xs**2 + (1 - xs)**2 / Fr(delta)) / k - (L + 1 / Fr(delta)) / k * (x - xs)**2))
    return {"knot_share": share, "quadratic": quad}


def check_two_thirds():
    """Z in {0, 2c}: gamma* = (1 + 2(k-2)c) / (2[1 + (k-3)c]) recovers the population optimum on
    the Z = 2c branch. With k = 2, c = 1/4, delta = 1: gamma* = 2/3, gaps 1/576 and 1/900."""
    def F(k, c, delta, gam):
        Sig, idx, knots = _identical_clusters(k, c, delta)
        return sum(_var(bridge_exact(_identical_clusters(k, Z, delta)[0], idx, knots, gam), Sig) for Z in (Fr(0), 2 * c)) / 2
    k, c = 2, Fr(1, 4)
    gs = (1 + 2 * (k - 2) * c) / (2 * (1 + (k - 3) * c))
    out = {"gamma_star": gs, "gap0": F(k, c, 1, 0) - F(k, c, 1, gs), "gap1": F(k, c, 1, 1) - F(k, c, 1, gs),
           "beats_grid": all(F(k, c, 1, gs) < F(k, c, 1, Fr(j, 30)) for j in range(31) if Fr(j, 30) != gs)}
    hits = True
    for k, c, delta in ((2, Fr(1, 4), 1), (3, Fr(1, 5), 3), (10, Fr(1, 4), 12), (6, Fr(2, 5), Fr(1, 2))):
        g = (1 + 2 * (k - 2) * c) / (2 * (1 + (k - 3) * c))
        L = 1 + (k - 1) * c
        hits = hits and 0 < g < 1 and _t(g, 2 * c, k) == 1 / L
        Sig, idx, knots = _identical_clusters(k, c, delta)
        w = bridge_exact(_identical_clusters(k, 2 * c, delta)[0], idx, knots, g)
        pop = bridge_exact(Sig, idx, knots, 1)          # population minimum variance
        hits = hits and w == pop
    out["general_formula_hits_population_optimum"] = hits
    out["ten_clusters_delta_12"] = (1 + 2 * 8 * Fr(1, 4)) / (2 * (1 + 7 * Fr(1, 4)))
    return out


def check_sign_threshold():
    """k = 10, c = 1/4, Z = c +- tau: G'(1)/V0''(1) = (8 - delta)/(1 + 13 delta/4), by exact jets
    on the scalar reduction."""
    k, c = 10, Fr(1, 4)
    L = 1 + (k - 1) * c
    out = {}
    for delta in (Fr(4), Fr(8), Fr(12)):
        g, z = Jet.var(1, "gamma"), Jet.var(c, "tau")
        t = (1 + (k - 2) * z - g * (k - 1) * z) / (1 + (k - 2) * z - g * (k - 1) * z * z)
        x = t / (t + delta)
        V = (L * x * x + (1 - x) * (1 - x) / delta) / k
        assert V.c[1][0] == 0                                   # endpoint flatness
        out[str(delta)] = (V.c[1][2] / (2 * V.c[2][0]), (8 - delta) / (1 + Fr(13, 4) * delta))
    return out


def _assemble(Spp, idx, parts):
    n = idx[-1][-1] + 1
    load = np.zeros((n, len(idx)))
    for i, I in enumerate(idx):
        load[I[0], i] = 1.0
        load[I[1:], i] = parts[i][0]
    S = load @ Spp @ load.T
    for i, I in enumerate(idx):
        J = I[1:]
        S[np.ix_(J, J)] += parts[i][1]
    return S


def check_compression(rng, trials=30):
    """A gateway cluster acts as its knot plus one independent asset of variance 1/delta_i, for the
    noiseless profile and under noise on the knot covariance that preserves the gateway structure."""
    err = 0.0
    for _ in range(trials):
        sizes = list(rng.integers(1, 6, size=rng.integers(2, 6)))
        sigma, idx, knots = gateway_model_cov(rng, sizes)
        k = len(idx)
        parts = []
        for I in idx:
            p, J = I[0], I[1:]
            beta = sigma[J, p] / sigma[p, p]
            parts.append((beta, sigma[np.ix_(J, J)] - np.outer(beta, beta) * sigma[p, p]))
        Spp = sigma[np.ix_(knots, knots)]
        deltas = [(1 - b) @ np.linalg.solve(E, 1 - b) for b, E in parts]
        idx_c = [[2 * i, 2 * i + 1] for i in range(k)]
        parts_c = [(np.zeros(1), np.array([[1 / deltas[i]]])) for i in range(k)]
        N = rng.standard_normal((k, k)); N = (N + N.T) / 2; np.fill_diagonal(N, 0)
        for tau in (0.0, 0.05, 0.15):
            Sh = Spp + tau * N
            if np.linalg.eigvalsh(Sh).min() <= 0:
                continue
            for gam in (0.0, 0.4, 0.8, 1.0):
                v = []
                for ix, pr in ((idx, parts), (idx_c, parts_c)):
                    w = bridge_unnormalized(_assemble(Sh, ix, pr), ix, [I[0] for I in ix], gam)
                    w = w / w.sum()
                    v.append(w @ _assemble(Spp, ix, pr) @ w)
                err = max(err, abs(v[0] - v[1]))
    return {"variance_gap": err}


def check_unit_betas(rng):
    """Every member has unit beta to its knot: the portfolio is the same at every gamma."""
    idx, knots = clusters([2, 3, 1, 2])
    Spp = random_spd(rng, 4)
    parts = [(np.ones(len(I) - 1), random_spd(rng, len(I) - 1, 0.3)) for I in idx]
    S = _assemble(Spp, idx, parts)
    ws = []
    for gam in (0.0, 0.25, 0.5, 0.75, 1.0):
        w = bridge_unnormalized(S, idx, knots, gam)
        ws.append(w / w.sum())
    return {"change_across_gamma": max(np.abs(w - ws[0]).max() for w in ws),
            "weight_on_members": np.abs(np.delete(ws[0], knots)).max()}


# ---------------------------------------------------------------------------
# Effective damping, lost precision, rank deficiency
# ---------------------------------------------------------------------------
def _kappa(S, gam):
    s = np.diag(S); nu = 1 / np.diag(np.linalg.inv(S)); h = np.linalg.solve(S, np.ones(len(S)))
    return ((1 - gam) + gam * nu * h) / ((1 - gam) * s + gam * nu), s, nu, h


def _compressed_variance(S_hat, S_pop, delta, kappa):
    """Variance of the two-tier portfolio in the compressed model (knots S, member precisions delta)."""
    K = np.diag(kappa)
    a = np.linalg.solve(K @ S_hat @ K + np.diag(delta), kappa + delta)
    J = (kappa + delta) @ a
    q = K @ a / J
    return q @ S_pop @ q + ((a / J) ** 2 * delta).sum(), J


def check_effective_damping(rng, trials=100):
    err = 0.0
    for _ in range(trials):
        S = random_spd(rng, int(rng.integers(2, 7)))
        for gam in (0.0, 0.3, 0.7, 0.95):
            kap, s, nu, h = _kappa(S, gam)
            r = nu / s
            lam = gam * r / (1 - gam + gam * r)
            err = max(err, np.abs(kap - ((1 - lam) / s + lam * h)).max())
    return {"kappa_vs_interpolation": err}


def check_lost_precision(rng, trials=100):
    """J = Z - L, V0 - V* = L/(Z(Z-L)), kappa - h in closed form (floating point), and the
    curvature V0''(1) = 2 p'W1 p / Z^2 in exact arithmetic via jets."""
    loss = regret = gap = 0.0
    for _ in range(trials):
        k = int(rng.integers(2, 7))
        S = random_spd(rng, k); delta = rng.uniform(0.2, 3.0, size=k)
        for gam in (0.0, 0.3, 0.7, 0.95):
            kap, s, nu, h = _kappa(S, gam)
            Z = h.sum() + delta.sum()
            V, J = _compressed_variance(S, S, delta, kap)
            W = np.linalg.inv(np.linalg.inv(S) + np.diag(kap * kap / delta))
            L = (h - kap) @ W @ (h - kap)
            loss = max(loss, abs(J - (Z - L)))
            regret = max(regret, abs(V - 1 / Z - L / (Z * (Z - L))))
            gap = max(gap, np.abs((kap - h) - (1 - gam) * (1 - s * h) / ((1 - gam) * s + gam * nu)).max())
    # exact curvature on a rational example
    S = [[Fr(4), Fr(1), Fr(1, 2)], [Fr(1), Fr(3), Fr(-1, 2)], [Fr(1, 2), Fr(-1, 2), Fr(2)]]
    delta = [Fr(1, 2), Fr(2), Fr(3, 4)]
    k = 3
    solve = lambda A, b: _solve_generic(A, b, Fr, lambda z: z != 0)
    h = solve(S, [1] * k)
    Sinv = [solve(S, [1 if i == j else 0 for i in range(k)]) for j in range(k)]
    nu = [1 / Sinv[i][i] for i in range(k)]
    s = [S[i][i] for i in range(k)]
    Z = sum(h) + sum(delta)
    pvec = [(s[i] * h[i] - 1) / nu[i] for i in range(k)]
    A1 = [[Sinv[j][i] + (h[i] ** 2 / delta[i] if i == j else 0) for j in range(k)] for i in range(k)]
    claim = 2 * sum(pi * xi for pi, xi in zip(pvec, solve(A1, pvec))) / Z ** 2
    g = Jet.var(1, "gamma")
    jsolve = lambda A, b: _solve_generic(A, b, Jet.of, lambda z: z.c[0][0] != 0)
    kap = [((1 - g) + g * nu[i] * h[i]) / ((1 - g) * s[i] + g * nu[i]) for i in range(k)]
    M = [[kap[i] * S[i][j] * kap[j] + (delta[i] if i == j else 0) for j in range(k)] for i in range(k)]
    b = [kap[i] + delta[i] for i in range(k)]
    a = jsolve(M, b)
    J = sum((bi * ai for bi, ai in zip(b, a)), Jet())
    V0 = 1 / J
    return {"J_vs_Z_minus_L": loss, "regret": regret, "kappa_minus_h": gap,
            "curvature_exact_match": 2 * V0.c[2][0] == claim, "V0_prime_1": V0.c[1][0]}


def check_gain_order():
    """F(1, tau) - min F  ~  G'(1)^2 / (2 V0''(1)) tau^4, identical clusters with k = 10, c = 1/4, delta = 4."""
    k, c, delta = 10, 0.25, 4.0
    L = 1 + (k - 1) * c
    Vx = lambda x: (L * x * x + (1 - x) ** 2 / delta) / k
    def F(gam, tau):
        return 0.5 * sum(Vx(_t(gam, z, k) / (_t(gam, z, k) + delta)) for z in (c + tau, c - tau))
    g, z = Jet.var(1, "gamma"), Jet.var(Fr(1, 4), "tau")
    t = (1 + (k - 2) * z - g * (k - 1) * z) / (1 + (k - 2) * z - g * (k - 1) * z * z)
    x = t / (t + Fr(4))
    V = (Fr(L) * x * x + (1 - x) * (1 - x) / Fr(4)) / k
    pred = float(V.c[1][2]) ** 2 / (2 * float(2 * V.c[2][0]))
    ratios = []
    for tau in (0.02, 0.04):
        gs = np.linspace(0.9, 1.0, 200001)
        ratios.append(float((F(1.0, tau) - F(gs, tau).min()) / (pred * tau ** 4)))
    return {"gain_over_prediction": ratios}


def check_rank_deficiency(rng):
    """Duplicated knots, the exposure path, non-commuting limits, and null-space-preserving noise."""
    out = {}
    S4 = [[1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]]
    idx, knots = [[0, 1], [2, 3]], [0, 2]
    out["bridge_below_one"] = {str(gq): _var(bridge_exact(S4, idx, knots, gq), S4) for gq in (Fr(0), Fr(1, 2), Fr(99, 100))}
    Sn = np.array(S4, float)
    D = np.zeros((4, 2))
    for i, (I, p) in enumerate((([0, 1], 2), ([2, 3], 0))):
        Q = Sn[np.ix_(I, I)] - np.outer(Sn[I, p], Sn[I, p]) / Sn[p, p]
        D[I, i] = np.linalg.pinv(Q) @ (np.ones(2) - Sn[I, p] / Sn[p, p])
    w = D @ np.linalg.solve(D.T @ Sn @ D, D.T @ np.ones(4)); w /= w.sum()
    out["local_pinv_endpoint"] = float(w @ Sn @ w)
    S2, d2 = np.ones((2, 2)), np.ones(2)
    h = np.linalg.pinv(S2) @ np.ones(2)
    path = [_compressed_variance(S2, S2, d2, (1 - lam) + lam * h)[0] for lam in np.linspace(0, 1, 21)]
    out["lambda_path_ends"] = (float(path[0]), float(path[-1]))
    out["lambda_path_decreasing"] = bool(np.all(np.diff(path) < 0))
    out["eps_then_gamma"] = float(_compressed_variance(S2 + 1e-6 * np.eye(2), S2, d2, _kappa(S2 + 1e-6 * np.eye(2), 1.0)[0])[0])
    out["gamma_then_eps"] = float(_compressed_variance(S2 + 1e-9 * np.eye(2), S2, d2, _kappa(S2 + 1e-9 * np.eye(2), 0.999)[0])[0])
    end = 0.0
    for _ in range(30):
        k = int(rng.integers(3, 7)); rk = int(rng.integers(1, k))
        B = np.column_stack([rng.standard_normal((k, rk - 1)), np.ones(k)])
        S = B @ B.T                                           # singular, with 1 in its range
        assert np.linalg.matrix_rank(S) < k and np.allclose(S @ np.linalg.pinv(S) @ np.ones(k), np.ones(k))
        delta = rng.uniform(0.3, 2.0, size=k)
        hh = np.linalg.pinv(S) @ np.ones(k)
        end = max(end, abs(_compressed_variance(S, S, delta, hh)[0] - 1 / (hh.sum() + delta.sum())))
    out["lambda_one_is_minvar"] = end
    # noise around a singular covariance
    B = rng.standard_normal((5, 3)); Sig = B @ B.T
    U = np.linalg.svd(Sig)[0]; R, Nn = U[:, :3], U[:, 3:]
    X = rng.standard_normal((3, 3)); X = (X + X.T) / 2; X /= np.abs(np.linalg.eigvalsh(X)).max()
    E_ok = R @ X @ R.T                                   # supported on range(Sigma): E ker(Sigma) = 0
    E_bad = E_ok + (Nn[:, :1] @ R[:, :1].T + R[:, :1] @ Nn[:, :1].T)   # couples the kernel to the range
    tau = 0.2 * np.linalg.eigvalsh(Sig)[2]               # a fifth of the smallest nonzero eigenvalue
    mins = lambda E: min(np.linalg.eigvalsh(Sig + sgn * tau * E).min() for sgn in (1, -1))
    out["noise"] = {"preserving_min_eig": float(mins(E_ok)), "preserving_E_ker": float(np.abs(E_ok @ Nn).max()),
                    "violating_min_eig": float(mins(E_bad))}
    return out


def check_interior_zero_direction():
    """Issue 11: S = [[4, 3/2], [3/2, 1]], singleton clusters, b_1 = 1 - (3/2) gamma vanishes at
    gamma = 2/3. The continuous extension keeps w = (-1/4, 5/4) and variance 7/8 there."""
    S = np.array([[4.0, 1.5], [1.5, 1.0]]); idx, knots = [[0], [1]], [0, 1]
    out = {}
    for g in (2 / 3 - 1e-6, 2 / 3, 2 / 3 + 1e-6):
        w = bridge_unnormalized(S, idx, knots, g); w = w / w.sum()
        out[f"{g:.7f}"] = (float(w[0]), float(w[1]), float(w @ S @ w))
    Sx = [[Fr(4), Fr(3, 2)], [Fr(3, 2), Fr(1)]]
    out["exact_at_two_thirds"] = bridge_exact(Sx, idx, knots, Fr(2, 3))
    # the gamma = 1 regressions must survive: a vanishing block of Sigma^{-1}u still gives the optimum
    out["endpoint_regressions"] = check_zero_direction()
    return out


def check_scale_invariance(rng, trials=40):
    """Issue 12: rescaling Sigma or u must not change fully invested weights."""
    worst = 0.0
    for _ in range(trials):
        sigma, idx, knots, mu = _random_problem(rng)
        for u in (np.ones(len(mu)), mu):
            for g in (0.0, 0.5, 1.0):
                ref = bridge_unnormalized(sigma, idx, knots, g, u); ref = ref / ref.sum()
                for cs, cu in ((1e12, 1.0), (1e-9, 1.0), (1.0, 1e8), (1e6, 1e-7)):
                    w = bridge_unnormalized(cs * sigma, idx, knots, g, cu * u); w = w / w.sum()
                    worst = max(worst, np.abs(w - ref).max())
    diag = bridge_unnormalized(1e12 * np.diag([1.0, 2.0]), [[0], [1]], [0, 1], 0.0)
    return {"max_change_under_rescaling": worst, "diag_1e12": (diag / diag.sum()).tolist()}


def main():
    rng = np.random.default_rng(0)
    r = check_sufficiency(rng); print("Prop 1 sufficiency        ", r); assert r["pair"] < TOL
    r = check_bridge_right(rng); print("Prop 2 right end          ", r)
    assert r["stacked"] < TOL and r["recipe"] < TOL
    r = check_bridge_left(rng); print("Prop 2 left end = NCO     ", r); assert r["nco"] < TOL
    r = check_violation(rng); print("model violated (min err)  ", r)
    assert r["pair"] > 1e-3 and r["stacked"] > 1e-4 and r["recipe"] > 1e-4
    r = check_diagnostic(rng); print("diagnostic                ", r)
    assert r["model"]["R"] < TOL and r["violated"]["R"] > 1e-2
    assert r["violated"]["other_knot_coef"] < TOL  # the regression test false-accepts
    r = check_change_of_vars(rng); print("change of variables       ", r)
    assert r["mapped_back"] < TOL and np.allclose(r["unmapped_example"], [0.2, 0.8])
    r = check_proxy(rng); print("Prop 3 proxy identity     ", r); assert r["identity"] < TOL
    r = check_symmetric(rng); print("symmetric factor formula  ", r); assert r["formula"] < TOL
    r = check_quotient(rng); print("complements compose       ", r); assert r["compose"] < TOL
    r = check_loss(rng); print("loss Delta_i              ", r)
    assert r["zero_under_model"] < TOL and r["min_eig_violated"] > -1e-9
    assert all(abs(loss - r["c2"]) < 1e-6 for _, loss in r["R_vs_loss"])          # loss stays at c^2
    assert r["R_vs_loss"][-1][0] < 1e-2 * r["R_vs_loss"][0][0] * 1.0001            # while R -> 0
    r = check_precision_sparsity(rng); print("precision sparsity        ", r)
    assert r["model"] < 1e-8 and r["violated"] > 1e-3
    r = check_unnormalized(rng); print("unnormalized bridge       ", r)
    assert r["matches_two_tier"] < 1e-8 and r["zero_total_cluster"] < 1e-8
    r = check_interior_example(); print("interior example          ", {k: (str(v) if isinstance(v, Fr) else v) for k, v in r.items()})
    assert r["Z0_variance"] == Fr(56, 169)
    assert r["F0_minus_Fhalf"] == Fr(3, 1210) and r["F1_minus_Fhalf"] == Fr(5, 1452)
    assert r["P0"] == -6 and r["Pquarter"] == Fr(1585, 256) and r["dP_min_on_grid"] > 0
    assert abs(r["gamma_star"] - 0.558733606) < 1e-8 and r["F_star_is_grid_min"]
    r = check_exact_jets()
    print("exact jets                ", {k: {a: (str(b) if isinstance(b, Fr) else tuple(map(str, b))) for a, b in v.items()} for k, v in r.items()})
    three, two = r["three_cluster"], r["two_cluster"]
    for ex in (three, two):
        assert ex["V0_prime_1"] == 0           # endpoint flatness, exactly
        assert ex["odd_in_tau"][0] == 0        # first-order noise does not move the variance at the optimum
        assert ex["V0_pp_1"] > 0
    assert three["G_prime_1"] == Fr(-4720238484060245648386156800, 806907939268294475102923640251)
    assert three["V0_pp_1"] == Fr(114868070650483200, 47963470838273449)
    assert two["G_prime_1"] == Fr(31779128936, 554523204167) and two["V0_pp_1"] == Fr(48136, 3571279)
    r = check_full_coupling_example(); print("full-coupling spot check  ", {a: float(b) for a, b in r["min_margin"].items()})
    assert all(m > 0 for m in r["min_margin"].values())
    r = check_endpoint_shift(); print("endpoint shift            ", r)
    for tau, gstar, pred in r["rows"]:
        assert abs((1 - gstar) - (1 - pred)) <= 0.25 * max(1 - pred, 1e-3) + 2e-3, (tau, gstar, pred)
    r = check_incremental_cost(); print("incremental cost H        ", r)
    assert max(abs(v) for v in r["H_values"]) < 10 * (min(abs(v) for v in r["H_values"]) + 1e-12)
    r = check_zero_direction(); print("zero direction at gamma=1 ", r)
    assert r["numpy_error"] < 1e-12 and r["exact_matches"]
    r = check_interior_zero_direction(); print("interior zero direction   ", {a: (str(b) if not isinstance(b, dict) else "…") for a, b in r.items()})
    for key, (w0, w1, v) in [(k2, v2) for k2, v2 in r.items() if k2[0].isdigit()]:
        assert abs(w0 + 0.25) < 1e-9 and abs(w1 - 1.25) < 1e-9 and abs(v - 7 / 8) < 1e-9, key
    assert r["exact_at_two_thirds"] == [Fr(-1, 4), Fr(5, 4)]
    r = check_scale_invariance(rng); print("scale invariance          ", r)
    assert r["max_change_under_rescaling"] < 1e-8 and np.allclose(r["diag_1e12"], [2 / 3, 1 / 3])
    r = check_sharpe_sign(); print("Sharpe sign               ", r)
    assert r["total"] < 0 and abs(r["sharpe_fully_invested"] + r["max_sharpe"]) < 1e-12 and r["sharpe_of_1_0"] > 0
    r = check_only_knots_move(rng); print("only the knots move       ", r)
    assert r["member_part"] < 1e-9 and r["knot_exposure"] < 1e-9
    r = check_equicorrelated_knots(); print("equicorrelated knots      ", r); assert r["closed_form"] < 1e-12
    r = check_degenerate_partitions(rng); print("degenerate partitions     ", r); assert r["bridge_minus_gmv"] < 1e-10
    r = check_identical_clusters(); print("identical clusters        ", {a: str(b) for a, b in r.items()})
    assert r["knot_share"] == 0 and r["quadratic"] == 0
    r = check_two_thirds(); print("two-thirds example        ", {a: str(b) for a, b in r.items()})
    assert r["gamma_star"] == Fr(2, 3) and r["gap0"] == Fr(1, 576) and r["gap1"] == Fr(1, 900)
    assert r["beats_grid"] and r["general_formula_hits_population_optimum"] and r["ten_clusters_delta_12"] == Fr(10, 11)
    r = check_sign_threshold(); print("sign threshold            ", {a: tuple(map(str, b)) for a, b in r.items()})
    assert all(got == want for got, want in r.values())
    assert r["4"][0] > 0 and r["8"][0] == 0 and r["12"][0] == Fr(-1, 10)
    r = check_compression(rng); print("cluster compression       ", r); assert r["variance_gap"] < 1e-10
    r = check_unit_betas(rng); print("unit betas                ", r)
    assert r["change_across_gamma"] < 1e-12 and r["weight_on_members"] < 1e-12
    r = check_effective_damping(rng); print("effective damping         ", r); assert r["kappa_vs_interpolation"] < 1e-12
    r = check_lost_precision(rng); print("lost precision            ", {a: (str(b) if isinstance(b, Fr) else b) for a, b in r.items()})
    assert r["J_vs_Z_minus_L"] < 1e-10 and r["regret"] < 1e-12 and r["kappa_minus_h"] < 1e-12
    assert r["curvature_exact_match"] and r["V0_prime_1"] == 0
    r = check_gain_order(); print("fourth-order gain         ", r)
    assert abs(r["gain_over_prediction"][0] - 1) < 0.06 and r["gain_over_prediction"][0] < r["gain_over_prediction"][1]
    r = check_rank_deficiency(rng); print("rank deficiency           ", {a: ({x: str(y) for x, y in b.items()} if a == "bridge_below_one" else b) for a, b in r.items()})
    assert all(v == Fr(3, 8) for v in r["bridge_below_one"].values()) and abs(r["local_pinv_endpoint"] - 0.5) < 1e-12
    assert abs(r["lambda_path_ends"][0] - 3 / 8) < 1e-12 and abs(r["lambda_path_ends"][1] - 1 / 3) < 1e-12 and r["lambda_path_decreasing"]
    assert abs(r["eps_then_gamma"] - 1 / 3) < 1e-5 and abs(r["gamma_then_eps"] - 3 / 8) < 1e-5 and r["lambda_one_is_minvar"] < 1e-10
    assert r["noise"]["preserving_min_eig"] > -1e-12 and r["noise"]["preserving_E_ker"] < 1e-12 and r["noise"]["violating_min_eig"] < -1e-6
    print("certificate ok")


if __name__ == "__main__":
    main()
