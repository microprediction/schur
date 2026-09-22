"""Certificates for "The implied covariance of an allocator".

Part 1 (exact, stdlib only). Proposition: for any w with 1'w = 1 the matrix
Sigma' = 11' + eps Q M Q' is positive definite and satisfies Sigma' w = 1, so
w is the global minimum-variance portfolio of Sigma'. Checked in exact
rational arithmetic on a long-short target, including positive definiteness
by leading principal minors.

Part 2 (exact, stdlib only). Lemma: with c = w'Sw and r = c1 - Sw, the matrix
Delta = (rw' + wr')/(w'w) - (r'w) ww'/(w'w)^2 is symmetric, satisfies
Delta w = r, has column space inside span{w, r}, and is the minimum-norm
such correction, certified by <Delta, E> = 0 for every symmetric E with
Ew = 0.

Part 3 (numerical). The distortion table, the shrinkage-family fits for HRP,
the bridge table, and the estimator-versus-allocator comparison.
Requires numpy and scipy; the bridge table additionally imports the
`allocation` package, and is skipped with a message if it is absent.

Run:  python verify_implied.py
"""
from fractions import Fraction as Fr

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


# --------------------------------------------------------------- exact linalg
def mmul(X, Y):
    return [[sum(X[i][k] * Y[k][j] for k in range(len(Y))) for j in range(len(Y[0]))]
            for i in range(len(X))]


def mvec(X, v):
    return [sum(X[i][k] * v[k] for k in range(len(v))) for i in range(len(X))]


def outer(a, b):
    return [[a[i] * b[j] for j in range(len(b))] for i in range(len(a))]


def madd(X, Y):
    return [[X[i][j] + Y[i][j] for j in range(len(X[0]))] for i in range(len(X))]


def mscale(X, s):
    return [[s * X[i][j] for j in range(len(X[0]))] for i in range(len(X))]


def det(X):
    """Exact determinant by fraction-free elimination on a copy."""
    A = [row[:] for row in X]
    n = len(A)
    d = Fr(1)
    for i in range(n):
        p = next((k for k in range(i, n) if A[k][i] != 0), None)
        if p is None:
            return Fr(0)
        if p != i:
            A[i], A[p] = A[p], A[i]
            d = -d
        d *= A[i][i]
        inv = Fr(1) / A[i][i]
        for k in range(i + 1, n):
            f = A[k][i] * inv
            if f:
                for j in range(i, n):
                    A[k][j] -= f * A[i][j]
    return d


def is_pd(X):
    return all(det([row[:k + 1] for row in X[:k + 1]]) > 0 for k in range(len(X)))


def is_sym(X):
    return all(X[i][j] == X[j][i] for i in range(len(X)) for j in range(len(X)))


# ------------------------------------------------- Part 1: the proposition
def part1():
    print("\nPart 1. Every portfolio is a minimum-variance portfolio (exact)")
    # A long-short target with a unit budget.
    w = [Fr(3, 2), Fr(-1, 2), Fr(5, 4), Fr(-3, 4), Fr(-1, 2)]
    n = len(w)
    check("target budget is one", sum(w) == 1, f"1'w = {sum(w)}")
    check("target is long-short", any(x < 0 for x in w),
          f"{sum(1 for x in w if x < 0)} of {n} weights negative")

    one = [Fr(1)] * n
    ww = sum(x * x for x in w)
    # Q = I - ww'/(w'w) is the exact projector onto the complement of span{w}.
    Q = [[(Fr(1) if i == j else Fr(0)) - w[i] * w[j] / ww for j in range(n)]
         for i in range(n)]
    # M is an arbitrary SPD matrix: a diagonally dominant symmetric one.
    M = [[Fr(1, 1 + abs(i - j)) + (Fr(n) if i == j else Fr(0)) for j in range(n)]
         for i in range(n)]
    check("M is symmetric positive definite", is_sym(M) and is_pd(M))

    eps = Fr(1, 3)
    QMQ = mmul(mmul(Q, M), Q)
    Sig = madd(outer(one, one), mscale(QMQ, eps))

    check("Sigma' is symmetric", is_sym(Sig))
    check("Sigma' is positive definite", is_pd(Sig),
          f"leading minors all > 0, det = {det(Sig)}")
    check("Sigma' w = 1 exactly", mvec(Sig, w) == one)
    # w is then the GMV portfolio: Sigma'^{-1} 1 = w and 1'w = 1.
    check("w is the global minimum-variance portfolio of Sigma'",
          mvec(Sig, w) == one and sum(w) == 1)

    # The claim uses no positivity of w, so repeat on a second sign pattern.
    w2 = [Fr(-2), Fr(7, 3), Fr(1, 3), Fr(1, 2), Fr(-1, 6)]
    ww2 = sum(x * x for x in w2)
    Q2 = [[(Fr(1) if i == j else Fr(0)) - w2[i] * w2[j] / ww2 for j in range(n)]
          for i in range(n)]
    Sig2 = madd(outer(one, one), mscale(mmul(mmul(Q2, M), Q2), eps))
    check("second long-short target: Sigma' PD and Sigma' w = 1",
          sum(w2) == 1 and is_pd(Sig2) and mvec(Sig2, w2) == one)


# ------------------------------------------------------ Part 2: the lemma
def part2():
    print("\nPart 2. The minimal implied correction is symmetric of rank two (exact)")
    n = 5
    # An exact SPD sample-covariance stand-in.
    S = [[Fr(1, 1 + abs(i - j)) + (Fr(2) if i == j else Fr(0)) for j in range(n)]
         for i in range(n)]
    check("S is symmetric positive definite", is_sym(S) and is_pd(S))
    w = [Fr(1, 2), Fr(-1, 4), Fr(1, 3), Fr(1, 6), Fr(1, 4)]
    check("budget is one", sum(w) == 1)

    one = [Fr(1)] * n
    c = sum(w[i] * S[i][j] * w[j] for i in range(n) for j in range(n))
    Sw = mvec(S, w)
    r = [c - Sw[i] for i in range(n)]
    ww = sum(x * x for x in w)
    rw = sum(r[i] * w[i] for i in range(n))

    D = madd(mscale(madd(outer(r, w), outer(w, r)), Fr(1) / ww),
             mscale(outer(w, w), -rw / (ww * ww)))

    check("Delta is symmetric", is_sym(D))
    check("Delta w = r exactly", mvec(D, w) == r)
    X = madd(S, D)
    check("(S + Delta) w = c 1 exactly", mvec(X, w) == [c] * n, f"c = {c}")

    # Column space inside span{w, r}: every 3x3 minor of [D | w | r] columns
    # vanishes, i.e. rank([w r]) = rank([w r D]) .
    cols = [[w[i] for i in range(n)], [r[i] for i in range(n)]]
    cols_ext = cols + [[D[i][j] for i in range(n)] for j in range(n)]
    check("rank of [w r] is two", exact_rank(cols) == 2)
    check("column space of Delta lies in span{w, r}",
          exact_rank(cols_ext) == 2, "rank([w r Delta]) = 2")

    # Minimum norm: Delta is orthogonal to the tangent space {E = E' : Ew = 0}.
    worst = Fr(0)
    for a in range(n):
        for b in range(a, n):
            E = tangent(n, a, b, w)
            if E is None:
                continue
            ip = sum(D[i][j] * E[i][j] for i in range(n) for j in range(n))
            worst = max(worst, abs(ip))
    check("Delta is orthogonal to every symmetric E with E w = 0",
          worst == 0, f"largest |<Delta, E>| over a spanning set = {worst}")


def exact_rank(cols):
    """Rank of the matrix whose COLUMNS are the given exact vectors."""
    rows = [list(c) for c in zip(*cols)] if cols else []
    m, k = len(rows), len(rows[0]) if rows else 0
    A = [row[:] for row in rows]
    rank = 0
    for j in range(k):
        p = next((i for i in range(rank, m) if A[i][j] != 0), None)
        if p is None:
            continue
        A[rank], A[p] = A[p], A[rank]
        inv = Fr(1) / A[rank][j]
        for i in range(m):
            if i != rank and A[i][j] != 0:
                f = A[i][j] * inv
                for jj in range(k):
                    A[i][jj] -= f * A[rank][jj]
        rank += 1
    return rank


def tangent(n, a, b, w):
    """A symmetric E supported on rows/cols {a, b} and a correction, with
    E w = 0; returns None when the construction degenerates."""
    E = [[Fr(0)] * n for _ in range(n)]
    E[a][b] = E[b][a] = Fr(1)
    if a == b:
        E[a][a] = Fr(1)
    v = mvec(E, w)
    # Project out: E - (vw' + wv')/(w'w) + (v'w) ww'/(w'w)^2 kills E w.
    ww = sum(x * x for x in w)
    vw = sum(v[i] * w[i] for i in range(n))
    F = madd(E, mscale(madd(outer(v, w), outer(w, v)), Fr(-1) / ww))
    F = madd(F, mscale(outer(w, w), vw / (ww * ww)))
    return F if mvec(F, w) == [Fr(0)] * n else None


# ------------------------------------------------------ Part 3: the numbers
def part3():
    try:
        import numpy as np
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        from scipy.optimize import minimize
    except Exception as exc:                                   # pragma: no cover
        print(f"\nPart 3 skipped: {exc}")
        return

    print("\nPart 3. Distortion, shrinkage fits, the bridge, and the comparison")

    def market(rng, nb=5, per=8, rin=0.7, rout=0.15):
        n = nb * per
        C = np.full((n, n), rout)
        for b in range(nb):
            s = slice(b * per, (b + 1) * per)
            C[s, s] = rin
        np.fill_diagonal(C, 1.0)
        v = np.exp(rng.normal(0.0, 0.4, n))
        return v[:, None] * C * v[None, :]

    def tree_order(S):
        d = np.sqrt(np.diag(S))
        R = np.clip(S / np.outer(d, d), -1.0, 1.0)
        D = np.sqrt(np.maximum(0.5 * (1.0 - R), 0.0))
        np.fill_diagonal(D, 0.0)
        Z = linkage(squareform(D, checks=False), method="single")
        return Z, list(leaves_list(Z))

    def cluster_var(S, idx):
        sub = S[np.ix_(idx, idx)]
        iv = 1.0 / np.diag(sub)
        iv /= iv.sum()
        return float(iv @ sub @ iv)

    def hrp(S):
        n = S.shape[0]
        _, order = tree_order(S)
        w = np.ones(n)
        cl = [order]
        while cl:
            nxt = []
            for c in cl:
                if len(c) <= 1:
                    continue
                h = len(c) // 2
                l, r = c[:h], c[h:]
                vl, vr = cluster_var(S, l), cluster_var(S, r)
                a = 1.0 - vl / (vl + vr)
                w[l] *= a
                w[r] *= 1.0 - a
                nxt += [l, r]
            cl = nxt
        return w / w.sum()

    def min_var(S, ridge=0.0):
        x = np.linalg.solve(S + ridge * np.eye(S.shape[0]), np.ones(S.shape[0]))
        return x / x.sum()

    def inv_var(S):
        w = 1.0 / np.diag(S)
        return w / w.sum()

    def risk_parity(S, iters=600):
        n = S.shape[0]
        w = np.ones(n) / n
        for _ in range(iters):
            w = np.maximum(w * (1.0 / (n * (S @ w))), 1e-12)
            w /= w.sum()
        return w

    def implied(S, w):
        c = float(w @ S @ w)
        r = c * np.ones_like(w) - S @ w
        ww = float(w @ w)
        D = (np.outer(r, w) + np.outer(w, r)) / ww - float(r @ w) * np.outer(w, w) / ww ** 2
        ev = np.linalg.eigvalsh(S + D)
        return np.linalg.norm(D) / np.linalg.norm(S), ev.min() / ev.max(), D, r

    def block_filter(S, k):
        d = np.sqrt(np.diag(S))
        R = S / np.outer(d, d)
        from scipy.cluster.hierarchy import fcluster
        Z, _ = tree_order(S)
        lab = fcluster(Z, t=k, criterion="maxclust")
        F = np.zeros_like(R)
        for a in np.unique(lab):
            for b in np.unique(lab):
                ia, ib = np.where(lab == a)[0], np.where(lab == b)[0]
                blk = R[np.ix_(ia, ib)]
                m = blk.mean() if a != b else (blk.sum() - len(ia)) / max(len(ia) ** 2 - len(ia), 1)
                F[np.ix_(ia, ib)] = m
        np.fill_diagonal(F, 1.0)
        ev, V = np.linalg.eigh(F)
        F = V @ np.diag(np.maximum(ev, 1e-8)) @ V.T
        dd = np.sqrt(np.diag(F))
        F = F / np.outer(dd, dd)
        return d[:, None] * F * d[None, :], lab

    # --- Table 1: distortion by allocator
    rng = np.random.default_rng(7)
    ALLOC = {"minimum variance": min_var, "risk parity": risk_parity, "HRP": hrp,
             "inverse variance": inv_var, "equal weight": lambda S: np.ones(len(S)) / len(S)}
    dist = {k: [] for k in ALLOC}
    cond = {k: [] for k in ALLOC}
    rank_ok, span_ok = True, 0.0
    for _ in range(300):
        Sig = market(rng)
        X = rng.normal(size=(120, 40)) @ np.linalg.cholesky(Sig).T
        S = np.cov(X, rowvar=False)
        for k, f in ALLOC.items():
            w = f(S)
            d, c, D, r = implied(S, w)
            dist[k].append(d)
            cond[k].append(c)
            if k == "HRP":
                rank_ok &= np.linalg.matrix_rank(D, tol=1e-9 * np.linalg.norm(D)) == 2
                B = np.linalg.qr(np.column_stack([w, r]))[0]
                span_ok = max(span_ok, np.linalg.norm(D - B @ B.T @ D @ B @ B.T) / np.linalg.norm(D))
    print("\n  Table 1  implied-estimator distortion, median of 300 sample covariances")
    print(f"  {'allocator':22s}{'||Delta||/||S||':>17s}{'lambda_min/lambda_max':>23s}")
    for k in ALLOC:
        print(f"  {k:22s}{np.median(dist[k]):17.3f}{np.median(cond[k]):23.4f}")
    check("minimum variance has zero distortion", np.median(dist["minimum variance"]) < 1e-12)

    # Direction, not just size: is the implied covariance nearer block structure?
    def block_gap(M, lab):
        d = np.sqrt(np.abs(np.diag(M)))
        R = M / np.outer(d, d)
        P = np.zeros_like(R)
        for a in np.unique(lab):
            for b in np.unique(lab):
                ia, ib = np.where(lab == a)[0], np.where(lab == b)[0]
                P[np.ix_(ia, ib)] = R[np.ix_(ia, ib)].mean()
        np.fill_diagonal(P, np.diag(R))
        return np.linalg.norm(R - P) / np.linalg.norm(R)

    rng2 = np.random.default_rng(5)
    gs, gh, gf = [], [], []
    for _ in range(200):
        Sig = market(rng2)
        X = rng2.normal(size=(120, 40)) @ np.linalg.cholesky(Sig).T
        S = np.cov(X, rowvar=False)
        F, lab = block_filter(S, 5)
        w = hrp(S)
        _, _, D, _ = implied(S, w)
        gs.append(block_gap(S, lab)); gh.append(block_gap(S + D, lab)); gf.append(block_gap(F, lab))
    print(f"\n  distance to the nearest block-constant correlation matrix, median of 200")
    print(f"    sample covariance {np.median(gs):.3f}   HRP implied {np.median(gh):.3f}"
          f"   block-filtered {np.median(gf):.3f}")
    check("HRP's implied covariance is FURTHER from block structure than the sample",
          np.median(gh) > np.median(gs),
          f"{np.median(gh):.3f} against {np.median(gs):.3f}")
    check("the HRP correction has rank two in every draw", rank_ok)
    check("it lies in span{w, r}", span_ok < 1e-12, f"largest residual {span_ok:.1e}")

    # --- Table 2: which shrinkage is HRP
    rng = np.random.default_rng(101)
    def fit_flat(S, w):
        Dg = np.diag(np.diag(S))
        g = np.linspace(0, 1, 201)
        v = [np.abs(min_var(x * S + (1 - x) * Dg, 1e-10) - w).sum() for x in g]
        return g[int(np.argmin(v))], min(v)

    def fit_cc(S, w):
        d = np.sqrt(np.diag(S))
        R = S / np.outer(d, d)
        rb = (R.sum() - len(R)) / (len(R) ** 2 - len(R))
        F = np.full_like(R, rb)
        np.fill_diagonal(F, 1.0)
        F = d[:, None] * F * d[None, :]
        g = np.linspace(0, 1, 201)
        v = [np.abs(min_var(x * S + (1 - x) * F, 1e-10) - w).sum() for x in g]
        return g[int(np.argmin(v))], min(v)

    def fit_ridge(S, w):
        tr = np.trace(S) / S.shape[0]
        g = np.geomspace(1e-6, 1e3, 200) * tr
        v = [np.abs(min_var(S + a * np.eye(S.shape[0])) - w).sum() for a in g]
        return g[int(np.argmin(v))] / tr, min(v)

    def lca(Z, n, order):
        L = np.zeros((n, n), dtype=int)
        def rec(idx, depth):
            if len(idx) <= 1:
                return
            h = len(idx) // 2
            l, r = idx[:h], idx[h:]
            for i in l:
                for j in r:
                    L[i, j] = L[j, i] = depth + 1
            rec(l, depth + 1)
            rec(r, depth + 1)
        rec(order, 0)
        return L

    def fit_taper(S, w, L):
        lv = sorted(set(L[L > 0].tolist()))
        def make(t):
            T = np.ones_like(S)
            for k, l in enumerate(lv):
                T[L == l] = t[k]
            return S * T
        f = lambda t: np.abs(min_var(make(np.clip(t, 0, 1.5)), 1e-10) - w).sum()
        best = None
        for x0 in (np.full(len(lv), .5), np.linspace(.1, .9, len(lv)), np.full(len(lv), .05)):
            r = minimize(f, x0, method="Nelder-Mead",
                         options=dict(maxiter=4000, xatol=1e-4, fatol=1e-8))
            if best is None or r.fun < best.fun:
                best = r
        return np.clip(best.x, 0, 1.5), best.fun, lv

    rows = {"inverse variance (no fit)": [], "diagonal target": [], "constant correlation": [],
            "ridge": [], "tree taper, one dial per level": []}
    pars = {"diagonal target": [], "constant correlation": [], "ridge": []}
    taps = []
    for _ in range(120):
        Sig = market(rng)
        X = rng.normal(size=(120, 40)) @ np.linalg.cholesky(Sig).T
        S = np.cov(X, rowvar=False)
        Z, order = tree_order(S)
        w = hrp(S)
        rows["inverse variance (no fit)"].append(np.abs(w - inv_var(S)).sum())
        l, d = fit_flat(S, w); rows["diagonal target"].append(d); pars["diagonal target"].append(l)
        l, d = fit_cc(S, w); rows["constant correlation"].append(d); pars["constant correlation"].append(l)
        a, d = fit_ridge(S, w); rows["ridge"].append(d); pars["ridge"].append(a)
        t, d, lv = fit_taper(S, w, lca(Z, 40, order)); rows["tree taper, one dial per level"].append(d)
        taps.append(t)
    print("\n  Table 2  best fit to the HRP weights within each family, L1, median of 120")
    print(f"  {'family':34s}{'L1 distance':>13s}{'at':>22s}")
    for k in rows:
        p = f"lambda = {np.median(pars[k]):.3f}" if k in pars and k != "ridge" else (
            f"a/tr(S)/n = {np.median(pars['ridge']):.3g}" if k == "ridge" else "")
        print(f"  {k:34s}{np.median(rows[k]):13.4f}{p:>22s}")
    print(f"  fitted taper multipliers by tree level (level 1 is the top split): "
          + ", ".join(f"{np.median(np.array(taps)[:, k]):.3f}" for k in range(np.array(taps).shape[1])))
    base = np.median(rows["inverse variance (no fit)"])
    best = min(np.median(rows[k]) for k in rows if k != "inverse variance (no fit)")
    check("no shrinkage family reproduces the HRP weights", best > 0.15,
          f"the best fit still misplaces {best:.1%} of the portfolio")
    check("fitting a shrinkage buys little over not fitting at all", best > 0.7 * base,
          f"best fit {best:.4f} against an unfitted inverse-variance baseline of {base:.4f}")

    # --- The parsimony bound: a free taper is underdetermined, so it fits.
    from scipy.optimize import least_squares
    rng = np.random.default_rng(404)
    iu = np.triu_indices(40, 1)
    miss = []
    for _ in range(10):
        Sig = market(rng)
        X = rng.normal(size=(120, 40)) @ np.linalg.cholesky(Sig).T
        S = np.cov(X, rowvar=False)
        w = hrp(S)
        def resid(t):
            T = np.ones((40, 40))
            T[iu] = t
            T = np.triu(T, 1)
            T = T + T.T + np.eye(40)
            return min_var(S * T, 1e-10) - w
        r = least_squares(resid, np.ones(len(iu[0])), method="trf",
                          xtol=1e-14, ftol=1e-14, max_nfev=200)
        miss.append(np.abs(resid(r.x)).sum())
    print(f"\n  free symmetric taper: {40*39//2} parameters against 40 constraints, "
          f"median L1 miss {np.median(miss):.1e}")
    check("a free taper reproduces HRP, so the negative result is about parsimony",
          np.median(miss) < 1e-5, f"median miss {np.median(miss):.1e}")

    # --- Table 3: the bridge
    try:
        import os, sys
        sib = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "allocation")
        if os.path.isdir(sib):
            sys.path.insert(0, sib)
        from allocation._schur.bridge import bridge_weights, bisection_tree
        from allocation._schur.seriation import seriate
    except Exception as exc:
        print(f"\n  Table 3 skipped, the allocation package is not importable: {exc}")
    else:
        rng = np.random.default_rng(11)
        G = np.linspace(0, 1, 11)
        dd = {g: [] for g in G}
        cc = {g: [] for g in G}
        vv = {g: [] for g in G}
        for _ in range(200):
            Sig = market(rng)
            X = rng.normal(size=(120, 40)) @ np.linalg.cholesky(Sig).T
            S = np.cov(X, rowvar=False)
            tr = bisection_tree(seriate(S)[0], leaf_size=1)
            for g in G:
                w = bridge_weights(S, tr, gamma=float(g), eta=1.0, split="dial")
                d, c, _, _ = implied(S, w)
                dd[g].append(d); cc[g].append(c); vv[g].append(float(w @ Sig @ w))
        print("\n  Table 3  along the bridge, median of 200")
        print(f"  {'gamma':>7s}{'distortion':>13s}{'lmin/lmax':>12s}{'true variance':>16s}")
        for g in G:
            print(f"  {g:7.1f}{np.median(dd[g]):13.3f}{np.median(cc[g]):12.4f}{np.median(vv[g]):16.5f}")
        mono = all(np.median(dd[G[i]]) >= np.median(dd[G[i + 1]]) - 1e-12 for i in range(len(G) - 1))
        check("distortion is monotone in gamma", mono)
        check("distortion vanishes at the far end", np.median(dd[1.0]) < 1e-12)

        # The crossing into the positive definite cone, on a finer grid, and
        # whether it selects a gamma. These numbers appear in the prose, so
        # they are computed here rather than asserted.
        fine = np.linspace(0.0, 1.0, 101)
        cross, best = [], []
        rng2 = np.random.default_rng(23)
        for _ in range(200):
            Sig = market(rng2)
            X = rng2.normal(size=(120, 40)) @ np.linalg.cholesky(Sig).T
            S = np.cov(X, rowvar=False)
            tr = bisection_tree(seriate(S)[0], leaf_size=1)
            rat, var = [], []
            for g in fine:
                w = bridge_weights(S, tr, gamma=float(g), eta=1.0, split="dial")
                _, c, _, _ = implied(S, w)
                rat.append(c); var.append(float(w @ Sig @ w))
            rat = np.asarray(rat)
            ok = np.where(rat > 0)[0]
            cross.append(fine[ok[0]] if len(ok) else np.nan)
            best.append(fine[int(np.argmin(var))])
        cross = np.asarray(cross, dtype=float); best = np.asarray(best)
        fin = np.isfinite(cross)
        q = np.nanpercentile(cross, [10, 50, 90])
        corr = float(np.corrcoef(cross[fin], best[fin])[0, 1])
        print(f"\n  positive-definite crossing, 101-point grid, 200 markets")
        print(f"    crosses in {int(fin.sum())} of {len(cross)} markets")
        print(f"    median crossing {q[1]:.2f}, deciles {q[0]:.2f} and {q[2]:.2f}")
        print(f"    median variance-minimising gamma {np.median(best):.2f}")
        print(f"    correlation between the two {corr:+.3f}")
        check("the crossing is a stable interval", 0.2 < q[0] and q[2] < 0.7,
              f"deciles {q[0]:.2f} to {q[2]:.2f}")
        check("the crossing does not select the best gamma", abs(corr) < 0.2,
              f"correlation {corr:+.3f} against a median optimum of {np.median(best):.2f}")

    # --- Table 4: the controlled comparison
    rng = np.random.default_rng(3)
    print("\n  Table 4  one market, one covariance, two allocators and two estimators")
    print(f"  {'T/n':>6s}{'HRP on raw':>13s}{'min-var on raw':>17s}{'min-var on filtered':>22s}")
    kept = {}
    for T in (20, 30, 40, 60, 120, 400):
        a, b, c = [], [], []
        for _ in range(250):
            Sig = market(rng)
            X = rng.normal(size=(T, 40)) @ np.linalg.cholesky(Sig).T
            S = np.cov(X, rowvar=False)
            F, _ = block_filter(S, 5)
            wh = hrp(S); a.append(wh @ Sig @ wh)
            wm = min_var(S, 1e-8); b.append(wm @ Sig @ wm)
            wf = min_var(F); c.append(wf @ Sig @ wf)
        kept[T] = (np.median(a), np.median(b), np.median(c))
        print(f"  {T/40:6.2f}{kept[T][0]:13.4f}{kept[T][1]:17.4f}{kept[T][2]:22.4f}")
    check("the filtered optimizer beats HRP at every sample size",
          all(v[2] < v[0] for v in kept.values()))
    check("HRP beats the raw optimizer only near the singular point",
          sum(1 for v in kept.values() if v[0] < v[1]) <= 2)

    # --- The rank-deficient corner, which Table 4 does not reach.
    print("\n  Table 5  below the table's range, n = 40")
    print(f"  {'T/n':>6s}{'HRP':>10s}{'filtered':>11s}{'beats HRP':>11s}"
          f"{'inv var':>10s}{'beats HRP':>11s}")
    deep = {}
    for T in (4, 8, 12, 20):
        a, b, c = [], [], []
        for _ in range(300):
            Sig = market(rng)
            X = rng.normal(size=(T, 40)) @ np.linalg.cholesky(Sig).T
            S = np.cov(X, rowvar=False)
            F, _ = block_filter(S, 5)
            wh = hrp(S); a.append(wh @ Sig @ wh)
            wf = min_var(F, 1e-10); b.append(wf @ Sig @ wf)
            iv = inv_var(S); c.append(iv @ Sig @ iv)
        a, b, c = map(np.array, (a, b, c))
        deep[T] = (np.median(a), np.median(b), float(np.mean(b < a)),
                   np.median(c), float(np.mean(c < a)))
        print(f"  {T/40:6.2f}{deep[T][0]:10.4f}{deep[T][1]:11.4f}{deep[T][2]:10.0%}"
              f"{deep[T][3]:11.4f}{deep[T][4]:10.0%}")
    check("at T/n = 1/10 the filtered optimizer is WORSE than HRP",
          deep[4][1] > deep[4][0] and deep[4][2] < 0.5,
          f"filtered {deep[4][1]:.4f} against HRP {deep[4][0]:.4f}, wins {deep[4][2]:.0%}")
    check("the lead reverses between T/n = 1/10 and 1/5",
          deep[8][1] < deep[8][0] < deep[4][1],
          f"ratio HRP/filtered {deep[8][0]/deep[8][1]:.2f} at 1/5")
    check("inverse variance beats HRP at every ratio from 1/10 to 1/2",
          all(v[4] > 0.5 for v in deep.values()),
          "win rates " + ", ".join(f"{v[4]:.0%}" for v in deep.values()))


if __name__ == "__main__":
    print("Certificates for 'The implied covariance of an allocator'")
    part1()
    part2()
    part3()
    bad = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(bad)} of {len(CHECKS)} checks passed")
    if bad:
        for n in bad:
            print(f"  FAILED: {n}")
        raise SystemExit(1)
