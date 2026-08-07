"""Monte Carlo for the sample-covariance section of "Neither End of the Bridge".

Model: Sigma-hat_T = (1/T) sum_t x_t x_t', x_t iid N(0, Sigma), for the
four-asset family of the paper. L_T(gamma) = w' Sigma w is the realized loss
of the implemented recursion; w_MV = Sigma-hat^{-1} 1 (normalized) is the
sample minimum-variance benchmark, which the implemented recursion at
gamma = 1 does NOT reproduce on non-exchangeable input.

The mean of L_T is not a usable summary for the raw implemented recursion:
the fitness sum nu(A_gamma) + nu(D_gamma) can cross zero at small T (a
sampled augmented block can be indefinite), the split has a simple pole
there, L_T a second-order one, and by the paper's pole criterion the
expectation is infinite. This script prints (1) the paper's median table
with the sample-MV benchmark and the tail column, (2) the mean-vs-median
stability diagnostic, and (3) the small-ball frequency of the fitness sum
at the full-coupling end. Requires numpy.

Run:  python wishart_experiment.py
"""
import numpy as np

VOL = np.array([1.0, 1.0, 2.0, 2.0])
RHO = np.array([[1.0, 0.5, 0.3, 0.3],
                [0.5, 1.0, 0.3, 0.3],
                [0.3, 0.3, 1.0, 0.5],
                [0.3, 0.3, 0.5, 1.0]])
SIGMA = np.outer(VOL, VOL) * RHO
I2 = np.eye(2)


def _inv2(X):
    det = X[:, 0, 0] * X[:, 1, 1] - X[:, 0, 1] * X[:, 1, 0]
    out = np.empty_like(X)
    out[:, 0, 0], out[:, 1, 1] = X[:, 1, 1], X[:, 0, 0]
    out[:, 0, 1], out[:, 1, 0] = -X[:, 0, 1], -X[:, 1, 0]
    return out / det[:, None, None]


def _aug(P, Bm, Q, g):
    BQi = Bm @ _inv2(Q)
    a0 = P - g * BQi @ np.swapaxes(Bm, 1, 2)
    r = I2[None] - g * BQi
    M = _inv2(r) @ a0
    return 0.5 * (M + np.swapaxes(M, 1, 2))


def _naive(X):
    iw = 1.0 / np.stack([X[:, 0, 0], X[:, 1, 1]], axis=1)
    w = iw / iw.sum(axis=1, keepdims=True)
    v = np.einsum('ni,nij,nj->n', w, X, w)
    return w, v


def batch_weights(S, g):
    """The n = 4 two-block collapse recursion, vectorized over draws."""
    A, B, D = S[:, :2, :2], S[:, :2, 2:], S[:, 2:, 2:]
    Aa = _aug(A, B, D, g) if g else A
    Da = _aug(D, np.swapaxes(B, 1, 2), A, g) if g else D
    wA, vA = _naive(Aa)
    wD, vD = _naive(Da)
    aL = vD / (vA + vD)
    return np.concatenate([aL[:, None] * wA, (1 - aL)[:, None] * wD], axis=1)


def fitness_sum(S, g):
    A, B, D = S[:, :2, :2], S[:, :2, 2:], S[:, 2:, 2:]
    Aa = _aug(A, B, D, g) if g else A
    Da = _aug(D, np.swapaxes(B, 1, 2), A, g) if g else D
    return _naive(Aa)[1] + _naive(Da)[1]


def sample_mv(S):
    iv = np.linalg.solve(S, np.ones((S.shape[0], 4))[..., None])[..., 0]
    return iv / iv.sum(axis=1, keepdims=True)


def main():
    L = np.linalg.cholesky(SIGMA)
    gammas = np.round(np.arange(0, 1.0001, 0.02), 4)

    # (1) the paper's table: medians, sample-MV benchmark, tail frequency
    rng = np.random.default_rng(20260802)
    N = 200_000
    print(f"{'T':>4} {'g*':>5} {'M(0)':>7} {'M(g*)':>7} {'M(1)':>7}"
          f" {'medMV':>7} {'P[L(1)>10]':>10}")
    for T in (15, 25, 50, 100, 250):
        X = rng.standard_normal((N, T, 4)) @ L.T
        Sh = np.einsum('nti,ntj->nij', X, X) / T
        med = {}
        for g in gammas:
            w = batch_weights(Sh, g)
            med[g] = np.median(np.einsum('ni,ij,nj->n', w, SIGMA, w))
        gstar = min(gammas, key=lambda g: med[g])
        w1 = batch_weights(Sh, 1.0)
        L1 = np.einsum('ni,ij,nj->n', w1, SIGMA, w1)
        wmv = sample_mv(Sh)
        Lmv = np.einsum('ni,ij,nj->n', wmv, SIGMA, wmv)
        print(f"{T:>4} {gstar:>5.2f} {med[0.0]:>7.4f} {med[gstar]:>7.4f}"
              f" {med[1.0]:>7.4f} {np.median(Lmv):>7.4f} {(L1 > 10).mean():>10.4f}")

    # (2) mean never settles, median does (T = 25, gamma = 1)
    print("\nmean vs median of L_T(1) at T = 25 (seed 7):")
    for N2 in (4_000, 40_000, 400_000):
        rng2 = np.random.default_rng(7)
        X = rng2.standard_normal((N2, 25, 4)) @ L.T
        Sh = np.einsum('nti,ntj->nij', X, X) / 25
        w = batch_weights(Sh, 1.0)
        V = np.einsum('ni,ij,nj->n', w, SIGMA, w)
        print(f"  N = {N2:>7,}: mean = {V.mean():>12.1f}   median = {np.median(V):.4f}")

    # (3) small-ball frequency of the fitness sum at the full-coupling end
    print("\nP[|fitness sum| <= eps]/eps at gamma = 1 (linear scaling supports")
    print("the pole criterion's small-ball hypothesis):")
    for T in (15, 25):
        rng3 = np.random.default_rng(13)
        X = rng3.standard_normal((400_000, T, 4)) @ L.T
        Sh = np.einsum('nti,ntj->nij', X, X) / T
        Dn = np.abs(fitness_sum(Sh, 1.0))
        row = "  ".join(f"{(Dn <= e).mean() / e:.3f}" for e in (3e-2, 1e-2, 3e-3, 1e-3, 3e-4))
        print(f"  T = {T:>3}: {row}   (eps = 3e-2 .. 3e-4)")


if __name__ == "__main__":
    main()
