"""Monte Carlo for the sample-covariance table in "Neither End of the Bridge"
(section on whole-matrix noise).

Model: Sigma-hat = (1/T) sum_t x_t x_t', x_t iid N(0, Sigma), for the
four-asset family of the paper. For each T, the same 200,000 draws are shared
across the coupling grid (step 0.02), and the expected out-of-sample variance
F(gamma) = E[w' Sigma w] is estimated with w from the collapse recursion.
Paired differences against the argmin give the standard errors quoted in the
paper. Requires numpy.

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


def batch_weights(S, g):
    """The n = 4 two-block collapse recursion, vectorized over draws."""
    A, B, D = S[:, :2, :2], S[:, :2, 2:], S[:, 2:, 2:]

    def inv2(X):
        det = X[:, 0, 0] * X[:, 1, 1] - X[:, 0, 1] * X[:, 1, 0]
        out = np.empty_like(X)
        out[:, 0, 0], out[:, 1, 1] = X[:, 1, 1], X[:, 0, 0]
        out[:, 0, 1], out[:, 1, 0] = -X[:, 0, 1], -X[:, 1, 0]
        return out / det[:, None, None]

    def aug(P, Bm, Q):
        BQi = Bm @ inv2(Q)
        a0 = P - g * BQi @ np.swapaxes(Bm, 1, 2)
        r = I2[None] - g * BQi
        M = inv2(r) @ a0
        return 0.5 * (M + np.swapaxes(M, 1, 2))

    Aa = aug(A, B, D) if g else A
    Da = aug(D, np.swapaxes(B, 1, 2), A) if g else D

    def naive(X):
        iw = 1.0 / np.stack([X[:, 0, 0], X[:, 1, 1]], axis=1)
        w = iw / iw.sum(axis=1, keepdims=True)
        v = np.einsum('ni,nij,nj->n', w, X, w)
        return w, v

    wA, vA = naive(Aa)
    wD, vD = naive(Da)
    aL = vD / (vA + vD)
    return np.concatenate([aL[:, None] * wA, (1 - aL)[:, None] * wD], axis=1)


def main():
    rng = np.random.default_rng(20260802)
    N = 200_000
    L = np.linalg.cholesky(SIGMA)
    gammas = np.round(np.arange(0, 1.0001, 0.02), 4)
    # The mean of the raw recursion's out-of-sample variance is infinite on
    # part of the bridge at small T (near-singular couplings land inside with
    # positive density and the pole is non-integrable), so the paper's table
    # reports medians, which are stable, with the tail frequency separately.
    print(f"{'T':>4}  {'gamma*':>6}  {'medF(0)':>8}  {'medF(g*)':>8}"
          f"  {'medF(1)':>8}  {'P[V(g*)<V(0)]':>13}  {'P[V(1)>10]':>10}")
    for T in (15, 25, 50, 100, 250):
        X = rng.standard_normal((N, T, 4)) @ L.T
        S_hat = np.einsum('nti,ntj->nij', X, X) / T
        V = {}
        for g in gammas:
            w = batch_weights(S_hat, g)
            V[g] = np.einsum('ni,ij,nj->n', w, SIGMA, w)
        med = {g: np.median(V[g]) for g in gammas}
        gstar = min(gammas, key=lambda g: med[g])
        winrate = (V[gstar] < V[0.0]).mean()
        tail = (V[1.0] > 10).mean()
        print(f"{T:>4}  {gstar:>6.2f}  {med[0.0]:>8.4f}  {med[gstar]:>8.4f}"
              f"  {med[1.0]:>8.4f}  {winrate:>13.4f}  {tail:>10.4f}")


if __name__ == "__main__":
    main()
