"""Exact certificate for the four-asset example in "Neither End of the Bridge".

Proposition (exact example). n = 4 assets in two blocks {1,2}, {3,4};
volatilities (1,1,2,2); within-block correlation 1/2; cross-block correlation
3/10. The covariance estimate errs in the (1,3) entry by +/- (3/20)*2 with
probability 1/2 each. For the Schur-complementary recursion (gamma-scaled
Schur augmentation, inverse-naive-variance splits), the expected out-of-sample
variance F satisfies, in exact rational arithmetic,

    F(9/10) < F(0)   and   F(9/10) < F(1),

hence every minimizer of F on [0,1] is strictly interior.

Every step of the recursion is a field operation, so F at rational arguments
is an exact rational number and the two inequalities are certificates. This
script needs only the Python standard library (fractions). If numpy and the
`allocation` package are importable, it additionally cross-checks the closed
form against allocation._schur.coupling.compute_weights and certifies that
all augmented blocks stay positive definite on both noise branches.

Run:  python neither_end_certificate.py
"""
from fractions import Fraction as Fr

# ------------------------------------------------------------- exact 2x2 core
def mmul(X, Y):
    return [[sum(X[i][k] * Y[k][j] for k in range(len(Y))) for j in range(len(Y[0]))]
            for i in range(len(X))]


def mtrans(X):
    return [list(r) for r in zip(*X)]


def minv2(X):
    (a, b), (c, d) = X
    det = a * d - b * c
    return [[d / det, -b / det], [-c / det, a / det]]


def msub(X, Y):
    return [[X[i][j] - Y[i][j] for j in range(len(X[0]))] for i in range(len(X))]


def mscale(X, s):
    return [[s * X[i][j] for j in range(len(X[0]))] for i in range(len(X))]


def sym(X):
    return [[(X[i][j] + X[j][i]) / 2 for j in range(len(X))] for i in range(len(X))]


I2 = [[Fr(1), Fr(0)], [Fr(0), Fr(1)]]


def aug(A, B, D, g):
    """sym( (I - g B D^-1)^-1 (A - g B D^-1 B^T) ): the Schur augmentation."""
    BDi = mmul(B, minv2(D))
    a0 = msub(A, mscale(mmul(BDi, mtrans(B)), g))
    r = msub(I2, mscale(BDi, g))
    return sym(mmul(minv2(r), a0))


def naive_var(X):
    w = [1 / X[i][i] for i in range(len(X))]
    s = sum(w)
    w = [wi / s for wi in w]
    return sum(w[i] * X[i][j] * w[j] for i in range(len(X)) for j in range(len(X)))


def weights(S, g):
    """The n=4 two-block Schur recursion, exactly."""
    A = [row[:2] for row in S[:2]]
    D = [row[2:] for row in S[2:]]
    B = [row[2:] for row in S[:2]]
    Aa = aug(A, B, D, g) if g != 0 else A
    Da = aug(D, mtrans(B), A, g) if g != 0 else D
    vL, vR = naive_var(Aa), naive_var(Da)
    aL = vR / (vL + vR)
    return [aL * (1 / Aa[0][0]) / (1 / Aa[0][0] + 1 / Aa[1][1]),
            aL * (1 / Aa[1][1]) / (1 / Aa[0][0] + 1 / Aa[1][1]),
            (1 - aL) * (1 / Da[0][0]) / (1 / Da[0][0] + 1 / Da[1][1]),
            (1 - aL) * (1 / Da[1][1]) / (1 / Da[0][0] + 1 / Da[1][1])]


# ------------------------------------------------------------------ the model
RW, RC, S_ = Fr(1, 2), Fr(3, 10), Fr(2)
TAU = Fr(3, 20)


def Sigma(eps=Fr(0)):
    v = [Fr(1), Fr(1), S_, S_]
    C = [[Fr(1), RW, RC, RC], [RW, Fr(1), RC, RC],
         [RC, RC, Fr(1), RW], [RC, RC, RW, Fr(1)]]
    Sg = [[v[i] * v[j] * C[i][j] for j in range(4)] for i in range(4)]
    Sg[0][2] += eps * S_
    Sg[2][0] += eps * S_
    return Sg


ST = Sigma()


def V(g, eps):
    w = weights(Sigma(eps), g)
    return sum(w[i] * ST[i][j] * w[j] for i in range(4) for j in range(4))


def F(g):
    return (V(g, TAU) + V(g, -TAU)) / 2


# --------------------------- continuity certificate (exact, all gamma in [0,1])
# F(gamma) is a rational function of gamma; the endpoint-exclusion argument
# needs F continuous on the WHOLE interval, i.e. no denominator vanishes for
# any gamma in [0,1], not merely on a grid. Every denominator below is a
# polynomial in gamma with Fraction coefficients; strict positivity on [0,1]
# is certified exactly: if all Bernstein coefficients of p on [a,b] are
# positive then p > 0 on [a,b]; otherwise bisect (terminates for strictly
# positive p). Polynomials are lists of Fractions, lowest degree first.

def padd(p, q):
    n = max(len(p), len(q))
    return [(p[i] if i < len(p) else 0) + (q[i] if i < len(q) else 0) for i in range(n)]


def psub(p, q):
    return padd(p, [-c for c in q])


def pmul(p, q):
    out = [Fr(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return out


def peval(p, x):
    v = Fr(0)
    for c in reversed(p):
        v = v * x + c
    return v


def bernstein_positive(p, a=Fr(0), b=Fr(1), depth=0):
    """Exact certificate that polynomial p > 0 on [a, b]."""
    if depth > 40:
        raise RuntimeError("subdivision limit; polynomial may touch zero")
    # q(t) = p(a + (b-a) t) on t in [0,1]
    q = [Fr(0)]
    shift, scale = [a, b - a], [Fr(1)]
    for c in p:
        q = padd(q, [c * s for s in scale])
        scale = pmul(scale, shift)
    m = len(q) - 1
    from math import comb
    bern = []
    for i in range(m + 1):
        bern.append(sum(Fr(comb(i, j), comb(m, j)) * q[j] for j in range(i + 1)))
    if all(c > 0 for c in bern):
        return True
    if peval(p, a) <= 0 or peval(p, b) <= 0:
        raise RuntimeError(f"denominator not positive at an endpoint of [{a},{b}]")
    mid = (a + b) / 2
    return bernstein_positive(p, a, mid, depth + 1) and bernstein_positive(p, mid, b, depth + 1)


def continuity_certificate(eps):
    """Certify, for all gamma in [0,1]: det r > 0 for both blocks, and both
    augmented blocks SPD (positive diagonal numerators and determinant
    numerators, given det r > 0). This makes F(. , eps) continuous on [0,1]."""
    S = Sigma(eps)
    A = [row[:2] for row in S[:2]]
    D = [row[2:] for row in S[2:]]
    B = [row[2:] for row in S[:2]]
    G = [Fr(0), Fr(1)]                      # the polynomial 'gamma'
    ok = True
    for (P, Bm, Q) in ((A, B, D), (D, [list(r) for r in zip(*B)], A)):
        Qi = minv2(Q)
        BQi = mmul(Bm, Qi)                  # constant matrix
        # r(g) = I - g BQi ; det r = 1 - g tr + g^2 det
        t = BQi[0][0] + BQi[1][1]
        d = BQi[0][0] * BQi[1][1] - BQi[0][1] * BQi[1][0]
        detr = [Fr(1), -t, d]
        ok &= bernstein_positive(detr)
        # numerator of A_gamma: sym( adj(r) @ (P - g BQi Bm^T) ), denominator det r
        BQiBt = mmul(BQi, mtrans(Bm))
        Ac = [[psub([P[i][j]], pmul(G, [BQiBt[i][j]])) for j in range(2)] for i in range(2)]
        # adj of r = [[r11, -r01], [-r10, r00]] with r = I - g BQi
        r00 = psub([Fr(1)], pmul(G, [BQi[0][0]]))
        r01 = pmul(G, [-BQi[0][1]])
        r10 = pmul(G, [-BQi[1][0]])
        r11 = psub([Fr(1)], pmul(G, [BQi[1][1]]))
        adjr = [[r11, [-c for c in r01]], [[-c for c in r10], r00]]
        N = [[padd(pmul(adjr[i][0], Ac[0][j]), pmul(adjr[i][1], Ac[1][j]))
              for j in range(2)] for i in range(2)]
        Ns = [[ [ (N[i][j][k] + N[j][i][k]) / 2 for k in range(len(N[i][j])) ]
               for j in range(2)] for i in range(2)]
        # diagonals of A_gamma = Ns[i][i]/detr : positive iff numerator positive
        ok &= bernstein_positive(Ns[0][0])
        ok &= bernstein_positive(Ns[1][1])
        # det A_gamma = det(Ns)/detr^2 : positive iff det(Ns) positive
        detN = psub(pmul(Ns[0][0], Ns[1][1]), pmul(Ns[0][1], Ns[1][0]))
        ok &= bernstein_positive(detN)
    return ok


def main():
    ghat = Fr(9, 10)
    F0, Fh, F1 = F(Fr(0)), F(ghat), F(Fr(1))
    print(f"exact values (as floats): F(0) = {float(F0):.6f}, "
          f"F(9/10) = {float(Fh):.6f}, F(1) = {float(F1):.6f}")
    print(f"F(0) - F(9/10) = {float(F0 - Fh):.6e}  (exact rational, positive: {F0 - Fh > 0})")
    print(f"F(1) - F(9/10) = {float(F1 - Fh):.6e}  (exact rational, positive: {F1 - Fh > 0})")
    assert F0 - Fh > 0 and F1 - Fh > 0
    assert V(Fr(0), TAU) == V(Fr(0), -TAU) == V(Fr(0), Fr(0))  # blindness lemma
    assert continuity_certificate(TAU) and continuity_certificate(-TAU)
    print("\ncontinuity certificate: every denominator polynomial is strictly "
          "positive on [0,1]\n(Bernstein expansion, exact rational arithmetic, "
          "both noise branches), so F is\ncontinuous on [0,1], a minimizer "
          "exists, and min F <= F(9/10) < min(F(0), F(1)).")
    print("\nQED: every minimizer of F on [0,1] is strictly interior; "
          "F(0) is exactly noise-free.")

    # optional: cross-check against the allocation package and certify SPD
    try:
        import numpy as np
        from allocation._schur.coupling import compute_weights, schur_augmentation, _is_spd
    except ImportError:
        print("\n(numpy/allocation not importable: package cross-check skipped; "
              "the certificate above is stdlib-only and complete)")
        return
    for g in [0.0, 0.3, 0.55, 0.9, 1.0]:
        for e in (float(TAU), 0.0, -float(TAU)):
            Sf = np.array([[float(x) for x in row]
                           for row in Sigma(Fr(e).limit_denominator(10 ** 6))])
            wp = compute_weights(np.arange(4), Sf, g, force_spd=True)
            we = [float(x) for x in weights(Sigma(Fr(e).limit_denominator(10 ** 6)),
                                            Fr(g).limit_denominator(10 ** 6))]
            assert np.abs(np.asarray(we) - wp).max() < 1e-10, (g, e)
    for g in np.linspace(0, 1, 41):
        for e in (float(TAU), -float(TAU)):
            Sf = np.array([[float(x) for x in row]
                           for row in Sigma(Fr(e).limit_denominator(10 ** 6))])
            A, B, D = Sf[:2, :2], Sf[:2, 2:], Sf[2:, 2:]
            assert _is_spd(schur_augmentation(A, B, D, float(g)))
            assert _is_spd(schur_augmentation(D, B.T, A, float(g)))
    print("\ncross-check: matches allocation._schur.coupling.compute_weights (1e-10); "
          "all augmented blocks SPD on both branches.")


if __name__ == "__main__":
    main()
