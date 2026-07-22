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

    print()
    jet_certificates()
    print()
    family_checks()

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





# ---------------- exact boundary derivatives via dual-number jets (Example 1,
# Remark on the structural cancellation). A dual number a + b*eps with
# eps^2 = 0 propagates exact values and first derivatives in gamma through the
# recursion; all arithmetic is over exact rationals.

class Dual:
    __slots__ = ('a', 'b')
    def __init__(self, a, b=Fr(0)):
        self.a, self.b = Fr(a), Fr(b)
    @staticmethod
    def of(x):
        return x if isinstance(x, Dual) else Dual(x)
    def __add__(s, o):
        o = Dual.of(o); return Dual(s.a + o.a, s.b + o.b)
    __radd__ = __add__
    def __neg__(s): return Dual(-s.a, -s.b)
    def __sub__(s, o):
        o = Dual.of(o); return Dual(s.a - o.a, s.b - o.b)
    def __rsub__(s, o): return Dual.of(o) - s
    def __mul__(s, o):
        o = Dual.of(o); return Dual(s.a * o.a, s.a * o.b + s.b * o.a)
    __rmul__ = __mul__
    def __truediv__(s, o):
        o = Dual.of(o)
        return Dual(s.a / o.a, (s.b * o.a - s.a * o.b) / (o.a * o.a))
    def __rtruediv__(s, o): return Dual.of(o) / s


def _jet_weights4(S):
    """weights() with gamma = 0 + eps, entries lifted to Dual."""
    GAMMA = Dual(Fr(0), Fr(1))
    D_ = Dual.of
    A = [[D_(S[i][j]) for j in range(2)] for i in range(2)]
    D = [[D_(S[2 + i][2 + j]) for j in range(2)] for i in range(2)]
    B = [[D_(S[i][2 + j]) for j in range(2)] for i in range(2)]
    I2d = [[Dual(1), Dual(0)], [Dual(0), Dual(1)]]

    def dmmul(X, Y):
        return [[sum((X[i][k] * Y[k][j] for k in range(2)), Dual(0))
                 for j in range(2)] for i in range(2)]

    def dinv2(X):
        (a, b), (c, d) = X
        det = a * d - b * c
        return [[d / det, Dual(0) - b / det], [Dual(0) - c / det, a / det]]

    def daug(P, Bm, Q):
        BQi = dmmul(Bm, dinv2(Q))
        a0 = [[P[i][j] - GAMMA * dmmul(BQi, [list(r) for r in zip(*Bm)])[i][j]
               for j in range(2)] for i in range(2)]
        r = [[I2d[i][j] - GAMMA * BQi[i][j] for j in range(2)] for i in range(2)]
        M = dmmul(dinv2(r), a0)
        return [[(M[i][j] + M[j][i]) / Dual(2) for j in range(2)] for i in range(2)]

    def dnaive(X):
        w = [Dual(1) / X[0][0], Dual(1) / X[1][1]]
        s = w[0] + w[1]
        w = [wi / s for wi in w]
        return sum((w[i] * X[i][j] * w[j] for i in range(2) for j in range(2)), Dual(0))

    Aa = daug(A, B, D)
    Da = daug(D, [list(r) for r in zip(*B)], A)
    vL, vR = dnaive(Aa), dnaive(Da)
    aL = vR / (vL + vR)
    iA0, iA1 = Dual(1) / Aa[0][0], Dual(1) / Aa[1][1]
    iD0, iD1 = Dual(1) / Da[0][0], Dual(1) / Da[1][1]
    return [aL * iA0 / (iA0 + iA1), aL * iA1 / (iA0 + iA1),
            (Dual(1) - aL) * iD0 / (iD0 + iD1), (Dual(1) - aL) * iD1 / (iD0 + iD1)]


def boundary_derivative(SigTrue, SigPlus, SigMinus):
    """Exact (F(0), F'(0)) for two-point noise, via jets."""
    def Vjet(S):
        w = _jet_weights4(S)
        return sum((w[i] * Dual(SigTrue[i][j]) * w[j]
                    for i in range(4) for j in range(4)), Dual(0))
    Fj = (Vjet(SigPlus) + Vjet(SigMinus)) / Dual(2)
    return Fj.a, Fj.b


def jet_certificates():
    # Example 1 (boundary): A = D = [[1,-1/2],[-1/2,2]], B = ones/10, E = e1 e1^T
    A = [[Fr(1), Fr(-1, 2)], [Fr(-1, 2), Fr(2)]]
    Bx = [[Fr(1, 10)] * 2, [Fr(1, 10)] * 2]

    def SigB(tau, sign):
        Bh = [[Bx[i][j] for j in range(2)] for i in range(2)]
        Bh[0][0] += sign * tau
        S = [[Fr(0)] * 4 for _ in range(4)]
        for i in range(2):
            for j in range(2):
                S[i][j] = A[i][j]; S[2 + i][2 + j] = A[i][j]
                S[i][2 + j] = Bh[i][j]; S[2 + i][j] = Bh[j][i]
        return S

    ST = SigB(Fr(0), 1)
    print("Example 1: exact boundary derivatives dF/dgamma(0; tau)")
    for tau in (Fr(0), Fr(1, 10), Fr(3, 20), Fr(1, 5), Fr(3, 10)):
        _, d = boundary_derivative(ST, SigB(tau, +1), SigB(tau, -1))
        pred = Fr(-1, 700) + Fr(8, 189) * tau * tau
        assert d == pred, (tau, d, pred)
        print(f"  tau = {str(tau):5}:  {str(d):>12}  ==  -1/700 + (8/189) tau^2")
    print("  sign flip certified: negative at tau = 3/20, positive at tau = 1/5;"
          " threshold tau^2 = 27/800.")

    # Example 2 family: cancellation G'(0) = 0, both noise variants
    def SigA_entry(eps):
        return Sigma(eps)

    def SigA_rho(eps):
        v = [Fr(1), Fr(1), S_, S_]
        C = [[Fr(1), RW, RC + eps, RC + eps], [RW, Fr(1), RC + eps, RC + eps],
             [RC + eps, RC + eps, Fr(1), RW], [RC + eps, RC + eps, RW, Fr(1)]]
        return [[v[i] * v[j] * C[i][j] for j in range(4)] for i in range(4)]

    print("Example 2 family: structural cancellation at the HRP end")
    for name, Sg in (("(1,3)-entry noise", SigA_entry), ("rho_c all-cross noise", SigA_rho)):
        STa = Sg(Fr(0))
        _, v0p = boundary_derivative(STa, STa, STa)
        assert v0p == Fr(-216, 3125)
        for tau in (Fr(1, 10), Fr(3, 20), Fr(1, 5)):
            _, d = boundary_derivative(STa, Sg(tau), Sg(-tau))
            assert d == v0p, (name, tau)
        print(f"  {name}: dF/dgamma(0; tau) = V0'(0) = -216/3125 exactly "
              f"at tau in {{1/10, 3/20, 1/5}}")


def family_checks():
    """Exact checks for the solved exchangeable family (Section 5 of the
    paper): the scalar reduction, the exact objective, and the closed forms.
    All equalities are exact rational identities at the sampled points."""
    a, d, c = Fr(3, 4), Fr(3), Fr(3, 5)
    Sv = a + d
    K = Sv - 2 * c

    def Sig_rho(eps):
        v = [Fr(1), Fr(1), S_, S_]
        C = [[Fr(1), RW, RC + eps, RC + eps], [RW, Fr(1), RC + eps, RC + eps],
             [RC + eps, RC + eps, Fr(1), RW], [RC + eps, RC + eps, RW, Fr(1)]]
        return [[v[i] * v[j] * C[i][j] for j in range(4)] for i in range(4)]

    STf = Sig_rho(Fr(0))

    def xgz(g, z):
        return (d - g * z) / (Sv - 2 * g * z)

    def Q(x):
        return a * x * x + d * (1 - x) * (1 - x) + 2 * c * x * (1 - x)

    xs = xgz(Fr(1), c)

    def V0c(g):
        return Fr(9) * (44 * g * g - 200 * g + 275) / (Fr(5) * (8 * g - 25) ** 2)

    def Fc(g, e):
        tot = Fr(0)
        for sgn in (1, -1):
            z = c + sgn * e
            tot += ((g * z - c) / (Sv - 2 * g * z)) ** 2
        return Q(xs) + ((d - a) ** 2 / K) * tot / 2

    ok = True
    for g in (Fr(0), Fr(1, 4), Fr(1, 2), Fr(3, 4), Fr(9, 10), Fr(1)):
        for eps in (Fr(0), Fr(1, 10), Fr(-3, 20), Fr(1, 4)):
            w = weights(Sig_rho(eps), g)
            ok &= (w[0] == w[1]) and (w[2] == w[3])          # within = (1/2, 1/2)
            x = w[0] + w[1]
            ok &= x == xgz(g, c + 2 * eps)                    # x = x_gamma(z), e = 2 tau
            V = sum(w[i] * STf[i][j] * w[j] for i in range(4) for j in range(4))
            ok &= V == Q(x)                                   # realized variance
    assert ok
    print("family: reduction identities exact (within-weights 1/2, x = x_gamma(z), V = Q(x))")

    okF = all(Fc(g, Fr(0)) == V0c(g) for g in (Fr(0), Fr(1, 3), Fr(1, 2), Fr(4, 5), Fr(1)))
    assert okF
    okE = True
    for g in (Fr(0), Fr(2, 5), Fr(7, 10), Fr(1)):
        for e in (Fr(0), Fr(1, 5), Fr(2, 5), Fr(3, 5)):
            tau = e / 2
            wP = weights(Sig_rho(tau), g)
            wM = weights(Sig_rho(-tau), g)
            VP = sum(wP[i] * STf[i][j] * wP[j] for i in range(4) for j in range(4))
            VM = sum(wM[i] * STf[i][j] * wM[j] for i in range(4) for j in range(4))
            okE &= (VP + VM) / 2 == Fc(g, e)
    assert okE
    print("family: exact objective (20) and closed form V0 (21) match the recursion")


if __name__ == "__main__":
    main()
