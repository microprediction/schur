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
    print()
    surrogate_expansion_check()
    print()
    singularity_certificates()
    print()
    whole_matrix_checks()
    print()
    lw_checks()
    print()
    endpoint_mismatch_certificate()

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
    print("        (grids avoid the singular point gamma = a/(c+e); see below)")


def surrogate_expansion_check():
    """Corollary 1: the surrogate optimizer matches the exact optimizer's
    small-noise expansion through order e^2. With V0, G the closed forms (21),
    t = -V0'/G' has slope V0''(1)/G'(1) at gamma = 1, so the surrogate solves
    t(gamma) = e^2 with 1 - gamma = (G'(1)/V0''(1)) e^2 + O(e^4); the exact
    coefficient is (S+4c)/(c^2 K) (Theorem 4(iv)). Both equal 1025/153.
    Also pins t(1/2) = 1323/2875, the corollary's mismatch point at e = c."""
    den = [Fr(-25), Fr(8)]                                       # 8g - 25
    V0n = [x * Fr(9, 5) for x in (Fr(275), Fr(-200), Fr(44))]
    V0d = pmul(den, den)
    Gn = pmul([Fr(0), Fr(0), Fr(1500)], [Fr(1), Fr(16)])
    Gd = pmul(pmul(den, den), pmul(den, den))

    def pder(p):
        return [p[i] * i for i in range(1, len(p))]

    def rat_der(n, dd):
        return psub(pmul(pder(n), dd), pmul(n, pder(dd))), pmul(dd, dd)

    V0p_n, V0p_d = rat_der(V0n, V0d)
    V0pp_n, V0pp_d = rat_der(V0p_n, V0p_d)
    Gp_n, Gp_d = rat_der(Gn, Gd)
    V0pp1 = peval(V0pp_n, Fr(1)) / peval(V0pp_d, Fr(1))
    Gp1 = peval(Gp_n, Fr(1)) / peval(Gp_d, Fr(1))
    assert V0pp1 > 0 and Gp1 > 0
    assert Gp1 / V0pp1 == Fr(1025, 153)
    a, d, c = Fr(3, 4), Fr(3), Fr(3, 5)
    Sv = a + d
    assert (Sv + 4 * c) / (c * c * (Sv - 2 * c)) == Fr(1025, 153)
    half = Fr(1, 2)
    t_half = (-(peval(V0p_n, half) / peval(V0p_d, half))
              / (peval(Gp_n, half) / peval(Gp_d, half)))
    assert t_half == Fr(1323, 2875)
    print("surrogate expansion: G'(1)/V0''(1) = (S+4c)/(c^2 K) = 1025/153 exactly,")
    print("          so surrogate and exact optimizers share the e^2 coefficient")
    print("          (Corollary 1); t(1/2) = 1323/2875")


def singularity_certificates():
    """Exact checks for Lemma 3 / Theorem 4(v) / Remark on shorting:
    (1) the raw recursion is undefined exactly at gamma = a/(c+e) on the
        upper noise branch (division by an exact zero), while x_gamma(z)
        extends it continuously from both sides;
    (2) at e = c the upper branch at gamma = 1 shorts: weights are exactly
        (2/3, 2/3, -1/6, -1/6), with the estimated covariance still PD;
    (3) the first-order condition at the singular point is positive for all
        e in [3/20, 3/5]: clearing positive denominators gives P(e) with
        nonnegative coefficients and positive constant term (also certified
        by Bernstein expansion), hence gamma*(e) < a/(c+e)."""
    a, d, c = Fr(3, 4), Fr(3), Fr(3, 5)
    Sv, K = a + d, a + d - 2 * c

    def Sig_rho(eps):
        v = [Fr(1), Fr(1), S_, S_]
        C = [[Fr(1), RW, RC + eps, RC + eps], [RW, Fr(1), RC + eps, RC + eps],
             [RC + eps, RC + eps, Fr(1), RW], [RC + eps, RC + eps, RW, Fr(1)]]
        return [[v[i] * v[j] * C[i][j] for j in range(4)] for i in range(4)]

    def xgz(g, z):
        return (d - g * z) / (Sv - 2 * g * z)

    # (1) singularity at gamma = a/z, upper branch e = c (tau = 3/10, z = 6/5)
    tau, z = Fr(3, 10), Fr(6, 5)
    g_sing = a / z                               # = 5/8
    try:
        weights(Sig_rho(tau), g_sing)
        raise AssertionError("recursion unexpectedly defined at the singular point")
    except ZeroDivisionError:
        pass
    for h in (Fr(1, 100), Fr(-1, 100), Fr(1, 10**6), Fr(-1, 10**6)):
        w = weights(Sig_rho(tau), g_sing + h)
        assert w[0] + w[1] == xgz(g_sing + h, z)  # recursion == extension off the point
    assert xgz(g_sing, z) == 1                    # extension gives the D block zero
    print("singularity: recursion divides by an exact zero at gamma = 5/8, e = 3/5;")
    print("             x_gamma(z) matches the recursion on both sides and equals 1 there")

    # (2) shorting at gamma = 1, upper branch e = c
    w = weights(Sig_rho(tau), Fr(1))
    assert w == [Fr(2, 3), Fr(2, 3), Fr(-1, 6), Fr(-1, 6)], w
    assert xgz(Fr(1), z) == Fr(4, 3)
    Sg = Sig_rho(tau)                             # estimated covariance still PD
    m1 = Sg[0][0]
    m2 = Sg[0][0] * Sg[1][1] - Sg[0][1] ** 2
    det3 = (Sg[0][0] * (Sg[1][1] * Sg[2][2] - Sg[1][2] ** 2)
            - Sg[0][1] * (Sg[0][1] * Sg[2][2] - Sg[1][2] * Sg[0][2])
            + Sg[0][2] * (Sg[0][1] * Sg[1][2] - Sg[1][1] * Sg[0][2]))
    # 4x4 determinant by cofactor along the last row (exact)
    def det4(M):
        def det3x3(N):
            return (N[0][0] * (N[1][1] * N[2][2] - N[1][2] * N[2][1])
                    - N[0][1] * (N[1][0] * N[2][2] - N[1][2] * N[2][0])
                    + N[0][2] * (N[1][0] * N[2][1] - N[1][1] * N[2][0]))
        tot = Fr(0)
        for j in range(4):
            minor = [[M[i][k] for k in range(4) if k != j] for i in range(1, 4)]
            tot += (-1) ** j * M[0][j] * det3x3(minor)
        return tot
    assert m1 > 0 and m2 > 0 and det3 > 0 and det4(Sg) > 0
    print("shorting: at gamma = 1, e = 3/5 (upper branch) the exact weights are")
    print("          (2/3, 2/3, -1/6, -1/6); the estimated covariance is PD")

    # (3) P(e) = z+(a-c)(S z+ - 2a z-)^3 + z-(a z- - c z+) z+^2 (S-2a)^3 > 0
    zp, zm = [c, Fr(1)], [c, Fr(-1)]              # polynomials in e
    lin = padd([x * Sv for x in zp], [x * (-2 * a) for x in zm])   # S z+ - 2a z-
    t1 = pmul([x * (a - c) for x in zp], pmul(pmul(lin, lin), lin))
    inner = padd([x * a for x in zm], [x * (-c) for x in zp])      # a z- - c z+
    t2 = [x * (Sv - 2 * a) ** 3 for x in pmul(pmul(zm, inner), pmul(zp, zp))]
    P = padd(t1, t2)
    assert P == [Fr(177147, 400000), Fr(0), Fr(6561, 800), Fr(1215, 32), Fr(23733, 640)], P
    assert all(cf >= 0 for cf in P) and P[0] > 0   # positivity by inspection
    assert bernstein_positive(P, Fr(3, 20), Fr(3, 5))
    # sign agreement with the first-order condition, at exact rational e
    def foc_at_sing(e):
        def rp(u):
            return 2 * K * (u - c) / (Sv - 2 * u) ** 3
        g = a / (c + e)
        return ((c + e) * rp(a) + (c - e) * rp(g * (c - e))) / 2
    for k in range(1, 46):
        e = Fr(3, 20) + Fr(k, 100)
        if e > Fr(3, 5):
            break
        assert foc_at_sing(e) > 0
    print("FOC at the singular point: P(e) = 177147/400000 + (6561/800)e^2 +")
    print("          (1215/32)e^3 + (23733/640)e^4, all coefficients nonnegative;")
    print("          Bernstein-certified positive on [3/20, 3/5]; direct FOC positive")
    print("          at 45 exact points; hence gamma*(e) < a/(c+e)  (Theorem 4(v))")

    # (4) Lemma 3 denominator identities: augmented diagonals and fitness sum
    #     (A_g)_ii = nu_A + 1/4,  (D_g)_ii = nu_D + 1,  and the sum formula;
    #     grid avoids gamma z = a.
    for g in (Fr(1, 3), Fr(3, 5), Fr(9, 10), Fr(1)):
        for eps in (Fr(0), Fr(3, 20), Fr(3, 10)):
            zz = c + 2 * eps
            Sg4 = Sig_rho(eps)
            A2 = [row[:2] for row in Sg4[:2]]
            D2 = [row[2:] for row in Sg4[2:]]
            B2 = [row[2:] for row in Sg4[:2]]
            Aa = aug(A2, B2, D2, g) if g != 0 else A2
            Da = aug(D2, mtrans(B2), A2, g) if g != 0 else D2
            nuA = (a * d - g * zz * zz) / (d - g * zz)
            nuD = (a * d - g * zz * zz) / (a - g * zz)
            assert Aa[0][0] == Aa[1][1] == nuA + Fr(1, 4)
            assert Da[0][0] == Da[1][1] == nuD + 1
            assert nuD + 1 == (a * (d + 1) - g * zz * (zz + 1)) / (a - g * zz)
            assert naive_var(Aa) == nuA and naive_var(Da) == nuD
            assert (nuA + nuD ==
                    (a * d - g * zz * zz) * (Sv - 2 * g * zz)
                    / ((d - g * zz) * (a - g * zz)))
    print("Lemma 3 denominators: (A_g)_ii = nu_A + 1/4, (D_g)_ii = nu_D + 1 =")
    print("          (a(d+1) - gz(z+1))/(a - gz), and the fitness-sum formula,")
    print("          all exact on a 12-point grid; only hole is gamma z = a")

    # (5) two-singularity counterexample from the third referee report:
    #     a=1, d=6, c=11/10 meets a<d, 4c^2<ad, S>6c, yet BOTH branches are
    #     singular at e=1/20 (so restricting Lemma 3's location claim to the
    #     numerical family, where c < a, is necessary).
    ax, dx, cx, ex = Fr(1), Fr(6), Fr(11, 10), Fr(1, 20)
    assert ax < dx and 4 * cx * cx < ax * dx and ax + dx > 6 * cx
    assert not (cx < ax)
    zlo, zhi = cx - ex, cx + ex
    for zc, gs in ((zlo, Fr(20, 21)), (zhi, Fr(20, 23))):
        assert gs * zc == ax and Fr(0) < gs < Fr(1)
    print("counterexample: a=1, d=6, c=11/10, e=1/20 satisfies the general")
    print("          hypotheses yet is singular on BOTH branches (gamma = 20/21")
    print("          and 20/23); c < a is what confines the family to one")


def whole_matrix_checks():
    """Exact checks for the whole-matrix noise section:
    (1) rank-one exchangeable noise (every entry perturbed): the recursion
        reduces exactly to the scalar bridge with (ahat, dhat, zhat), and the
        exact objective identity of Lemma (whole-matrix) holds;
    (2) interiority certificate: with independent equal-amplitude signs on
        all three channels and tau in {1/50, 1/100}, F(ghat) < min(F(0), F(1))
        exactly at ghat = 1 - (13081/1377) tau^2, the predicted optimum;
    (3) sharpness (Xi < 0): with noise confined to the dhat channel, F(1) is
        strictly below F(gamma) at every grid point gamma = k/50 < 1;
    (4) entrywise noise (exchangeability broken): exact enumeration over all
        sign states certifies strictly interior minimizers."""
    import itertools
    a, d, c = Fr(3, 4), Fr(3), Fr(3, 5)
    Sv, K = a + d, a + d - 2 * c

    def Sig_rankone(al, de, ze):
        A = [[Fr(1) + al, Fr(1, 2) + al], [Fr(1, 2) + al, Fr(1) + al]]
        D = [[Fr(4) + de, Fr(2) + de], [Fr(2) + de, Fr(4) + de]]
        z = c + ze
        Sg = [[Fr(0)] * 4 for _ in range(4)]
        for i in range(2):
            for j in range(2):
                Sg[i][j] = A[i][j]
                Sg[2 + i][2 + j] = D[i][j]
                Sg[i][2 + j] = z
                Sg[2 + i][j] = z
        return Sg

    ST0 = Sig_rankone(Fr(0), Fr(0), Fr(0))

    def Q(x):
        return a * x * x + d * (1 - x) * (1 - x) + 2 * c * x * (1 - x)

    xstar = (d - c) / K

    def Vr(g, al, de, ze):
        w = weights(Sig_rankone(al, de, ze), g)
        return sum(w[i] * ST0[i][j] * w[j] for i in range(4) for j in range(4))

    # (1) reduction and exact objective identity
    ok = True
    for g in (Fr(0), Fr(1, 3), Fr(7, 10), Fr(1)):
        for (al, de, ze) in ((Fr(1, 10), Fr(-1, 10), Fr(1, 10)),
                             (Fr(-1, 20), Fr(1, 7), Fr(-1, 9))):
            ah, dh, zh = a + al, d + de, c + ze
            x = (dh - g * zh) / (ah + dh - 2 * g * zh)
            w = weights(Sig_rankone(al, de, ze), g)
            ok &= (w[0] == w[1]) and (w[2] == w[3]) and (w[0] + w[1] == x)
            V = Vr(g, al, de, ze)
            ok &= V == Q(x)
            h = (a - c) * de - (d - c) * al
            ok &= (Q(x) - Q(xstar)
                   == ((d - a) * (g * zh - c) + h) ** 2
                   / (K * (ah + dh - 2 * g * zh) ** 2))
    assert ok
    print("whole-matrix (rank-one): recursion == scalar bridge with "
          "(ahat, dhat, zhat);\n          exact objective identity holds "
          "(every entry of Sigma-hat noisy)")

    # (2) interiority certificate at the predicted optimum
    def F8(g, tau):
        tot = Fr(0)
        for sa, sd, sz in itertools.product((1, -1), repeat=3):
            tot += Vr(g, sa * tau, sd * tau, sz * tau)
        return tot / 8

    for tau in (Fr(1, 50), Fr(1, 100)):
        ghat = 1 - Fr(13081, 1377) * tau * tau
        assert Fr(0) < ghat < Fr(1)
        Fh, F0, F1 = F8(ghat, tau), F8(Fr(0), tau), F8(Fr(1), tau)
        assert Fh < F0 and Fh < F1
        print(f"whole-matrix interior: tau = {tau}, ghat = 1 - (13081/1377)tau^2"
              f" = {ghat}:\n          F(ghat) < min(F(0), F(1)) exactly")

    # (3) sharpness: pure dhat-channel noise, full coupling optimal on the grid
    def F2d(g, ed):
        return (Vr(g, Fr(0), ed, Fr(0)) + Vr(g, Fr(0), -ed, Fr(0))) / 2

    for ed in (Fr(1, 10), Fr(1, 4)):
        F1v = F2d(Fr(1), ed)
        assert all(F2d(Fr(k, 50), ed) > F1v for k in range(50))
        print(f"whole-matrix sharpness: dhat-only noise, amplitude {ed}: "
              f"F(1) < F(k/50) for all k < 50\n          (Xi = -(189/500) "
              f"E[delta^2] < 0; support does not decide, the sign of Xi does)")

    # (4) entrywise noise: exact enumeration certificates
    VOL = [Fr(1), Fr(1), Fr(2), Fr(2)]
    RHOM = [[Fr(1), RW, RC, RC], [RW, Fr(1), RC, RC],
            [RC, RC, Fr(1), RW], [RC, RC, RW, Fr(1)]]
    OFFD = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    DIAG = [(0, 0), (1, 1), (2, 2), (3, 3)]

    def Sig_entry(tau, entries, signs):
        Sg = [[VOL[i] * VOL[j] * RHOM[i][j] for j in range(4)] for i in range(4)]
        for (i, j), s in zip(entries, signs):
            if i == j:
                Sg[i][i] = VOL[i] * VOL[i] * (1 + s * tau)
            else:
                Sg[i][j] = VOL[i] * VOL[j] * (RHOM[i][j] + s * tau)
                Sg[j][i] = Sg[i][j]
        return Sg

    def Fent(g, tau, entries):
        tot, n_sing = Fr(0), 0
        for s in itertools.product((1, -1), repeat=len(entries)):
            try:
                w = weights(Sig_entry(tau, entries, s), g)
            except ZeroDivisionError:
                n_sing += 1
                continue
            tot += sum(w[i] * ST[i][j] * w[j] for i in range(4) for j in range(4))
        return tot / (2 ** len(entries) - n_sing), n_sing

    for label, entries, tau, ghat in (
            ("six off-diagonal entries", OFFD, Fr(1, 5), Fr(2, 5)),
            ("all ten entries", OFFD + DIAG, Fr(1, 10), Fr(11, 12))):
        (F0, s0), (Fh, sh), (F1, s1) = (Fent(Fr(0), tau, entries),
                                        Fent(ghat, tau, entries),
                                        Fent(Fr(1), tau, entries))
        assert s0 == sh == s1 == 0
        assert F0 - Fh > 0 and F1 - Fh > 0
        print(f"entrywise ({label}, tau = {tau}): every sign state defined;\n"
              f"          F(0) - F({ghat}) = {float(F0 - Fh):.4e} and "
              f"F(1) - F({ghat}) = {float(F1 - Fh):.4e},\n"
              f"          both exactly positive: every minimizer is interior")

    tau, ghat = Fr(3, 20), Fr(12, 25)
    (F0, s0), (Fh, sh), (F1, s1) = (Fent(Fr(0), tau, OFFD + DIAG),
                                    Fent(ghat, tau, OFFD + DIAG),
                                    Fent(Fr(1), tau, OFFD + DIAG))
    assert s0 == sh == 0 and s1 == 56
    assert F0 - Fh > 0 and F1 > 1
    print(f"entrywise (all ten, tau = 3/20): F(0) - F(12/25) = "
          f"{float(F0 - Fh):.4e} > 0 exactly;\n          at gamma = 1, "
          f"{s1}/1024 states are singular and the defined mean is "
          f"{float(F1):.1f}:\n          the minimum-variance end degrades by "
          f"orders of magnitude")


def lw_checks():
    """Exact checks for the Ledoit-Wolf remark:
    (1) shrinkage bias at the minimum-variance end: at zero noise and
        shrinkage intensity rho, F is strictly decreasing into gamma = 1
        (the flatness of Lemma 2 is destroyed, slope ~ -(243/(32 K^3)) rho);
    (2) LW-consistent shrinkage below threshold: with rho = kappa_LW tau^2,
        kappa_LW = 1600/1019, the minimizer is still interior;
    (3) over-shrinkage: at fixed rho = 1/20 and tau = 1/50, full coupling on
        the shrunk input beats every grid point (the endpoint wins)."""
    def Sig_noise(tau):
        v = [Fr(1), Fr(1), S_, S_]
        C = [[Fr(1), RW, RC + tau, RC + tau], [RW, Fr(1), RC + tau, RC + tau],
             [RC + tau, RC + tau, Fr(1), RW], [RC + tau, RC + tau, RW, Fr(1)]]
        return [[v[i] * v[j] * C[i][j] for j in range(4)] for i in range(4)]

    STl = Sig_noise(Fr(0))

    def shrink(Sg, rho):
        mu = sum(Sg[i][i] for i in range(4)) / 4
        out = [[(1 - rho) * Sg[i][j] for j in range(4)] for i in range(4)]
        for i in range(4):
            out[i][i] += rho * mu
        return out

    def Flw(g, tau, rho):
        tot = Fr(0)
        for s in (1, -1):
            w = weights(shrink(Sig_noise(s * tau), rho), g)
            tot += sum(w[i] * STl[i][j] * w[j] for i in range(4) for j in range(4))
        return tot / 2

    # (1) bias slope: F(1) < F(1 - h) at zero noise for shrunk input
    rho = Fr(1, 100)
    for h in (Fr(1, 100), Fr(1, 1000)):
        assert Flw(Fr(1), Fr(0), rho) < Flw(1 - h, Fr(0), rho)
    print("LW bias: at zero noise, shrinkage makes F strictly decreasing into "
          "gamma = 1\n          (flatness of Lemma 2 destroyed; slope "
          "-(243/(32 K^3)) rho)")

    # (2) LW-consistent shrinkage: interior survives
    tau = Fr(1, 25)
    rho = Fr(1600, 1019) * tau * tau
    ghat = Fr(24, 25)
    Fh = Flw(ghat, tau, rho)
    assert Fh < Flw(Fr(0), tau, rho) and Fh < Flw(Fr(1), tau, rho)
    print(f"LW survives: tau = 1/25, rho = (1600/1019) tau^2: "
          f"F(24/25) < min(F(0), F(1)) exactly\n          (kappa_LW = 1600/1019 "
          f"< kappa* = 656/51: optimal shrinkage is an order of\n          "
          f"magnitude below the threshold at which the endpoint takes over)")

    # (3) over-shrinkage: fixed rho, endpoint wins on the grid
    tau, rho = Fr(1, 50), Fr(1, 20)
    F1 = Flw(Fr(1), tau, rho)
    assert all(Flw(Fr(k, 50), tau, rho) > F1 for k in range(15, 50))
    print("LW over-shrinkage: fixed rho = 1/20, tau = 1/50: F(1) < F(k/50) for "
          "k = 15..49\n          (bias slope dominates; full coupling on the "
          "shrunk input is optimal)")


def endpoint_mismatch_certificate():
    """Exact witness that the implemented recursion at gamma = 1 is not the
    sample minimum-variance portfolio on non-exchangeable input: an integer
    SPD matrix M with M (1,1,1,0)' = 2*ones (so w_MV = (1/3,1/3,1/3,0)
    exactly) on which the recursion returns (1330,1197,952,204)/3683."""
    M = [[Fr(8), Fr(-7), Fr(1), Fr(-6)],
         [Fr(-7), Fr(10), Fr(-1), Fr(7)],
         [Fr(1), Fr(-1), Fr(2), Fr(1)],
         [Fr(-6), Fr(7), Fr(1), Fr(11)]]

    def det_n(X):
        if len(X) == 1:
            return X[0][0]
        tot = Fr(0)
        for j in range(len(X)):
            minor = [[X[i][k] for k in range(len(X)) if k != j]
                     for i in range(1, len(X))]
            tot += (-1) ** j * X[0][j] * det_n(minor)
        return tot

    minors = [det_n([r[:k] for r in M[:k]]) for k in (1, 2, 3, 4)]
    assert minors == [Fr(8), Fr(31), Fr(58), Fr(230)]
    v = [Fr(1), Fr(1), Fr(1), Fr(0)]
    Mv = [sum(M[i][j] * v[j] for j in range(4)) for i in range(4)]
    assert Mv == [Fr(2)] * 4                      # M^{-1} 1 prop. (1,1,1,0)
    w_impl = weights(M, Fr(1))
    assert w_impl == [Fr(1330, 3683), Fr(1197, 3683),
                      Fr(952, 3683), Fr(204, 3683)]
    w_mv = [Fr(1, 3), Fr(1, 3), Fr(1, 3), Fr(0)]
    assert all(w_impl[i] != w_mv[i] for i in range(4))
    print("endpoint mismatch: integer SPD witness (minors 8, 31, 58, 230) with")
    print("          w_MV = (1/3, 1/3, 1/3, 0) exactly, while the implemented")
    print("          recursion at gamma = 1 returns (1330,1197,952,204)/3683;")
    print("          the two disagree in every entry, exactly")


if __name__ == "__main__":
    main()
