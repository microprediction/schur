"""Exact-arithmetic certificate for "HERC Is a Corner of a Schur Square".

Every check runs in rational arithmetic (fractions.Fraction) or in truncated
power series with rational coefficients. No floating point enters a claim.
Run:  python3 verify_herc_square.py
"""
from fractions import Fraction as Fr
from itertools import product
import random

# ----------------------------------------------------------------------------
# linear algebra over any field-like type
def sub(M, I, J): return [[M[i][j] for j in J] for i in I]
def matmul(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
def matvec(A, x): return [sum(A[i][k]*x[k] for k in range(len(x))) for i in range(len(A))]
def transpose(A): return [list(r) for r in zip(*A)]
def inv(M):
    n = len(M); A = [list(r) + [Fr(int(i == j)) for j in range(n)] for i, r in enumerate(M)]
    for c in range(n):
        p = next(r for r in range(c, n) if A[r][c] != 0); A[c], A[p] = A[p], A[c]
        pv = A[c][c]; A[c] = [x/pv for x in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]; A[r] = [x - f*y for x, y in zip(A[r], A[c])]
    return [r[n:] for r in A]
def solve(M, x): return matvec(inv(M), x)
def quad(x, M, y=None):
    y = x if y is None else y
    return sum(x[i]*M[i][j]*y[j] for i in range(len(x)) for j in range(len(y)))
def normalize(w):
    s = sum(w); return [x/s for x in w]
def gmv(S): return normalize(solve(S, [Fr(1)]*len(S)))

# ----------------------------------------------------------------------------
# truncated multivariate power series (for exact derivatives)
class Jet:
    __slots__ = ('c', 'ords')
    def __init__(self, c, ords):
        self.ords = ords; self.c = {k: v for k, v in c.items() if v != 0}
    @classmethod
    def const(cls, x, ords): return cls({(0,)*len(ords): Fr(x)}, ords)
    @classmethod
    def var(cls, i, x0, ords):
        e = [0]*len(ords); e[i] = 1
        return cls({(0,)*len(ords): Fr(x0), tuple(e): Fr(1)}, ords)
    def _lift(self, o): return o if isinstance(o, Jet) else Jet.const(o, self.ords)
    def __add__(self, o):
        o = self._lift(o); c = dict(self.c)
        for k, v in o.c.items(): c[k] = c.get(k, 0) + v
        return Jet(c, self.ords)
    __radd__ = __add__
    def __neg__(self): return Jet({k: -v for k, v in self.c.items()}, self.ords)
    def __sub__(self, o): return self + (-self._lift(o))
    def __rsub__(self, o): return self._lift(o) + (-self)
    def __mul__(self, o):
        o = self._lift(o); c = {}
        for k1, v1 in self.c.items():
            for k2, v2 in o.c.items():
                k = tuple(a+b for a, b in zip(k1, k2))
                if all(a <= m for a, m in zip(k, self.ords)): c[k] = c.get(k, 0) + v1*v2
        return Jet(c, self.ords)
    __rmul__ = __mul__
    def inv(self):
        a0 = self.c.get((0,)*len(self.ords), Fr(0)); assert a0 != 0
        d = (self - a0)*(Fr(1)/a0); out = Jet.const(1, self.ords); term = Jet.const(1, self.ords)
        for _ in range(sum(self.ords)): term = term*(-d); out = out + term
        return out*(Fr(1)/a0)
    def __truediv__(self, o): return self*self._lift(o).inv()
    def __rtruediv__(self, o): return self._lift(o)*self.inv()
    def __ne__(self, o): return self.c.get((0,)*len(self.ords), Fr(0)) != 0 if o == 0 else NotImplemented
    def __eq__(self, o): return not self.__ne__(0) if o == 0 else NotImplemented
    __hash__ = None
    def coef(self, *k): return self.c.get(tuple(k), Fr(0))

# ----------------------------------------------------------------------------
# the square
def condition(Q, b, I, J, g):
    """Pair of I conditioned on J inside (Q, b), damped by g."""
    A, B, D = sub(Q, I, I), sub(Q, I, J), sub(Q, J, J)
    BD = matmul(B, inv(D))
    Qc = [[A[i][j] - g*sum(BD[i][k]*B[j][k] for k in range(len(J))) for j in range(len(I))] for i in range(len(I))]
    bc = [b[I[i]] - g*sum(BD[i][k]*b[J[k]] for k in range(len(J))) for i in range(len(I))]
    return Qc, bc

def diagonal_bridge(Q, b, eta):
    """Unnormalized leaf weights b_i^c / Q_i^c, asset i conditioned on its leaf-mates, damped by eta."""
    n = len(Q); u = []
    for i in range(n):
        J = [j for j in range(n) if j != i]
        if J:
            Qc, bc = condition(Q, b, [i], J, eta); u.append(bc[0]/Qc[0][0])
        else:
            u.append(b[0]/Q[0][0])
    return u

def cluster_pair(S, u, C, gamma):
    n = len(S); J = [j for j in range(n) if j not in C]
    return (sub(S, C, C), [u[i] for i in C]) if not J else condition(S, u, C, J, gamma)

def square(S, clusters, gamma, eta, u=None):
    """The two-dial family. Returns the fully invested portfolio."""
    n = len(S); u = u or [Fr(1)]*n; w = [Fr(0)]*n
    for C in clusters:
        Q, b = cluster_pair(S, u, C, gamma)
        z = diagonal_bridge(Q, b, eta)
        zQz = quad(z, Q)
        if zQz == 0:                                   # a vanishing direction contributes nothing
            continue
        scale = sum(bi*zi for bi, zi in zip(b, z))/zQz   # = 1/(b^T z * nu), with no division by b^T z
        for k, i in enumerate(C): w[i] = z[k]*scale
    return normalize(w)

def herc(S, clusters):
    """HERC with Raffinot's split read as inverse risk: budget ∝ 1/Var(IVP_C), IVP inside."""
    n = len(S); w = [Fr(0)]*n
    for C in clusters:
        A = sub(S, C, C); wC = normalize([1/A[k][k] for k in range(len(C))]); nu = quad(wC, A)
        for k, i in enumerate(C): w[i] = wC[k]/nu
    return normalize(w)

def leaves(t): return list(t) if isinstance(t, list) else leaves(t[0]) + leaves(t[1])
def clusters_under(t): return [t] if isinstance(t, list) else clusters_under(t[0]) + clusters_under(t[1])

def herc_topdown(S, tree, aggregate):
    """Top-down HERC on a dendrogram cut into final clusters. aggregate in {'precision', 'variance'}."""
    n = len(S); w = [Fr(0)]*n
    def var_ivp(C):
        A = sub(S, C, C); return quad(normalize([1/A[k][k] for k in range(len(C))]), A)
    def rec(t, budget):
        if isinstance(t, list):
            A = sub(S, t, t); wC = normalize([1/A[k][k] for k in range(len(t))])
            for k, i in enumerate(t): w[i] = budget*wC[k]
            return
        L, R = t
        if aggregate == 'precision':
            pL = sum(1/var_ivp(C) for C in clusters_under(L)); pR = sum(1/var_ivp(C) for C in clusters_under(R))
            aL = pL/(pL + pR)
        else:
            rL = sum(var_ivp(C) for C in clusters_under(L)); rR = sum(var_ivp(C) for C in clusters_under(R))
            aL = 1 - rL/(rL + rR)
        rec(L, budget*aL); rec(R, budget*(1 - aL))
    rec(tree, Fr(1)); return w

def tree_conditioned_pairs(S, u, tree, gamma):
    """Pairs for the final clusters obtained by conditioning top-down along the dendrogram."""
    out = {}
    def rec(Q, b, t, idx):
        if isinstance(t, list):
            out[tuple(t)] = (Q, b); return
        L, R = t; nL = len(leaves(L)); I = list(range(nL)); J = list(range(nL, len(Q)))
        QL, bL = condition(Q, b, I, J, gamma); QR, bR = condition(Q, b, J, I, gamma)
        rec(QL, bL, L, idx[:nL]); rec(QR, bR, R, idx[nL:])
    order = leaves(tree); Q = sub(S, order, order); b = [u[i] for i in order]
    rec(Q, b, tree, order); return out

def random_spd(n, rng, scale=4):
    G = [[Fr(rng.randint(-scale, scale)) for _ in range(n)] for _ in range(n)]
    S = matmul(G, transpose(G))
    for i in range(n): S[i][i] += n
    return S

def equicorrelated(sig, rho):
    n = len(sig); return [[sig[i]*sig[j]*(Fr(1) if i == j else rho) for j in range(n)] for i in range(n)]

checks = []
def check(name, cond):
    checks.append((name, bool(cond))); print(('ok  ' if cond else 'FAIL'), name)

rng = random.Random(2026)
clusters3 = [[0, 1, 2], [3, 4], [5, 6, 7]]; tree3 = ([0, 1, 2], ([3, 4], [5, 6, 7]))
clusters2 = [[0, 1, 2, 3], [4, 5, 6, 7]]; tree2 = ([0, 1, 2, 3], [4, 5, 6, 7])
S = random_spd(8, rng)

# 1-3 corners
check('1. (0,0) is HERC (inverse naive variance budgets, inverse variance inside)', square(S, clusters3, Fr(0), Fr(0)) == herc(S, clusters3))
check('2. (1,1) is the global minimum-variance portfolio', square(S, clusters3, Fr(1), Fr(1)) == gmv(S))
S22 = [[Fr(2), Fr(1)], [Fr(1), Fr(1)]]   # issue #26: b_{1}(1) = 0 on SPD input; the global direction is (0, 1)
check('2b. a vanishing companion at the far end (issue #26): Sigma = [[2,1],[1,1]], singleton clusters, gamma = 1 gives (0, 1)',
      square(S22, [[0], [1]], Fr(1), Fr(1)) == gmv(S22) == [Fr(0), Fr(1)]
      and square(S22, [[0], [1]], Fr(0), Fr(1)) == [Fr(1, 3), Fr(2, 3)] and square(S22, [[0], [1]], Fr(1, 2), Fr(1)) == [Fr(1, 4), Fr(3, 4)])
u = [Fr(rng.randint(1, 9), 4) for _ in range(8)]
check('3. (1,1) with companion u is Sigma^{-1} u', square(S, clusters3, Fr(1), Fr(1), u) == normalize(solve(S, u)))

# 4-5 Raffinot's split: precision reading telescopes; the library sum rule does not
tree3b = (([0, 1, 2], [3, 4]), [5, 6, 7])
check('4. top-down with additive precisions equals flat inverse fitness on two different trees',
      herc_topdown(S, tree3, 'precision') == herc(S, clusters3) == herc_topdown(S, tree3b, 'precision'))
check('5. library sum-of-variances rule agrees for two clusters and differs for three',
      herc_topdown(S, tree2, 'variance') == herc(S, clusters2) and herc_topdown(S, tree3, 'variance') != herc(S, clusters3))

# 6 tree-composed conditioning agrees with flat conditioning at gamma = 1
pairs = tree_conditioned_pairs(S, [Fr(1)]*8, tree3, Fr(1))
agree = all(pairs[tuple(C)] == cluster_pair(S, [Fr(1)]*8, C, Fr(1)) for C in clusters3)
check('6. conditioning composed down the dendrogram equals conditioning on everything at gamma = 1', agree)

# 7 diagonal bridge closed form and Stevens
Q = random_spd(5, rng); b = [Fr(rng.randint(1, 5)) for _ in range(5)]; Qi = inv(Q); Qb = matvec(Qi, b)
ok = True
for eta in (Fr(0), Fr(1, 3), Fr(1)):
    u_ = diagonal_bridge(Q, b, eta)
    closed = [((1 - eta)*b[i] + eta*Qb[i]/Qi[i][i])/((1 - eta)*Q[i][i] + eta/Qi[i][i]) for i in range(5)]
    ok = ok and u_ == closed
check('7. diagonal bridge: numerator and denominator are linear in eta between the naive and the Stevens quantities', ok)
# Stevens: (Q^{-1}1)_i = (1 - sum_j beta_ij) / (Q_ii (1 - R_i^2))
ones = [Fr(1)]*5; st = True
for i in range(5):
    J = [j for j in range(5) if j != i]
    beta = matmul(sub(Q, [i], J), inv(sub(Q, J, J)))[0]
    resid = Q[i][i] - sum(beta[k]*Q[J[k]][i] for k in range(4))
    st = st and matvec(Qi, ones)[i] == (1 - sum(beta))/resid
check('8. Stevens: minimum-variance weight = (1 - sum of hedge betas) / residual variance', st)

# 9 long-only frontier on the leaf dial
eta_plus = min((b[i]/(b[i] - Qb[i]/Qi[i][i]) for i in range(5) if Qb[i] < 0), default=None)
if eta_plus is not None:
    lo = all(x >= 0 for x in diagonal_bridge(Q, b, eta_plus)); hi = any(x < 0 for x in diagonal_bridge(Q, b, eta_plus + Fr(1, 100)))
    check('9. long-only frontier eta_+ = min b_i/(b_i - s_i): nonnegative at eta_+, a short just beyond', lo and hi)
else:
    check('9. long-only frontier (no short at eta = 1 in this draw; frontier is 1)', all(x >= 0 for x in Qb))

# 10 the cut: one cluster and n clusters both give the diagonal bridge
one = [list(range(8))]; single = [[i] for i in range(8)]
ok = True
for t in (Fr(0), Fr(2, 5), Fr(1)):
    wd = normalize(diagonal_bridge(S, [Fr(1)]*8, t))
    ok = ok and square(S, one, Fr(0), t) == wd and square(S, single, t, Fr(0)) == wd
check('10. k = 1 in eta and k = n in gamma both trace the diagonal bridge; HERC is inverse variance at both cuts', ok)

# 11-14 equicorrelated cluster: straight segment, theta map, eta_+, exact interior optimum
sig = [Fr(1), Fr(2), Fr(2)]; rho = Fr(1, 4); n3 = 3
E = equicorrelated(sig, rho); s = sum(1/x for x in sig)
kap = lambda r: r/(1 + (n3 - 2)*r); kappa = kap(rho)
a = [1/x**2 for x in sig]; c = [(s - 1/sig[i])/sig[i] for i in range(n3)]; alpha, beta = sum(a), sum(c)
wstar, wivp = gmv(E), normalize(a)
def w_of_m(m): return normalize([a[i] - m*c[i] for i in range(n3)])
def theta(m): return m*(alpha - kappa*beta)/(kappa*(alpha - m*beta))
ok = beta == s*s - alpha
for eta in (Fr(1, 5), Fr(1, 2), Fr(9, 10)):
    w = normalize(diagonal_bridge(E, [Fr(1)]*n3, eta)); m = eta*kappa
    ok = ok and w == w_of_m(m) == [wstar[i] + (1 - theta(m))*(wivp[i] - wstar[i]) for i in range(n3)]
check('11. equicorrelated leaf: w(eta) lies on the segment from inverse variance to cluster minimum variance, theta as stated', ok)
sig4 = [Fr(1), Fr(2), Fr(3), Fr(5)]; E4 = equicorrelated(sig4, Fr(1, 3)); s4 = sum(1/x for x in sig4); k4 = Fr(1, 3)/(1 + 2*Fr(1, 3))
eta_plus4 = min(1/(k4*sig4[i]*(s4 - 1/sig4[i])) for i in range(4))
check('12. equicorrelated eta_+ = min_i 1/(kappa sigma_i s_{-i}); equals 6/11 for vols (1,2,3,5) and rho = 1/3',
      eta_plus4 == Fr(6, 11) and all(x >= 0 for x in diagonal_bridge(E4, [Fr(1)]*4, eta_plus4))
      and any(x < 0 for x in diagonal_bridge(E4, [Fr(1)]*4, eta_plus4 + Fr(1, 50))))
Vs = quad(wstar, E); Delta = quad(wivp, E) - Vs
def F2(eta): return sum(quad(w_of_m(eta*kap(r)), E) for r in (Fr(0), 2*rho))/2
ok = all(F2(eta) == Vs + Delta/2*sum((1 - theta(eta*kap(r)))**2 for r in (Fr(0), 2*rho)) for eta in (Fr(0), Fr(1, 3), Fr(1)))
check('13. F(eta) = V* + Delta E[(1 - theta)^2] on the leaf dial', ok)
eta_star = kap(rho)/kap(2*rho)
check('14. exact interior optimum eta* = (1+2(n-2)rho)/(2(1+(n-2)rho)) = 3/5 for vols (1,2,2), rho = 1/4; F(0)-F(eta*) = 1/45, F(1)-F(eta*) = 1/20',
      eta_star == Fr(3, 5) and F2(Fr(0)) - F2(eta_star) == Fr(1, 45) and F2(Fr(1)) - F2(eta_star) == Fr(1, 20))

# 15 sign criterion at eta = 1 under symmetric noise rho +- tau, both signs
def leaf_end(sig, rho, tau=Fr(1, 1000)):
    n = len(sig); E = equicorrelated(sig, rho); s = sum(1/x for x in sig)
    kap = lambda r: r/(1 + (n - 2)*r); kappa = kap(rho); q = (n - 2)*rho
    a = [1/x**2 for x in sig]; c = [(s - 1/sig[i])/sig[i] for i in range(n)]; alpha, beta = sum(a), sum(c)
    wstar, wivp = gmv(E), normalize(a); Vs = quad(wstar, E); Delta = quad(wivp, E) - Vs
    phi = lambda x: x*(x - kappa)/(alpha - x*beta)**3
    dF = 2*Delta*alpha**2*(alpha - kappa*beta)/kappa**2*sum(phi(kap(rho + sg*tau)) for sg in (-1, 1))/2
    crit = alpha*(1 - q) + kappa*beta*(2 + q)
    # predicted second-order coefficient and shift
    G_eta = 2*Delta*alpha**2*(alpha - kappa*beta)/kappa**2*crit/((1 + q)**4*(alpha - kappa*beta)**4)
    V0pp = 2*Delta*alpha**2/(alpha - kappa*beta)**2
    shift = -crit/(rho**2*(1 + q)**2*(alpha - kappa*beta))
    return dF, crit, G_eta, V0pp, shift, Delta
ok = True
for sg_, rho_ in (([Fr(1), Fr(2), Fr(3), Fr(5)], Fr(1, 10)), ([Fr(1), Fr(20), Fr(30), Fr(50)], Fr(9, 10))):
    dF, crit, G_eta, V0pp, shift, Delta = leaf_end(sg_, rho_)
    ok = ok and (dF > 0) == (crit > 0) and -G_eta/V0pp == shift
    # the exact derivative is tau^2 G_eta + O(tau^4): compare at two tau
    d1 = leaf_end(sg_, rho_, Fr(1, 1000))[0]; d2 = leaf_end(sg_, rho_, Fr(1, 2000))[0]
    ok = ok and abs(d1/Fr(1, 1000)**2 - G_eta) < abs(G_eta)/10**4 and abs(d2/Fr(1, 2000)**2 - G_eta) < abs(G_eta)/10**4
dF_a, crit_a = leaf_end([Fr(1), Fr(2), Fr(3), Fr(5)], Fr(1, 10))[:2]
dF_b, crit_b = leaf_end([Fr(1), Fr(20), Fr(30), Fr(50)], Fr(9, 10))[:2]
check('15. sign at the full-coupling end: interior iff alpha(1-q) + kappa beta (2+q) > 0; both signs occur; second-order coefficient and shift as stated',
      ok and crit_a > 0 and crit_b < 0)

# 16-17 blindness of the HERC edge and the joint local theorem
def perturb_cross(S, C1, eps):
    T = [row[:] for row in S]
    for i in C1:
        for j in range(len(S)):
            if j not in C1: T[i][j] += eps; T[j][i] += eps
    return T
Sp = perturb_cross(S, clusters3[0], Fr(1, 10))
check('16. the HERC edge (gamma = 0) reads no cross-cluster covariance: unchanged under a cross-block perturbation',
      all(square(S, clusters3, Fr(0), e) == square(Sp, clusters3, Fr(0), e) for e in (Fr(0), Fr(1, 2), Fr(1))))

ords = (2, 2, 2)
def Jc(x): return Jet.const(x, ords)
def sigma_gateway(r1, r2, cc, s1, s2):
    return [[1, r1*s1, cc, r2*s2*cc], [r1*s1, s1*s1, r1*s1*cc, r1*s1*r2*s2*cc],
            [cc, r1*s1*cc, 1, r2*s2], [r2*s2*cc, r1*s1*r2*s2*cc, r2*s2, s2*s2]]
def joint(r1, r2, cc, s1, s2):
    g = Jet.var(0, 1, ords); e = Jet.var(1, 1, ords); t = Jet.var(2, 0, ords)
    Sigma = [[Jc(x) for x in row] for row in sigma_gateway(r1, r2, cc, s1, s2)]
    F = Jc(0)
    for sg in product((-1, 1), repeat=3):
        Sh = sigma_gateway(Jc(r1) + sg[0]*t, Jc(r2) + sg[1]*t, Jc(cc) + sg[2]*t, Jc(s1), Jc(s2))
        F = F + quad(square(Sh, [[0, 1], [2, 3]], g, e), Sigma)
    F = F*Fr(1, 8)
    H = [[2*F.coef(2, 0, 0), F.coef(1, 1, 0)], [F.coef(1, 1, 0), 2*F.coef(0, 2, 0)]]
    gG = [F.coef(1, 0, 2), F.coef(0, 1, 2)]
    return F, H, gG, [-x for x in solve(H, gG)]
F, H, gG, sh = joint(Fr(1, 4), Fr(1, 2), Fr(1, 2), Fr(2), Fr(3))
pd = H[0][0] > 0 and H[0][0]*H[1][1] - H[0][1]**2 > 0
check('17. joint example A: V0 stationary at (1,1), Hessian positive definite, both second-order shifts negative (both dials interior)',
      F.coef(1, 0, 0) == 0 and F.coef(0, 1, 0) == 0 and pd and sh[0] < 0 and sh[1] < 0)
F, H, gG, sh = joint(Fr(1, 4), Fr(1, 2), Fr(1, 4), Fr(1, 2), Fr(3))
pd = H[0][0] > 0 and H[0][0]*H[1][1] - H[0][1]**2 > 0
check('18. joint example B: gamma shift positive (full coupling across clusters), eta shift negative (interior inside)',
      F.coef(1, 0, 0) == 0 and F.coef(0, 1, 0) == 0 and pd and sh[0] > 0 and sh[1] < 0)
print('   example B shifts per tau^2: %.3f, %.3f; edge shift for eta at gamma = 1: %.3f' % (float(sh[0]), float(sh[1]), float(-F.coef(0, 1, 2)/(2*F.coef(0, 2, 0)))))
# finite-noise confirmation at tau = 1/40: the predicted point beats the corner
def F_exact(p, g, e, tau):
    r1, r2, cc, s1, s2 = p; Sigma = sigma_gateway(*p); tot = Fr(0)
    for sg in product((-1, 1), repeat=3):
        Sh = sigma_gateway(r1 + sg[0]*tau, r2 + sg[1]*tau, cc + sg[2]*tau, s1, s2)
        tot += quad(square(Sh, [[0, 1], [2, 3]], g, e), Sigma)
    return tot/8
tau = Fr(1, 40); okj = True
for p, (sg_, se_) in (((Fr(1, 4), Fr(1, 2), Fr(1, 2), Fr(2), Fr(3)), joint(Fr(1, 4), Fr(1, 2), Fr(1, 2), Fr(2), Fr(3))[3]),
                      ((Fr(1, 4), Fr(1, 2), Fr(1, 4), Fr(1, 2), Fr(3)), joint(Fr(1, 4), Fr(1, 2), Fr(1, 4), Fr(1, 2), Fr(3))[3])):
    gp, ep = min(1 + sg_*tau**2, Fr(1)), 1 + se_*tau**2
    okj = okj and F_exact(p, gp, ep, tau) < F_exact(p, Fr(1), Fr(1), tau)
check('19. at tau = 1/40 the predicted point beats the full-coupling corner in both joint examples', okj)

# 20-21 volatility companion: inverse-vol leaves, budgets DR^2 / sigma-bar; HERC-vol uses DR / sigma-bar
def spd_with_square_diagonal(rng, n=8):
    G = [[Fr(rng.randint(-3, 3)) for _ in range(n)] for _ in range(n)]
    M = matmul(G, transpose(G))
    for i in range(n):
        r = int(float(M[i][i])**0.5) + 1; M[i][i] = Fr(r*r)     # inflate the diagonal to the next square
    return M, [Fr(int(float(M[i][i])**0.5)) for i in range(n)]
Rr, sig8 = spd_with_square_diagonal(rng)
assert all(sig8[i]**2 == Rr[i][i] for i in range(8))
w_sq = square(Rr, clusters3, Fr(0), Fr(0), sig8)
def dr2_over_sbar(C):
    A = sub(Rr, C, C); wl = normalize([1/sig8[i] for i in C])
    sbar = sum(sig8[i]*wl[k] for k, i in enumerate(C)); return (sbar**2/quad(wl, A))/sbar
okv = all(sum(w_sq[i] for i in C)/sum(w_sq[i] for i in clusters3[0]) == dr2_over_sbar(C)/dr2_over_sbar(clusters3[0]) for C in clusters3)
okl = all(w_sq[i]/w_sq[C[0]] == sig8[C[0]]/sig8[i] for C in clusters3 for i in C)
# cross-cluster noise alone still moves the inner dial through the mixed term of G
def joint_noise(r1, r2, cc, s1, s2, which):
    g = Jet.var(0, 1, ords); e = Jet.var(1, 1, ords); t_ = Jet.var(2, 0, ords)
    Sigma = [[Jc(x) for x in row] for row in sigma_gateway(r1, r2, cc, s1, s2)]
    F_ = Jc(0)
    for sg in product((-1, 1), repeat=len(which)):
        d = dict(zip(which, sg))
        Sh = sigma_gateway(Jc(r1) + d.get('r1', 0)*t_, Jc(r2) + d.get('r2', 0)*t_, Jc(cc) + d.get('c', 0)*t_, Jc(s1), Jc(s2))
        F_ = F_ + quad(square(Sh, [[0, 1], [2, 3]], g, e), Sigma)
    return F_*Fr(1, 2**len(which))
Fc = joint_noise(Fr(1, 4), Fr(1, 2), Fr(1, 2), Fr(2), Fr(3), ('c',))
check('22. noise in the cross-cluster correlation alone gives a nonzero eta-component of grad G', Fc.coef(0, 1, 2) != 0)

check('20. companion u = sigma: (0,0) holds inverse-volatility leaves with budgets proportional to DR^2 / sigma-bar', okv and okl)
check('21. companion u = sigma: (1,1) is the most diversified portfolio Sigma^{-1} sigma', square(Rr, clusters3, Fr(1), Fr(1), sig8) == normalize(solve(Rr, sig8)))

bad = [n for n, c in checks if not c]
print(f'\n{len(checks) - len(bad)} of {len(checks)} checks passed' + (f'; FAILED: {bad}' if bad else ''))
raise SystemExit(1 if bad else 0)
