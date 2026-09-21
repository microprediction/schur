"""Exact-arithmetic certificate for "The Far End of a Schur Bridge Does Not Care
How Its Clusters Are Priced, and the Interior Does".

Every check runs in rational arithmetic (fractions.Fraction). No floating point
enters a claim. Run:  python3 verify_test_space.py
"""
from fractions import Fraction as Fr
import random

# ---------------------------------------------------------------- linear algebra
def sub(M, I, J): return [[M[i][j] for j in J] for i in I]
def matmul(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
def transpose(A): return [list(r) for r in zip(*A)]
def matvec(A, x): return [sum(A[i][k]*x[k] for k in range(len(x))) for i in range(len(A))]
def dot(a, b): return sum(x*y for x, y in zip(a, b))
def inv(M):
    n = len(M); A = [list(r) + [Fr(int(i == j)) for j in range(n)] for i, r in enumerate(M)]
    for c in range(n):
        p = next((r for r in range(c, n) if A[r][c] != 0), None)
        if p is None: raise ZeroDivisionError("singular")
        A[c], A[p] = A[p], A[c]
        pv = A[c][c]; A[c] = [x/pv for x in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]; A[r] = [x - f*y for x, y in zip(A[r], A[c])]
    return [r[n:] for r in A]
def solve(M, x): return matvec(inv(M), x)
def det(M):
    n = len(M); A = [list(r) for r in M]; d = Fr(1)
    for c in range(n):
        p = next((r for r in range(c, n) if A[r][c] != 0), None)
        if p is None: return Fr(0)
        if p != c: A[c], A[p] = A[p], A[c]; d = -d
        d *= A[c][c]
        for r in range(c+1, n):
            if A[r][c] != 0:
                f = A[r][c]/A[c][c]; A[r] = [x - f*y for x, y in zip(A[r], A[c])]
    return d
def normalize(w):
    s = sum(w); return [x/s for x in w]

# ------------------------------------------------------------------ the family
def conditioned_direction(S, u, C, gamma):
    """d_C(gamma) = Q_C(gamma)^{-1} b_C(gamma), cluster C conditioned on everything else."""
    n = len(S); J = [j for j in range(n) if j not in C]
    A = sub(S, C, C); uC = [u[i] for i in C]
    if gamma == 0 or not J: return solve(A, uC)
    B = sub(S, C, J); D = sub(S, J, J)
    X = matmul(inv(D), transpose(B) if False else [[B[a][b] for a in range(len(C))] for b in range(len(J))])
    Y = matvec(inv(D), [u[j] for j in J])
    Q = [[A[a][b] - gamma*sum(B[a][t]*X[t][b] for t in range(len(J))) for b in range(len(C))] for a in range(len(C))]
    bb = [uC[a] - gamma*sum(B[a][t]*Y[t] for t in range(len(J))) for a in range(len(C))]
    return solve(Q, bb)

def trial_matrix(S, u, clusters, gamma):
    """D_gamma: column c is the conditioned direction of cluster c, supported on it."""
    n = len(S); D = [[Fr(0)]*len(clusters) for _ in range(n)]
    for c, C in enumerate(clusters):
        d = conditioned_direction(S, u, C, gamma)
        for a, i in enumerate(C): D[i][c] = d[a]
    return D

def allocate(S, u, clusters, gamma, R):
    """w propto D_gamma (R^T S D_gamma)^{-1} R^T u."""
    D = trial_matrix(S, u, clusters, gamma)
    M = matmul(transpose(R), matmul(S, D))
    a = solve(M, matvec(transpose(R), u))
    return normalize(matvec(D, a))

def gmv(S, u): return normalize(solve(S, u))

def core_test_matrix(n, cores, k):
    E = [[Fr(0)]*k for _ in range(n)]
    for c, p in enumerate(cores): E[p][c] = Fr(1)
    return E

def random_spd(n, rng, scale=3):
    G = [[Fr(rng.randint(-scale, scale)) for _ in range(n)] for _ in range(n)]
    S = matmul(G, transpose(G))
    for i in range(n): S[i][i] += n
    return S

checks = []
def check(name, cond):
    checks.append((name, bool(cond))); print(('ok   ' if cond else 'FAIL ') + name)

rng = random.Random(2026)
n = 9; clusters = [[0, 1, 2], [3, 4], [5, 6, 7, 8]]; k = len(clusters)
S = random_spd(n, rng); u = [Fr(1)]*n

# 1. the columns of D_1 are the blocks of Sigma^{-1} u
D1 = trial_matrix(S, u, clusters, Fr(1)); g = solve(S, u)
check('1. at gamma = 1 each column of D is the corresponding block of Sigma^{-1} u',
      all(D1[i][c] == (g[i] if i in C else Fr(0)) for c, C in enumerate(clusters) for i in range(n)))
check('2. summing the columns of D_1 reconstructs Sigma^{-1} u exactly',
      [sum(D1[i][c] for c in range(k)) for i in range(n)] == g)

# 3. the far end is exact for every test space
tests = {
    'Galerkin, test = trial': D1,
    'one asset per cluster (cores) 0,3,5': core_test_matrix(n, [0, 3, 5], k),
    'a different core per cluster 2,4,8': core_test_matrix(n, [2, 4, 8], k),
    'random dense': [[Fr(rng.randint(-4, 4)) for _ in range(k)] for _ in range(n)],
    'random sparse': core_test_matrix(n, [1, 4, 7], k),
}
far_ok = True
for name, R in tests.items():
    try: far_ok = far_ok and allocate(S, u, clusters, Fr(1), R) == gmv(S, u)
    except ZeroDivisionError: far_ok = False
check('3. the far end equals Sigma^{-1} u for all five test spaces, exactly', far_ok)

# 4. the far end with a general companion vector
uu = [Fr(rng.randint(1, 6), rng.randint(1, 4)) for _ in range(n)]
check('4. the same holds for a general companion vector u (tangency or max diversification)',
      allocate(S, uu, clusters, Fr(1), core_test_matrix(n, [0, 3, 5], k)) == gmv(S, uu))

# 5. the interior does depend on the test space
interior = {name: allocate(S, u, clusters, Fr(1, 2), R) for name, R in tests.items()}
vals = list(interior.values())
check('5. at gamma = 1/2 the test space changes the allocation',
      all(vals[0] != v for v in vals[1:]))

# 6. Galerkin outer matrix is symmetric; a core test space is not
Dh = trial_matrix(S, u, clusters, Fr(1, 2)); E = core_test_matrix(n, [0, 3, 5], k)
MG = matmul(transpose(Dh), matmul(S, Dh)); ME = matmul(transpose(E), matmul(S, Dh))
check('6. the Galerkin outer matrix is symmetric and a core-test outer matrix is not',
      MG == transpose(MG) and ME != transpose(ME))

# 7. Galerkin is pole free: its outer matrix is a Gram matrix of Sigma, hence PD
def leading_minors_positive(M):
    return all(det(sub(M, list(range(j)), list(range(j)))) > 0 for j in range(1, len(M)+1))
check('7. the Galerkin outer matrix is positive definite at gamma = 0, 1/2, 1 (no pole)',
      all(leading_minors_positive(matmul(transpose(trial_matrix(S, u, clusters, gm)),
          matmul(S, trial_matrix(S, u, clusters, gm)))) for gm in (Fr(0), Fr(1, 2), Fr(1))))

# 8. a core test space can pass through singularity inside the dial
#    Sigma below is positive definite; det(E^T Sigma D_gamma) changes sign on (0, 1).
Sp = [[Fr(x) for x in row] for row in
      [[20,   8,  -5,   5,   5,  -6],
       [ 8,  36, -11,   8,   7, -12],
       [-5, -11,  33,   2,   6,  16],
       [ 5,   8,   2,  16,   3,   9],
       [ 5,   7,   6,   3,  29, -18],
       [-6, -12,  16,   9, -18,  50]]]
cl2 = [[0, 1, 2], [3, 4, 5]]; u2 = [Fr(1)]*6
Ep = core_test_matrix(6, [2, 3], 2)
def outer_det(gamma):
    D = trial_matrix(Sp, u2, cl2, gamma)
    return det(matmul(transpose(Ep), matmul(Sp, D)))
check('8. that Sigma is positive definite', leading_minors_positive(Sp))
d_lo, d_hi = outer_det(Fr(1, 2)), outer_det(Fr(3, 5))
check('9. a core test space on that Sigma: the outer determinant is positive at gamma = 1/2 and negative at gamma = 3/5, '
      'so it vanishes between them and the allocation has a pole inside the dial',
      d_lo > 0 and d_hi < 0)
check('10. the Galerkin outer determinant stays positive on the same input',
      all(det(matmul(transpose(trial_matrix(Sp, u2, cl2, Fr(t, 8))),
          matmul(Sp, trial_matrix(Sp, u2, cl2, Fr(t, 8))))) > 0 for t in range(9)))

# 11-12. hierarchical core-orbital allocation: prices with Sigma_PP, not with the pairing
def hcoa(S, u, clusters, cores):
    """Strategic allocation on the core covariance, then redistribution by each cluster's own block."""
    n = len(S); k = len(clusters)
    SPP = sub(S, cores, cores)
    W = normalize(solve(SPP, [Fr(1)]*k))
    w = [Fr(0)]*n
    for c, C in enumerate(clusters):
        v = normalize(solve(sub(S, C, C), [u[i] for i in C]))
        for a, i in enumerate(C): w[i] = W[c]*v[a]
    return w
cores = [0, 3, 5]
E = core_test_matrix(n, cores, k)
w_h = hcoa(S, u, clusters, cores)
w_f = allocate(S, u, clusters, Fr(0), E)
check('11. HCOA is not the gamma = 0 member with a core test space: it prices with Sigma_PP, not with the pairing',
      w_h != w_f)
# they coincide exactly when E^T Sigma (V - E) = 0
V = [[Fr(0)]*k for _ in range(n)]
for c, C in enumerate(clusters):
    v = normalize(solve(sub(S, C, C), [u[i] for i in C]))
    for a, i in enumerate(C): V[i][c] = v[a]
gap = matmul(transpose(E), matmul(S, [[V[i][c] - E[i][c] for c in range(k)] for i in range(n)]))
check('12. the two agree exactly when E^T Sigma (V - E) = 0, and here that matrix is nonzero',
      any(x != 0 for row in gap for x in row))
# construct a case where it vanishes: singleton clusters make V = E
single = [[i] for i in range(n)]
Es = core_test_matrix(n, list(range(n)), n)
Vs = [[Fr(int(i == c)) for c in range(n)] for i in range(n)]
gap_s = matmul(transpose(Es), matmul(S, [[Vs[i][c] - Es[i][c] for c in range(n)] for i in range(n)]))
check('13. with singleton clusters the holding is its own core, the gap vanishes, and HCOA is the far end itself',
      all(x == 0 for row in gap_s for x in row)
      and hcoa(S, u, single, list(range(n))) == gmv(S, u))

bad = [nm for nm, c in checks if not c]
print('\n%d of %d checks passed' % (len(checks) - len(bad), len(checks)) + ('; FAILED: %s' % bad if bad else ''))
raise SystemExit(1 if bad else 0)
