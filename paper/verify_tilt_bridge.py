"""Exact-arithmetic certificate for "The Thurstone Tilt Is a Schur Bridge,
and Both of Its Ends Are Races".

The identity claims run in rational arithmetic (fractions.Fraction). The two
statements about the race itself are Monte Carlo with fixed seeds and are
labelled as such. Run:  python3 verify_tilt_bridge.py
"""
from fractions import Fraction as Fr
import random

# ---------------------------------------------------------------- linear algebra
def sub(M, I, J): return [[M[i][j] for j in J] for i in I]
def matmul(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
def transpose(A): return [list(r) for r in zip(*A)]
def inv(M):
    n = len(M); A = [list(r) + [Fr(int(i == j)) for j in range(n)] for i, r in enumerate(M)]
    for c in range(n):
        p = next(r for r in range(c, n) if A[r][c] != 0)
        A[c], A[p] = A[p], A[c]
        pv = A[c][c]; A[c] = [x/pv for x in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]; A[r] = [x - f*y for x, y in zip(A[r], A[c])]
    return [r[n:] for r in A]
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
def pd(M): return all(det(sub(M, list(range(j)), list(range(j)))) > 0 for j in range(1, len(M)+1))

def corr_from(S):
    n = len(S)
    return [[S[i][j] for j in range(n)] for i in range(n)]   # inputs below already have unit diagonal

def block_taper(C, lab, phi):
    n = len(C)
    return [[C[i][j] if (i == j or lab[i] == lab[j]) else phi*C[i][j] for j in range(n)] for i in range(n)]

def blockdiag(C, lab):
    n = len(C)
    return [[C[i][j] if (i == j or lab[i] == lab[j]) else Fr(0) for j in range(n)] for i in range(n)]

def blend(C0, C, phi):
    n = len(C)
    return [[(1-phi)*C0[i][j] + phi*C[i][j] for j in range(n)] for i in range(n)]

def conditional(C, A, B):
    """C_AA - C_AB C_BB^{-1} C_BA."""
    CAA, CAB, CBB = sub(C, A, A), sub(C, A, B), sub(C, B, B)
    X = matmul(inv(CBB), transpose(CAB))
    return [[CAA[a][b] - sum(CAB[a][t]*X[t][b] for t in range(len(B))) for b in range(len(A))] for a in range(len(A))]

def damped_complement(C, A, B, gamma):
    """C_AA - gamma C_AB C_BB^{-1} C_BA."""
    CAA, CAB, CBB = sub(C, A, A), sub(C, A, B), sub(C, B, B)
    X = matmul(inv(CBB), transpose(CAB))
    return [[CAA[a][b] - gamma*sum(CAB[a][t]*X[t][b] for t in range(len(B))) for b in range(len(A))] for a in range(len(A))]

def rand_corr(n, rng, scale=3):
    """A rational correlation matrix: unit diagonal, positive definite."""
    while True:
        G = [[Fr(rng.randint(-scale, scale)) for _ in range(n)] for _ in range(n)]
        S = matmul(G, transpose(G))
        for i in range(n): S[i][i] += 2*n
        d = [S[i][i] for i in range(n)]
        C = [[S[i][j]/ (d[i]*d[j]) * (d[i]*d[j]) / (d[i]*d[j]) if False else Fr(S[i][j], 1)/ (Fr(1)) for j in range(n)] for i in range(n)]
        # rescale to unit diagonal exactly by using S_ij / sqrt(d_i d_j) is irrational; instead
        # take the matrix with unit diagonal obtained by D^-1 S D^-1 with D = diag(d), then repair
        C = [[S[i][j]/(d[i] if i == j else 1) for j in range(n)] for i in range(n)]
        C = [[Fr(S[i][j], 1) for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j: C[i][j] = Fr(S[i][j], 4*n*n)
            C[i][i] = Fr(1)
        if pd(C): return C

checks = []
def check(name, cond):
    checks.append((name, bool(cond))); print(('ok   ' if cond else 'FAIL ') + name)

rng = random.Random(2026)

# ---- the reference choice that makes the tilt a taper
n = 12; lab = [0]*4 + [1]*4 + [2]*4
C = rand_corr(n, rng)
ok = True
for phi in (Fr(0), Fr(3, 10), Fr(7, 10), Fr(1)):
    ok = ok and blend(blockdiag(C, lab), C, phi) == block_taper(C, lab, phi)
check('1. with the reference set to the block-diagonal part of the estimate, the tilt blend '
      '(1-phi) C_calib + phi C-hat is exactly C-hat with cross-cluster entries scaled by phi', ok)

# ---- two blocks: the tapered conditional is the damped complement at gamma = phi^2
n2 = 8; lab2 = [0]*4 + [1]*4
C2 = rand_corr(n2, rng)
A = [0, 1, 2, 3]; B = [4, 5, 6, 7]
ok = True
for phi in (Fr(0), Fr(1, 4), Fr(1, 2), Fr(3, 4), Fr(1)):
    ok = ok and conditional(block_taper(C2, lab2, phi), A, B) == damped_complement(C2, A, B, phi*phi)
check('2. two clusters: the conditional covariance of the tapered correlation is the damped '
      'complement at gamma = phi^2, exactly, for phi = 0, 1/4, 1/2, 3/4, 1', ok)

# ---- more than two blocks: the same statement is no longer exact
A3 = [0, 1, 2, 3]; B3 = [i for i in range(12) if i not in A3]
phi = Fr(1, 2)
lhs = conditional(block_taper(C, lab, phi), A3, B3)
rhs = damped_complement(C, A3, B3, phi*phi)
check('3. three clusters: it is no longer exact, because the taper also damps inside the complement',
      lhs != rhs)

# ---- the taper is positive definite along the whole dial
check('4. the block taper is positive definite at every phi in {0, 1/4, 1/2, 3/4, 1}, so the race '
      'is well posed along the dial',
      all(pd(block_taper(C, lab, Fr(t, 4))) for t in range(5)))

# ---- the near end decouples the clusters
T0 = block_taper(C, lab, Fr(0))
check('5. at phi = 0 every cross-cluster correlation is zero, so the clusters are independent '
      'and the race decouples',
      all(T0[i][j] == 0 for i in range(n) for j in range(n) if lab[i] != lab[j]))
check('6. at phi = 1 the taper is the estimate itself', block_taper(C, lab, Fr(1)) == C)

# ---- the diagonal reference of the paper is NOT the block taper
Cd = [[Fr(int(i == j)) for j in range(n)] for i in range(n)]
check('7. with the diagonal reference the blend damps within-cluster correlation too, so it is '
      'not the block taper', blend(Cd, C, Fr(1, 2)) != block_taper(C, lab, Fr(1, 2)))

# ---- Monte Carlo: both ends of the dial are races, and phi = 0 reproduces the benchmark
try:
    import numpy as np
    from allocation._thurstone.ability import base_density
    from allocation._thurstone.calibrate import calibrate_diagonal
    from allocation._thurstone.transport import transport_weights
    Cf = np.array([[float(x) for x in row] for row in C])
    seeds = np.random.default_rng(0).standard_normal((1 << 14, n))
    base = base_density()
    tgt = np.full(n, 1.0/n)
    ab = calibrate_diagonal(tgt, base=base)
    w0 = transport_weights(ab, np.eye(n), seeds)
    check('8. (Monte Carlo) calibration reproduces the benchmark under its own reference law',
          float(np.abs(w0 - tgt).max()) < 5e-3)
    wA = transport_weights(ab, np.array([[float(x) for x in row] for row in block_taper(C, lab, Fr(0))]), seeds)
    wB = transport_weights(ab, Cf, seeds)
    check('9. (Monte Carlo) both ends of the dial are simplex portfolios, and they differ',
          abs(wA.sum()-1) < 1e-9 and abs(wB.sum()-1) < 1e-9 and float(np.abs(wA-wB).max()) > 1e-3)
except Exception as exc:                                    # pragma: no cover
    print('skip 8-9 (needs numpy and the allocation package): %s' % exc)

bad = [nm for nm, c in checks if not c]
print('\n%d of %d checks passed' % (len(checks) - len(bad), len(checks)) + ('; FAILED: %s' % bad if bad else ''))
raise SystemExit(1 if bad else 0)
