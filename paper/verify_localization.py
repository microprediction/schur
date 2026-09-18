"""Companion verification script for localization-is-schur-damping.tex.

  check_block_taper   Prop 1: C_gamma is PSD; the conditional covariance of the tapered matrix is
                      A - gamma B D^{-1} B'; the localized gain is sqrt(gamma) K*
  check_expected_error Prop 2: Monte Carlo over sampling noise agrees with
                      F(s) = A - (2s - s^2) Q + s^2 tau^2 G, and the trace is minimized at
                      s* = tr Q / (tr Q + tau^2 tr G), strictly inside (0, 1)
Run with `python verify_localization.py`; it needs only NumPy.
"""
import numpy as np


def random_spd(rng, n):
    M = rng.standard_normal((n, n))
    return M @ M.T / n + np.eye(n)


def check_block_taper(rng, trials=50):
    worst = {"taper_min_eig": 0.0, "tapered_min_eig": 0.0, "schur": 0.0, "gain": 0.0}
    for _ in range(trials):
        nA, nD = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        Sig = random_spd(rng, nA + nD)
        A, B, D = Sig[:nA, :nA], Sig[:nA, nA:], Sig[nA:, nA:]
        R = rng.uniform(0.1, 1.0) * np.eye(nD)
        Kstar = B @ np.linalg.inv(D + R)
        for g in (0.0, 0.25, 0.7, 1.0):
            C = np.ones_like(Sig); C[:nA, nA:] = np.sqrt(g); C[nA:, :nA] = np.sqrt(g)
            T = Sig * C
            worst["taper_min_eig"] = min(worst["taper_min_eig"], np.linalg.eigvalsh(C).min())
            worst["tapered_min_eig"] = min(worst["tapered_min_eig"], np.linalg.eigvalsh(T).min())
            schur = T[:nA, :nA] - T[:nA, nA:] @ np.linalg.solve(T[nA:, nA:], T[nA:, :nA])
            worst["schur"] = max(worst["schur"], np.abs(schur - (A - g * B @ np.linalg.solve(D, B.T))).max())
            gain = T[:nA, nA:] @ np.linalg.inv(T[nA:, nA:] + R)
            worst["gain"] = max(worst["gain"], np.abs(gain - np.sqrt(g) * Kstar).max())
    return worst


def check_expected_error(rng, draws=40000):
    nA, nD = 3, 4
    Sig = random_spd(rng, nA + nD)
    A, B, D = Sig[:nA, :nA], Sig[:nA, nA:], Sig[nA:, nA:]
    S = D + 0.5 * np.eye(nD)
    Sinv = np.linalg.inv(S)
    Q = B @ Sinv @ B.T
    tau = 0.6
    Es = rng.standard_normal((draws, nA, nD))
    G = np.einsum("kij,jl,kml->im", Es, Sinv, Es) / draws
    out = {"G_vs_trSinv_I": np.abs(G - np.trace(Sinv) * np.eye(nA)).max()}
    mc_gap = 0.0
    for s in (0.2, 0.6, 1.0):
        K = s * (B[None] + tau * Es) @ Sinv
        Fm = (A[None] - K @ B.T - B @ K.transpose(0, 2, 1) + K @ S @ K.transpose(0, 2, 1)).mean(axis=0)
        Ff = A - (2 * s - s * s) * Q + s * s * tau * tau * G
        mc_gap = max(mc_gap, np.abs(Fm - Ff).max())
    out["mc_vs_formula"] = mc_gap
    sstar = np.trace(Q) / (np.trace(Q) + tau * tau * np.trace(G))
    ss = np.linspace(0.001, 1.0, 20000)
    tr = np.trace(A) - (2 * ss - ss * ss) * np.trace(Q) + ss * ss * tau * tau * np.trace(G)
    out["s_star"] = sstar
    out["grid_argmin"] = float(ss[int(np.argmin(tr))])
    return out


def main():
    rng = np.random.default_rng(0)
    r = check_block_taper(rng); print("block taper        ", r)
    assert r["taper_min_eig"] > -1e-12 and r["tapered_min_eig"] > -1e-12
    assert r["schur"] < 1e-12 and r["gain"] < 1e-12
    r = check_expected_error(rng); print("expected error     ", r)
    assert r["mc_vs_formula"] < 2e-2 and r["G_vs_trSinv_I"] < 5e-2
    assert abs(r["s_star"] - r["grid_argmin"]) < 1e-3 and 0 < r["s_star"] < 1
    print("verification ok")


if __name__ == "__main__":
    main()
