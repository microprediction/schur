/* The Schur-complementary recursion of the paper (floats; direct port of the
   reference implementations), shared by all demo pages, plus the noise models
   and small canvas helpers. Everything is computed live; nothing is pre-baked.
   Exact rational certificates for the same claims:
   paper/neither_end_certificate.py in the repo. */
(function () {
  // ---------- core recursion ----------
  function mmul(X, Y) {
    const n = X.length, m = Y[0].length, k = Y.length, O = [];
    for (let i = 0; i < n; i++) {
      O.push([]);
      for (let j = 0; j < m; j++) {
        let s = 0;
        for (let t = 0; t < k; t++) s += X[i][t] * Y[t][j];
        O[i].push(s);
      }
    }
    return O;
  }
  function mT(X) { return X[0].map((_, j) => X.map(r => r[j])); }
  function minv2(X) {
    const [[a, b], [c, d]] = X, det = a * d - b * c;
    return [[d / det, -b / det], [-c / det, a / det]];
  }
  function aug(A, B, D, g) {
    const BDi = mmul(B, minv2(D));
    const BB = mmul(BDi, mT(B));
    const a0 = [[A[0][0] - g * BB[0][0], A[0][1] - g * BB[0][1]],
                [A[1][0] - g * BB[1][0], A[1][1] - g * BB[1][1]]];
    const r = [[1 - g * BDi[0][0], -g * BDi[0][1]], [-g * BDi[1][0], 1 - g * BDi[1][1]]];
    const M = mmul(minv2(r), a0);
    return [[M[0][0], (M[0][1] + M[1][0]) / 2], [(M[0][1] + M[1][0]) / 2, M[1][1]]];
  }
  function naive(X) {
    const w = [1 / X[0][0], 1 / X[1][1]], s = w[0] + w[1], u = [w[0] / s, w[1] / s];
    return u[0] * u[0] * X[0][0] + 2 * u[0] * u[1] * X[0][1] + u[1] * u[1] * X[1][1];
  }
  function weights4(S, g) {
    const A = [[S[0][0], S[0][1]], [S[1][0], S[1][1]]],
          D = [[S[2][2], S[2][3]], [S[3][2], S[3][3]]],
          B = [[S[0][2], S[0][3]], [S[1][2], S[1][3]]];
    const Aa = g ? aug(A, B, D, g) : A, Da = g ? aug(D, mT(B), A, g) : D;
    const vL = naive(Aa), vR = naive(Da), aL = vR / (vL + vR);
    const iA = [1 / Aa[0][0], 1 / Aa[1][1]], iD = [1 / Da[0][0], 1 / Da[1][1]];
    return [aL * iA[0] / (iA[0] + iA[1]), aL * iA[1] / (iA[0] + iA[1]),
            (1 - aL) * iD[0] / (iD[0] + iD[1]), (1 - aL) * iD[1] / (iD[0] + iD[1])];
  }
  function Vof(w, ST) {
    let v = 0;
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) v += w[i] * ST[i][j] * w[j];
    return v;
  }

  // ---------- families ----------
  // Family A: paper eq. (16). vols (1,1,2,2), within corr 1/2, cross corr 0.3.
  function famA(eps, mode) { // mode 'rho' (common cross) or 'entry' (one entry)
    const rc = 0.3, S = [[1, 0.5, 0, 0], [0.5, 1, 0, 0], [0, 0, 4, 2], [0, 0, 2, 4]];
    for (const [i, j] of [[0, 2], [0, 3], [1, 2], [1, 3]]) { S[i][j] = 2 * rc; S[j][i] = 2 * rc; }
    if (mode === 'rho') {
      for (const [i, j] of [[0, 2], [0, 3], [1, 2], [1, 3]]) { S[i][j] += 2 * eps; S[j][i] += 2 * eps; }
    } else { S[0][2] += 2 * eps; S[2][0] += 2 * eps; }
    return S;
  }
  // Family B: the boundary example. A = D = [[1,-1/2],[-1/2,2]], cross 0.1.
  function famB(eps) {
    const A = [[1, -0.5], [-0.5, 2]], b = 0.1;
    const S = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
      S[i][j] = A[i][j]; S[2 + i][2 + j] = A[i][j]; S[i][2 + j] = b; S[2 + i][j] = b;
    }
    S[0][2] += eps; S[2][0] += eps;
    return S;
  }
  function Fof(fam, mode, g, tau) {
    const ST = fam === 'A' ? famA(0, mode) : famB(0);
    const Sp = fam === 'A' ? famA(tau, mode) : famB(tau);
    const Sm = fam === 'A' ? famA(-tau, mode) : famB(-tau);
    return 0.5 * (Vof(weights4(Sp, g), ST) + Vof(weights4(Sm, g), ST));
  }
  function dF0(fam, mode, tau) {
    const h = 1e-7;
    return (Fof(fam, mode, h, tau) - Fof(fam, mode, 0, tau)) / h;
  }
  function argminF(F, lo, hi) { // ternary search on a callable
    for (let k = 0; k < 70; k++) {
      const m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3;
      if (F(m1) < F(m2)) hi = m2; else lo = m1;
    }
    return (lo + hi) / 2;
  }

  // ---------- whole-matrix noise (rank-one exchangeable; paper Section 6) ----------
  const ST_A = famA(0, 'rho');
  function famWhole(ea, ed, ez, s1, s2, s3) {
    const S = famA(0, 'rho');
    for (const i of [0, 1]) for (const j of [0, 1]) S[i][j] += s1 * ea;
    for (const i of [2, 3]) for (const j of [2, 3]) S[i][j] += s2 * ed;
    for (const [i, j] of [[0, 2], [0, 3], [1, 2], [1, 3]]) { S[i][j] += s3 * ez; S[j][i] += s3 * ez; }
    return S;
  }
  function Fwhole(g, ea, ed, ez) {
    let tot = 0;
    for (const s1 of [1, -1]) for (const s2 of [1, -1]) for (const s3 of [1, -1])
      tot += Vof(weights4(famWhole(ea, ed, ez, s1, s2, s3), g), ST_A);
    return tot / 8;
  }
  // Xi for independent channels, family numbers (Theorem 5(iv)).
  function XiOf(ea, ed, ez) {
    return 9963 / 320 * ez * ez + 1674 / 125 * ea * ea - 189 / 500 * ed * ed;
  }
  // 1 - gamma* ~ Xi / ((d-a)^2 c^2 K) = Xi / (37179/8000)
  const XI_DENOM = 37179 / 8000;

  // ---------- entrywise noise (independent signs; paper Section 6) ----------
  const VOLS = [1, 1, 2, 2];
  const RHOS = [[1, 0.5, 0.3, 0.3], [0.5, 1, 0.3, 0.3], [0.3, 0.3, 1, 0.5], [0.3, 0.3, 0.5, 1]];
  const OFFD = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
  const DIAG = [[0, 0], [1, 1], [2, 2], [3, 3]];
  function famEnt(tau, entries, bits) {
    const S = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) S[i][j] = VOLS[i] * VOLS[j] * RHOS[i][j];
    entries.forEach(([i, j], k) => {
      const s = (bits >> k) & 1 ? 1 : -1;
      if (i === j) S[i][i] = VOLS[i] * VOLS[i] * (1 + s * tau);
      else { S[i][j] = VOLS[i] * VOLS[j] * (RHOS[i][j] + s * tau); S[j][i] = S[i][j]; }
    });
    return S;
  }
  // Mean over all sign states; states with |V| > bigCut stand in for the exactly
  // singular branches of the rational computation and are counted separately.
  function Fent(g, tau, all) {
    const entries = all ? OFFD.concat(DIAG) : OFFD;
    const n = 1 << entries.length;
    let tot = 0, kept = 0, nBig = 0;
    const bigCut = 1e6;
    for (let b = 0; b < n; b++) {
      const V = Vof(weights4(famEnt(tau, entries, b), g), ST_A);
      if (!isFinite(V) || Math.abs(V) > bigCut) { nBig++; continue; }
      tot += V; kept++;
    }
    return { mean: tot / kept, nBig, nStates: n };
  }

  // ---------- sample covariance (seeded Monte Carlo; paper Section 6) ----------
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function chol4(S) {
    const L = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
    for (let i = 0; i < 4; i++) for (let j = 0; j <= i; j++) {
      let s = S[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(s) : s / L[j][j];
    }
    return L;
  }
  const CHOL_A = chol4(ST_A);
  function wisSample(T, N, seed) { // N sample covariances of T draws each
    const rnd = mulberry32(seed);
    function randn() {
      let u = 0, v = 0;
      while (u === 0) u = rnd();
      v = rnd();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    const out = [];
    for (let n = 0; n < N; n++) {
      const S = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
      for (let t = 0; t < T; t++) {
        const z = [randn(), randn(), randn(), randn()], x = [0, 0, 0, 0];
        for (let i = 0; i < 4; i++) for (let k = 0; k <= i; k++) x[i] += CHOL_A[i][k] * z[k];
        for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) S[i][j] += x[i] * x[j];
      }
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) S[i][j] /= T;
      out.push(S);
    }
    return out;
  }
  function wisF(Shats, g) {
    let tot = 0;
    for (const S of Shats) tot += Vof(weights4(S, g), ST_A);
    return tot / Shats.length;
  }

  // ---------- closed form for the solved family (paper eq. (20)) ----------
  const FA = { a: 0.75, d: 3, c: 0.6 };
  FA.S = FA.a + FA.d; FA.K = FA.S - 2 * FA.c; FA.xstar = (FA.d - FA.c) / FA.K;
  function Qx(x) {
    return FA.a * x * x + FA.d * (1 - x) * (1 - x) + 2 * FA.c * x * (1 - x);
  }
  function Fclosed(g, e) {
    let tot = 0;
    for (const s of [1, -1]) {
      const z = FA.c + s * e;
      tot += Math.pow((g * z - FA.c) / (FA.S - 2 * g * z), 2);
    }
    return Qx(FA.xstar) + ((FA.d - FA.a) * (FA.d - FA.a) / FA.K) * tot / 2;
  }

  // ---------- canvas helpers ----------
  const dpr = window.devicePixelRatio || 1;
  function setup(id, cssH) {
    const c = document.getElementById(id), cx = c.getContext('2d');
    const W = c.clientWidth || 820;
    c.width = W * dpr; c.height = cssH * dpr; c.style.height = cssH + 'px';
    cx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { c, cx, W, H: cssH };
  }
  function axes(cx, W, H, padL, padB, title) {
    cx.strokeStyle = '#ddd'; cx.strokeRect(padL, 10, W - padL - 10, H - padB - 10);
    cx.fillStyle = '#666'; cx.font = '11px sans-serif'; cx.textAlign = 'left';
    cx.fillText(title, padL, 8);
  }

  window.SCHUR = { weights4, Vof, famA, famB, Fof, dF0, argminF,
                   famWhole, Fwhole, XiOf, XI_DENOM,
                   Fent, wisSample, wisF, Qx, Fclosed, FA,
                   setup, axes };
})();
