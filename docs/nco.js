/* Shared code for the NCO bridge demos (schur.microprediction.org).
   Companion to "A Schur Bridge from Nested Clustered Optimization to Global Minimum Variance".
   Runs in the browser (window.NCO) and in node (module.exports) so the numerical claims can be
   checked headlessly. No dependencies. */
(function (root) {
  'use strict';

  /* ---------- linear algebra ---------- */
  function solve(A, b) {
    const n = A.length, M = A.map((r, i) => r.slice().concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      const d = M[c][c];
      if (Math.abs(d) < 1e-300) throw new Error('singular');
      for (let j = c; j <= n; j++) M[c][j] /= d;
      for (let r = 0; r < n; r++) if (r !== c && M[r][c] !== 0) {
        const f = M[r][c];
        for (let j = c; j <= n; j++) M[r][j] -= f * M[c][j];
      }
    }
    return M.map(r => r[n]);
  }
  function inv(A) {
    const n = A.length, cols = [];
    for (let j = 0; j < n; j++) cols.push(solve(A, A.map((_, i) => (i === j ? 1 : 0))));
    return A.map((_, i) => cols.map(c => c[i]));
  }
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  const matvec = (A, x) => A.map(r => dot(r, x));
  const quad = (S, w) => dot(w, matvec(S, w));
  const sub = (A, rows, cols) => rows.map(r => cols.map(c => A[r][c]));

  /* ---------- seeded random gateway model ---------- */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rnd) {
    const u = 1 - rnd(), v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function randomSPD(rnd, n, scale) {
    const A = [];
    for (let i = 0; i < n; i++) { A.push([]); for (let j = 0; j < n; j++) A[i].push(gauss(rnd)); }
    const S = [];
    for (let i = 0; i < n; i++) { S.push([]); for (let j = 0; j < n; j++) S[i].push((scale || 1) * (dot(A[i], A[j]) / n + (i === j ? 1 : 0))); }
    return S;
  }
  /* sizes[j] = number of remaining members in cluster j; index 0 of each cluster is its knot */
  function gatewayModel(seed, sizes) {
    const rnd = mulberry32(seed), k = sizes.length;
    const clusters = [], knots = [];
    let pos = 0;
    for (const m of sizes) { const I = []; for (let t = 0; t <= m; t++) I.push(pos + t); clusters.push(I); knots.push(pos); pos += m + 1; }
    const n = pos, S = randomSPD(rnd, k, 1);
    const betas = sizes.map(m => Array.from({ length: m }, () => 1 + 0.8 * gauss(rnd)));
    const resid = sizes.map(m => randomSPD(rnd, m, 0.3));
    const Sigma = Array.from({ length: n }, () => new Array(n).fill(0));
    clusters.forEach((Ij, j) => clusters.forEach((Il, l) => {
      const s = S[j][l];
      Ij.forEach((a, ia) => Il.forEach((b, ib) => {
        const la = ia === 0 ? 1 : betas[j][ia - 1], lb = ib === 0 ? 1 : betas[l][ib - 1];
        Sigma[a][b] += la * lb * s;
      }));
    }));
    clusters.forEach((I, j) => { for (let a = 1; a < I.length; a++) for (let b = 1; b < I.length; b++) Sigma[I[a]][I[b]] += resid[j][a - 1][b - 1]; });
    return { Sigma, clusters, knots, betas, resid, S };
  }

  /* ---------- the bridge ---------- */
  function conditionedPair(Sigma, I, others, gamma, u) {
    const n = Sigma.length; u = u || new Array(n).fill(1);
    const Sii = sub(Sigma, I, I), b = I.map(a => u[a]);
    if (others.length === 0) return { Q: Sii, b };
    const Spp = sub(Sigma, others, others), Sip = sub(Sigma, I, others);
    const coef = Sip.map(row => solve(Spp, row).map(v => gamma * v));       /* gamma * S_ip S_pp^{-1} */
    const Q = Sii.map((row, r) => row.map((v, c) => v - dot(coef[r], Sip[c])));
    const bb = b.map((v, r) => v - dot(coef[r], others.map(a => u[a])));
    return { Q, b: bb };
  }
  function directions(Sigma, clusters, knots, gamma, u) {
    return clusters.map((I, i) => {
      const others = knots.filter((_, j) => j !== i);
      const { Q, b } = conditionedPair(Sigma, I, others, gamma, u);
      return solve(Q, b);
    });
  }
  /* w ∝ D (D' S D)^{-1} D' u, a direction that vanishes is dropped */
  function bridge(Sigma, clusters, knots, gamma, u) {
    const n = Sigma.length; u = u || new Array(n).fill(1);
    const ds = directions(Sigma, clusters, knots, gamma, u);
    const cols = [];
    clusters.forEach((I, i) => { if (ds[i].some(v => Math.abs(v) > 1e-12)) { const c = new Array(n).fill(0); I.forEach((a, r) => c[a] = ds[i][r]); cols.push(c); } });
    const SD = cols.map(c => matvec(Sigma, c));
    const G = cols.map((c, i) => cols.map((_, j) => dot(c, SD[j])));
    const a = solve(G, cols.map(c => dot(c, u)));
    const w = new Array(n).fill(0);
    cols.forEach((c, i) => c.forEach((v, r) => w[r] += v * a[i]));
    const tot = w.reduce((s, v) => s + v, 0);
    return { w: w.map(v => v / tot), directions: ds };
  }
  function minVar(Sigma) { const w = solve(Sigma, new Array(Sigma.length).fill(1)); const t = w.reduce((s, v) => s + v, 0); return w.map(v => v / t); }

  /* ---------- identical clusters: the scalar picture ---------- */
  const t_of = (g, z, k) => (1 + (k - 2) * z - g * (k - 1) * z) / (1 + (k - 2) * z - g * (k - 1) * z * z);
  const x_of = (g, z, k, delta) => { const t = t_of(g, z, k); return t / (t + delta); };
  const V_of = (x, k, c, delta) => ((1 + (k - 1) * c) * x * x + (1 - x) * (1 - x) / delta) / k;
  const xStar = (k, c, delta) => 1 / (1 + delta * (1 + (k - 1) * c));
  const gammaStarTwoPoint = (k, c) => (1 + 2 * (k - 2) * c) / (2 * (1 + (k - 3) * c));
  function F_twoPoint(g, k, c, delta) { return 0.5 * (V_of(x_of(g, 0, k, delta), k, c, delta) + V_of(x_of(g, 2 * c, k, delta), k, c, delta)); }
  function F_symmetric(g, tau, k, c, delta) { return 0.5 * (V_of(x_of(g, c + tau, k, delta), k, c, delta) + V_of(x_of(g, c - tau, k, delta), k, c, delta)); }
  /* G'(1)/V0''(1) by central differences on the scalar formulas (analytic beyond gamma = 1) */
  function shiftCoefficient(k, c, delta) {
    const h = 1e-3, tau = 1e-3;
    const G = g => (F_symmetric(g, tau, k, c, delta) - F_symmetric(g, 0, k, c, delta)) / (tau * tau);
    const Gp = (G(1 + h) - G(1 - h)) / (2 * h);
    const V0 = g => F_symmetric(g, 0, k, c, delta);
    const Vpp = (V0(1 + h) - 2 * V0(1) + V0(1 - h)) / (h * h);
    return Gp / Vpp;
  }
  function argmin(f, lo, hi, n) {
    let best = lo, bv = Infinity;
    for (let i = 0; i <= n; i++) { const g = lo + (hi - lo) * i / n, v = f(g); if (v < bv) { bv = v; best = g; } }
    const w = (hi - lo) / n;
    for (let i = 0; i <= 200; i++) { const g = Math.max(lo, Math.min(hi, best - w + 2 * w * i / 200)), v = f(g); if (v < bv) { bv = v; best = g; } }
    return { g: best, v: bv };
  }

  /* ---------- effective damping and the compressed knot system ---------- */
  const lambda_of = (g, r) => g * r / (1 - g + g * r);
  /* knot exposures kappa_i(gamma) for knot covariance S (u = 1) */
  function kappas(S, gamma) {
    const k = S.length, one = new Array(k).fill(1), h = solve(S, one), Si = inv(S);
    return S.map((row, i) => { const s = row[i], nu = 1 / Si[i][i]; return ((1 - gamma) + gamma * nu * h[i]) / ((1 - gamma) * s + gamma * nu); });
  }
  /* variance of the compressed two-tier portfolio: knots S_hat for the estimate, S_pop for evaluation, member precisions delta */
  function compressedVariance(S_hat, S_pop, delta, kappa) {
    const k = S_hat.length;
    const M = S_hat.map((row, i) => row.map((v, j) => kappa[i] * v * kappa[j] + (i === j ? delta[i] : 0)));
    const b = kappa.map((v, i) => v + delta[i]);
    const a = solve(M, b), J = dot(b, a);
    const q = kappa.map((v, i) => v * a[i] / J);
    return quad(S_pop, q) + a.reduce((s, ai, i) => s + (ai / J) * (ai / J) * delta[i], 0);
  }
  /* duplicated knots: S = 11' + eps I, two clusters with unit residual precision */
  function duplicatedKnotsV0(gamma, eps) {
    const S = [[1 + eps, 1], [1, 1 + eps]], Spop = [[1, 1], [1, 1]];
    return compressedVariance(S, Spop, [1, 1], kappas(S, gamma));
  }
  function duplicatedKnotsLambdaPath(lambda) {
    const kap = [1 - lambda / 2, 1 - lambda / 2];          /* (1-lambda)/s + lambda h, h = S^+ 1 = (1/2, 1/2) */
    return compressedVariance([[1, 1], [1, 1]], [[1, 1], [1, 1]], [1, 1], kap);
  }
  /* lost precision: V0(gamma) - V* and the closed form L/(Z(Z-L)) */
  function lostPrecision(S, delta, gamma) {
    const k = S.length, one = new Array(k).fill(1), h = solve(S, one), kap = kappas(S, gamma);
    const Z = h.reduce((s, v) => s + v, 0) + delta.reduce((s, v) => s + v, 0);
    const Si = inv(S), W = inv(Si.map((row, i) => row.map((v, j) => v + (i === j ? kap[i] * kap[i] / delta[i] : 0))));
    const d = h.map((v, i) => v - kap[i]), L = quad(W, d);
    return { computed: compressedVariance(S, S, delta, kap) - 1 / Z, formula: L / (Z * (Z - L)) };
  }

  /* ---------- a small line chart with legend, crosshair and tooltip ---------- */
  const PALETTE = ['#4a3aff', '#ef6c00', '#0f8b6e', '#c0392b'];   /* validated, light surface */
  function lineChart(canvasId, opts) {
    if (typeof document === 'undefined') return;
    const c = document.getElementById(canvasId), cx = c.getContext('2d');
    const dpr = window.devicePixelRatio || 1, W = c.clientWidth || 820, H = opts.height || 300;
    c.width = W * dpr; c.height = H * dpr; c.style.height = H + 'px'; cx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cx.clearRect(0, 0, W, H);
    const padL = 52, padR = 16, padT = 26, padB = 34;
    const series = opts.series;
    let xmin = opts.xmin, xmax = opts.xmax, ymin = opts.ymin, ymax = opts.ymax;
    const allx = series.flatMap(s => s.points.map(p => p[0])), ally = series.flatMap(s => s.points.map(p => p[1])).filter(isFinite);
    if (xmin == null) xmin = Math.min(...allx); if (xmax == null) xmax = Math.max(...allx);
    if (ymin == null) ymin = Math.min(...ally); if (ymax == null) ymax = Math.max(...ally);
    if (ymax - ymin < 1e-12) { ymax += 1e-6; ymin -= 1e-6; }
    const padY = (ymax - ymin) * 0.06; ymin -= padY; ymax += padY;
    const X = x => padL + (x - xmin) / (xmax - xmin) * (W - padL - padR);
    const Y = y => H - padB - (y - ymin) / (ymax - ymin) * (H - padB - padT);
    /* grid and axes */
    cx.strokeStyle = '#ececec'; cx.lineWidth = 1;
    const yt = niceTicks(ymin, ymax, 5), xt = niceTicks(xmin, xmax, 6);
    yt.forEach(v => { cx.beginPath(); cx.moveTo(padL, Y(v)); cx.lineTo(W - padR, Y(v)); cx.stroke(); });
    cx.strokeStyle = '#cfcfcf'; cx.beginPath(); cx.moveTo(padL, padT); cx.lineTo(padL, H - padB); cx.lineTo(W - padR, H - padB); cx.stroke();
    cx.fillStyle = '#777'; cx.font = '11px sans-serif'; cx.textAlign = 'right';
    yt.forEach(v => cx.fillText(fmt(v), padL - 6, Y(v) + 4));
    cx.textAlign = 'center'; xt.forEach(v => cx.fillText(fmt(v), X(v), H - padB + 14));
    cx.fillStyle = '#666'; cx.textAlign = 'left'; cx.fillText(opts.title || '', padL, 14);
    if (opts.xlabel) { cx.textAlign = 'center'; cx.fillText(opts.xlabel, (padL + W - padR) / 2, H - 4); }
    /* reference lines */
    (opts.vlines || []).forEach(v => { cx.strokeStyle = v.color || '#999'; cx.setLineDash([4, 4]); cx.beginPath(); cx.moveTo(X(v.x), padT); cx.lineTo(X(v.x), H - padB); cx.stroke(); cx.setLineDash([]);
      cx.fillStyle = v.color || '#999'; const right = X(v.x) > W - padR - 150; cx.textAlign = right ? 'right' : 'left'; cx.fillText(v.label || '', X(v.x) + (right ? -4 : 4), padT + 12); });
    (opts.hlines || []).forEach(v => { cx.strokeStyle = v.color || '#999'; cx.setLineDash([4, 4]); cx.beginPath(); cx.moveTo(padL, Y(v.y)); cx.lineTo(W - padR, Y(v.y)); cx.stroke(); cx.setLineDash([]);
      cx.fillStyle = v.color || '#999'; cx.textAlign = 'right'; cx.fillText(v.label || '', W - padR - 4, Y(v.y) - 4); });
    /* series */
    series.forEach((s, i) => {
      cx.strokeStyle = s.color || PALETTE[i % PALETTE.length]; cx.lineWidth = 2; cx.setLineDash(s.dash || []);
      cx.beginPath(); let started = false;
      s.points.forEach(p => { if (!isFinite(p[1])) { started = false; return; } started ? cx.lineTo(X(p[0]), Y(p[1])) : cx.moveTo(X(p[0]), Y(p[1])); started = true; });
      cx.stroke(); cx.setLineDash([]);
    });
    /* legend */
    if (series.length > 1) {
      let lx = padL + 8; const ly = padT + 12;
      cx.font = '11px sans-serif'; cx.textAlign = 'left';
      series.forEach((s, i) => { if (!s.name || !s.name.trim()) return; const col = s.color || PALETTE[i % PALETTE.length]; cx.fillStyle = col; cx.fillRect(lx, ly - 5, 14, 3); cx.fillStyle = '#333'; cx.fillText(s.name, lx + 18, ly); lx += 18 + cx.measureText(s.name).width + 16; });
    }
    /* hover: crosshair and tooltip */
    let tip = c.parentNode.querySelector('.nco-tip');
    if (!tip) { tip = document.createElement('div'); tip.className = 'nco-tip'; tip.style.cssText = 'position:absolute;pointer-events:none;background:#fff;border:1px solid #ddd;border-radius:5px;padding:6px 8px;font-size:12px;color:#333;display:none;box-shadow:0 1px 4px rgba(0,0,0,.08);white-space:nowrap'; c.parentNode.style.position = 'relative'; c.parentNode.appendChild(tip); }
    const snapshot = cx.getImageData(0, 0, c.width, c.height);
    c.onmousemove = ev => {
      const r = c.getBoundingClientRect(), mx = ev.clientX - r.left, x = xmin + (mx - padL) / (W - padL - padR) * (xmax - xmin);
      cx.putImageData(snapshot, 0, 0);
      if (mx < padL || mx > W - padR) { tip.style.display = 'none'; return; }
      cx.strokeStyle = '#bbb'; cx.beginPath(); cx.moveTo(mx, padT); cx.lineTo(mx, H - padB); cx.stroke();
      let html = '<b>' + (opts.xname || 'x') + ' = ' + x.toFixed(3) + '</b>';
      series.forEach((s, i) => {
        if (!s.name || !s.name.trim()) return;
        let best = null; s.points.forEach(p => { if (isFinite(p[1]) && (best === null || Math.abs(p[0] - x) < Math.abs(best[0] - x))) best = p; });
        if (!best) return;
        const col = s.color || PALETTE[i % PALETTE.length];
        cx.fillStyle = '#fff'; cx.beginPath(); cx.arc(X(best[0]), Y(best[1]), 5, 0, 7); cx.fill();
        cx.fillStyle = col; cx.beginPath(); cx.arc(X(best[0]), Y(best[1]), 3.5, 0, 7); cx.fill();
        html += '<br><span style="display:inline-block;width:10px;height:3px;background:' + col + ';margin-right:6px;vertical-align:middle"></span>' + s.name + ': ' + fmt(best[1], 5);
      });
      tip.innerHTML = html; tip.style.display = 'block';
      tip.style.left = Math.min(mx + 14, W - 200) + 'px'; tip.style.top = (padT + 6) + 'px';
    };
    c.onmouseleave = () => { cx.putImageData(snapshot, 0, 0); tip.style.display = 'none'; };
  }
  function niceTicks(lo, hi, n) {
    const span = hi - lo, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map(m => m * mag).find(s => s >= raw) || mag * 10;
    const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(v); return out;
  }
  function fmt(v, d) { if (!isFinite(v)) return '–'; const a = Math.abs(v); if (a >= 1000 || (a < 1e-3 && a > 0)) return v.toExponential(2); return +v.toFixed(d || (a < 1 ? 4 : 3)) + ''; }

  const api = { solve, inv, dot, matvec, quad, mulberry32, gatewayModel, conditionedPair, directions, bridge, minVar,
    t_of, x_of, V_of, xStar, gammaStarTwoPoint, F_twoPoint, F_symmetric, shiftCoefficient, argmin,
    lambda_of, kappas, compressedVariance, duplicatedKnotsV0, duplicatedKnotsLambdaPath, lostPrecision,
    lineChart, PALETTE, fmt };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.NCO = api;
})(typeof window !== 'undefined' ? window : globalThis);
