/* Headless regressions for the NCO bridge demos. Run: node docs/nco-check.js */
const N = require('./nco.js');
let fails = 0;
const ok = (name, cond, detail) => { if (!cond) fails++; console.log((cond ? 'PASS' : 'FAIL') + '  ' + name + (detail !== undefined ? '  ' + detail : '')); };
const close = (a, b, tol) => Math.abs(a - b) < (tol || 1e-9);
// issue 11: interior vanishing direction, continuous extension
{ const S = [[4, 1.5], [1.5, 1]], idx = [[0], [1]], kn = [0, 1];
  for (const g of [2 / 3 - 1e-6, 2 / 3, 2 / 3 + 1e-6]) { const w = N.bridge(S, idx, kn, g).w; const v = N.quad(S, w);
    ok('issue 11: gamma=' + g.toFixed(7) + ' gives (-1/4, 5/4), variance 7/8', close(w[0], -0.25) && close(w[1], 1.25) && close(v, 7 / 8), w.map(x => x.toFixed(4)).join(',') + ' var ' + v.toFixed(5)); } }
// issue 12: scale invariance
{ for (const c of [1, 1e12, 1e-9]) { const w = N.bridge([[c, 0], [0, 2 * c]], [[0], [1]], [0, 1], 0).w; ok('issue 12: diag scale ' + c + ' -> (2/3, 1/3)', close(w[0], 2 / 3) && close(w[1], 1 / 3), w.join(',')); }
  const m = N.gatewayModel(3, [2, 1, 3]); const ref = N.bridge(m.Sigma, m.clusters, m.knots, 0.4).w;
  const big = N.bridge(m.Sigma.map(r => r.map(v => v * 1e9)), m.clusters, m.knots, 0.4).w;
  ok('issue 12: random gateway model, covariance x 1e9', Math.max(...ref.map((v, i) => Math.abs(v - big[i]))) < 1e-9); }
// issue 13: both duplicated-knot paths share their ends at every epsilon
{ for (const eps of [1, 0.1, 1e-4]) { const a = N.duplicatedKnotsV0(0, eps), b = N.duplicatedKnotsV0(1, eps), la = N.duplicatedKnotsLambdaPath(0, eps), lb = N.duplicatedKnotsLambdaPath(1, eps);
    ok('issue 13: eps=' + eps + ' ends shared', close(a, la) && close(b, lb), 'NCO ' + a.toFixed(5) + ' opt ' + b.toFixed(5)); }
  ok('issue 13: limits 3/8 and 1/3 as eps -> 0', close(N.duplicatedKnotsV0(0, 1e-9), 3 / 8, 1e-6) && close(N.duplicatedKnotsV0(1, 1e-9), 1 / 3, 1e-6)); }
// issue 14: admissible tau
{ ok('issue 14: k=12, c=0.02, tau=0.12 is rejected', !N.estimateAdmissible(12, 0.02 - 0.12));
  const tm = N.symmetricNoiseTauMax(12, 0.02); ok('issue 14: tau just inside the bound is admissible', N.estimateAdmissible(12, 0.02 - (tm - 1e-6)) && N.estimateAdmissible(12, 0.02 + (tm - 1e-6)), 'tauMax ' + tm.toFixed(4)); }
// issue 15: the expansion can leave the bridge; the optimum is the numerical minimum
{ const k = 2, c = 0.02, d = 1, tau = 0.05; const pred = 1 - N.shiftCoefficient(k, c, d) * tau * tau; const F = g => N.F_symmetric(g, tau, k, c, d); const best = N.argmin(F, 0, 1, 4000);
  ok('issue 15: prediction is outside [0,1] here', pred < 0 || pred > 1, pred.toFixed(3));
  ok('issue 15: feasible minimizer ~ 0.1319 with nonnegative endpoint costs', close(best.g, 0.13186, 2e-3) && F(0) - best.v >= 0 && F(1) - best.v >= 0, best.g.toFixed(5)); }
// standing claims
{ const F = g => N.F_twoPoint(g, 2, 0.25, 1); ok('two-thirds gaps 1/576, 1/900', close(F(0) - F(2 / 3), 1 / 576, 1e-12) && close(F(1) - F(2 / 3), 1 / 900, 1e-12));
  ok('threshold delta=8 at k=10, c=1/4', Math.abs(N.shiftCoefficient(10, 0.25, 8)) < 1e-3);
  const m = N.gatewayModel(7, [2, 3, 1, 2]); const w1 = N.bridge(m.Sigma, m.clusters, m.knots, 1).w, mv = N.minVar(m.Sigma);
  ok('gamma=1 equals minimum variance on a random gateway model', Math.max(...w1.map((v, i) => Math.abs(v - mv[i]))) < 1e-9); }
console.log(fails ? fails + ' FAILURE(S)' : 'all demo checks pass');
process.exit(fails ? 1 : 0);
