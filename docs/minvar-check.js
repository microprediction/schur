// Headless regressions for the minimum-variance demos (node minvar-check.js).
global.window = {};
require('./recursion.js');
const S = window.SCHUR;
let fails = 0;
function check(name, ok) { console.log((ok ? 'ok   ' : 'FAIL ') + name); if (!ok) fails++; }
// issue #28: every enumerated entrywise state is positive definite up to the slider caps
check('six off-diagonals: all 64 states PD at tau = 0.16', S.Fent(0.5, 0.16, false).nIndef === 0);
check('all ten entries: all 1024 states PD at tau = 0.12', S.Fent(0.5, 0.12, true).nIndef === 0);
check('six off-diagonals: indefinite states appear at tau = 0.2 (2 of 64)', S.Fent(0.5, 0.2, false).nIndef === 2);
check('all ten entries: indefinite states appear at tau = 0.15 (10 of 1024)', S.Fent(0.5, 0.15, true).nIndef === 10);
check('Fent reports valid=false outside the domain', S.Fent(0.5, 0.2, false).valid === false && S.Fent(0.5, 0.16, false).valid === true);
console.log(fails ? `${fails} check(s) failed` : 'all min-var demo checks pass');
process.exit(fails ? 1 : 0);
