import pathlib, re, sys
D = pathlib.Path("docs")
log = []
def rep(fn, old, new, n=1):
    p = D / fn; s = p.read_text()
    c = s.count(old)
    assert c == n, f"{fn}: expected {n} got {c} for {old[:60]!r}"
    p.write_text(s.replace(old, new)); log.append((fn, old, new))

# ---------- shared footer line on ten demo pages ----------
for fn in ["demo-blindness.html","demo-entrywise.html","demo-family.html","demo-frontier.html",
           "demo-hrp-endpoint.html","demo-mv-endpoint.html","demo-noise-cost.html",
           "demo-one-entry.html","demo-sample-covariance.html","demo-whole-matrix.html"]:
    rep(fn, "in your browser. Nothing is pre-baked.</p>", "in your browser. Nothing is precomputed.</p>")

# ---------- h1 em-dashes -> colons ----------
rep("demo-blindness.html", "<h1>Lemma 1 — HRP never sees the cross-block</h1>",
    "<h1>Lemma 1: HRP never sees the cross-block</h1>")
rep("demo-entrywise.html", "<h1>Entrywise noise — enumerate every sign state</h1>",
    "<h1>Entrywise noise: enumerate every sign state</h1>")
rep("demo-family.html", "<h1>Theorem 4 — the solved family: unique interior optimum, monotone in the noise</h1>",
    "<h1>Theorem 4: the solved family has a unique interior optimum, monotone in the noise</h1>")
rep("demo-frontier.html", "<h1>Theorem 6 — the clipped frontier</h1>", "<h1>Theorem 6: the clipped frontier</h1>")
rep("demo-hrp-endpoint.html", "<h1>Example 1 — HRP becomes locally optimal at finite noise</h1>",
    "<h1>Example 1: HRP becomes locally optimal at finite noise</h1>")
rep("demo-mv-endpoint.html", "<h1>Theorem 3 — small noise pushes the optimum off the minimum-variance end</h1>",
    "<h1>Theorem 3: small noise pushes the optimum off the minimum-variance end</h1>")
rep("demo-noise-cost.html", "<h1>Theorems 1 and 2 — the noise cost is &gamma;&tau;&sup2;, not &gamma;&sup2;&tau;&sup2;</h1>",
    "<h1>Theorems 1 and 2: the noise cost is &gamma;&tau;&sup2;, not &gamma;&sup2;&tau;&sup2;</h1>")
rep("demo-one-entry.html", "<h1>Example 2 — one-entry noise: interior without symmetry</h1>",
    "<h1>Example 2: one-entry noise, interior without symmetry</h1>")
rep("demo-sample-covariance.html", "<h1>The sample covariance — &gamma;* sweeps the bridge with T</h1>",
    "<h1>The sample covariance: &gamma;* sweeps the bridge with T</h1>")
rep("demo-whole-matrix.html", "<h1>Lemma 4 and Theorem 5 — whole-matrix noise: the sign of &Xi; decides</h1>",
    "<h1>Lemma 4 and Theorem 5: under whole-matrix noise the sign of &Xi; decides</h1>")

# ---------- falsification sentences in callouts ----------
rep("demo-blindness.html",
    "&gamma; = 0.6 bars drift with the noise.</div>",
    "&gamma; = 0.6 bars drift with the noise. Any movement in the &gamma; = 0 bars would refute the lemma.</div>")
rep("demo-entrywise.html",
    "What to look for: F(0) is no longer flat in &tau; (compare\n  Lemma 1: the diagonal channels charge HRP too), yet the minimum stays strictly\n  inside.",
    "What to look for: F(0) is not flat in &tau; here, unlike Lemma 1, because\n  the diagonal channels charge HRP too. The minimum still stays strictly inside;\n  a minimum at either end would refute the claim.")
rep("demo-family.html",
    "1 &minus; (1025/153)e&sup2; overlaid near e = 0.</div>",
    "1 &minus; (1025/153)e&sup2; overlaid near e = 0. A visible gap between the two\n  curves, or a minimizer that moved right as e grew, would refute the theorem.</div>")
rep("demo-frontier.html",
    "&asymp; 0.034, and its optimizer path drops to zero there.</div>",
    "&asymp; 0.034, and its optimizer path drops to zero there. A Family B optimizer\n  that stayed positive past s = 27/800 would refute the clipping.</div>")
rep("demo-hrp-endpoint.html",
    "parabola; the zero crossing is at &tau;* &asymp; 0.1837.</div>",
    "parabola; the zero crossing is at &tau;* &asymp; 0.1837. Dots off the parabola,\n  or a crossing elsewhere, would refute the formula.</div>")
rep("demo-mv-endpoint.html",
    "the line at larger e is the O(e&#8308;) term.</div>",
    "the line at larger e is the O(e&#8308;) term. Dots approaching a different slope\n  as e &darr; 0 would refute the coefficient.</div>")
rep("demo-noise-cost.html",
    "&gamma;&sup2;&tau;&sup2;, Family B diverges near zero; Family A does not\n  (the cancellation).</div>",
    "&gamma;&sup2;&tau;&sup2;, Family B diverges near zero; Family A does not\n  (the cancellation). A Family B curve that diverged under &gamma;&tau;&sup2; would\n  refute Theorem 2.</div>")
rep("demo-one-entry.html",
    "(Lemma 1 again: the noised entry is in the cross-block), while the right end\n  rises with the noise.</div>",
    "(Lemma 1 again: the noised entry is in the cross-block), while the right end\n  rises with the noise. A minimizer at 0 or 1 at &tau; = 0.15 would contradict the\n  certificate.</div>")
rep("demo-sample-covariance.html",
    "&gamma; &asymp; 0.96 and the tail is gone. The readout collects\n  &gamma;*(T) as you click through the sample sizes.</div>",
    "&gamma; &asymp; 0.96 and the tail is gone. The readout collects\n  &gamma;*(T) across the sample sizes. A median minimum that stayed put as T grew\n  would refute the sweep.</div>")
rep("demo-whole-matrix.html",
    "negative and the minimizer parks at &gamma; = 1 exactly, at every\n  amplitude.</div>",
    "negative and the minimizer parks at &gamma; = 1 exactly, at every\n  amplitude. An interior minimizer under &delta;-only noise would refute Theorem 5(ii).</div>")

# ---------- demo-family: self-grading + meandering sentence ----------
rep("demo-family.html",
    "&gamma;*(0.6) = &frac12; exactly. One honest footnote: for e &ge; 0.15 the raw\n  recursion is undefined at the single point &gamma; = 0.75/(0.6 + e), where a\n  block's augmented fitness diverges; the recursion is evaluated on a grid that avoids that one point, and the dashed closed form is the continuous extension\n  (equation (18) of the paper), which the recursion equals everywhere else.\n  Theorem 4(v)",
    "&gamma;*(0.6) = &frac12; exactly.</p>\n  <p>For e &ge; 0.15 the raw recursion is undefined at the single point\n  &gamma; = 0.75/(0.6 + e), where a block's augmented fitness diverges. The\n  recursion is evaluated on a grid that avoids that point, and the dashed closed\n  form is the continuous extension (equation (18) of the paper), which the\n  recursion equals everywhere else.\n  Theorem 4(v)")

# ---------- demo-noise-cost: split family definitions into their own paragraph ----------
rep("demo-noise-cost.html",
    "general law. Family A: volatilities (1,1,2,2),",
    "general law.</p>\n  <p>Family A: volatilities (1,1,2,2),")

# ---------- demo-sample-covariance: self-grading, ordinal scaffold, split ----------
rep("demo-sample-covariance.html",
    "<p>Two honest wrinkles, both findings of the paper rather than bugs. First, on\n  a sample covariance",
    "<p>Two wrinkles, both findings of the paper. On a sample covariance")
rep("demo-sample-covariance.html",
    "as a separate benchmark. Second, the <em>mean</em> of the raw recursion's\n  loss appears to diverge at small T: rare draws drive the fitness sum through zero, the\n  pole looks non-integrable, and a sample mean never settles; the paper gives a pole criterion and Monte Carlo evidence but stops short of a theorem. The plot therefore",
    "as a separate benchmark.</p>\n  <p>The <em>mean</em> of the raw recursion's loss appears to diverge at small T:\n  rare draws drive the fitness sum through zero, the pole looks non-integrable,\n  and a sample mean never settles. The paper gives a pole criterion and Monte\n  Carlo evidence but stops short of a theorem. The plot therefore")

# ---------- hub page card titles ----------
for old, new in [
    ("Lemma 1 — HRP never sees the\n      cross-block", "Lemma 1: HRP never sees the\n      cross-block"),
    ("Theorems 1 and 2 — the noise cost is\n      &gamma;&tau;&sup2;", "Theorems 1 and 2: the noise cost is\n      &gamma;&tau;&sup2;"),
    ("Example 1 — HRP becomes locally\n      optimal at finite noise", "Example 1: HRP becomes locally\n      optimal at finite noise"),
    ("Theorem 3 — small noise moves the\n      optimum off the minimum-variance end", "Theorem 3: small noise moves the\n      optimum off the minimum-variance end"),
    ("Theorem 4 — unique interior optimum,\n      monotone in the noise", "Theorem 4: unique interior optimum,\n      monotone in the noise"),
    ("Example 2 — one-entry noise, interior\n      without symmetry", "Example 2: one-entry noise, interior\n      without symmetry"),
    ("Theorem 5 — the sign of &Xi;\n      decides", "Theorem 5: the sign of &Xi;\n      decides"),
    ("Entrywise noise — enumerate every sign\n      state", "Entrywise noise: enumerate every sign\n      state"),
    ("The sample covariance —\n      &gamma;* sweeps the bridge with T", "The sample covariance:\n      &gamma;* sweeps the bridge with T"),
    ("Theorem 6 — the clipped frontier</a>", "Theorem 6: the clipped frontier</a>"),
]:
    rep("min-var-demos.html", old, new)

# ---------- bridge-demo ----------
B = "bridge-demo.html"
rep(B, "halves by inverse variance, and recurses — never inverting a matrix, but also never using the\n  covariance <em>between</em> the halves.",
       "halves by inverse variance, and recurses. It never inverts a matrix, and it never uses the\n  covariance <em>between</em> the halves.")
rep(B, "order comes from <em>Fiedler seriation</em> — the second eigenvector of the similarity graph's\n  Laplacian — which moves continuously",
       "order comes from <em>Fiedler seriation</em>, the second eigenvector of the similarity graph's\n  Laplacian, which moves continuously")
rep(B, "<h2>Try it: drag γ</h2>", "<h2>The bridge on 28 Dow names</h2>")
rep(B, "<p>Two things to notice. The weights deform <em>continuously</em> in γ — there is no point at\n  which the portfolio switches regime. And the slider quietly stops helping: past",
       "<p>The weights deform <em>continuously</em> in γ: there is no point at\n  which the portfolio switches regime. The slider also stops helping: past")
rep(B, "to 1.34× at the cap — a modest gain on this", "to 1.34× at the cap, a modest gain on this")
rep(B, "<p>In sample, variance falls as γ rises until the cap — more of the covariance is used.",
       "<p>In sample, variance falls as γ rises until the cap, since more of the covariance is used.")
rep(B, "mid-bridge, at γ ≈ 0.55, about 2.5% below HRP — a modest dip, but a real interior optimum:\n  the coupling parameter is a bias–variance dial, not a formality. The γ slider above moves the\n  marker here too.</p>",
       "mid-bridge, at γ ≈ 0.55, about 2.5% below HRP: a modest dip, but a real interior optimum.\n  The coupling parameter is a bias–variance dial, not a formality. A curve that fell\n  monotonically to γ = 1, or bottomed at γ = 0, would refute the interior optimum on this\n  data. The γ slider above moves the marker here too.</p>")
rep(B, "Ledoit–Wolf and the interior optimum survives — slightly deeper (0.97× at γ = 0.6) and γ = 1\n  still pays 24%.",
       "Ledoit–Wolf and the interior optimum survives, slightly deeper (0.97× at γ = 0.6), and γ = 1\n  still pays 24%.")
rep(B, "<p>Nor is the interior optimum an accident of this dataset — it is provable.",
       "<p>The interior optimum is provable, not an accident of this dataset.")
rep(B, "while γ = 0 is\n  structurally noise-immune — the HRP recursion never reads the cross block, so the minimizer",
       "while γ = 0 is\n  structurally noise-immune (the HRP recursion never reads the cross block), so the minimizer")
rep(B, "The only difference is the ordering fed to the\n  recursion — a dendrogram recomputed each week,",
       "The only difference is the ordering fed to the\n  recursion: a dendrogram recomputed each week,")
rep(B, "covariance — the extra trades are pure reordering, not information. The gap is the point of\n  replacing the combinatorial step with a spectral one.",
       "covariance; the extra trades are pure reordering, not information. Replacing the\n  combinatorial step with a spectral one saves that gap.")
rep(B, "can. What the spectral step buys is fewer such events, a coordinate crossing rather than a\n  linkage reordering, and the lower turnover above is an empirical result, not a continuity\n  theorem.</p>",
       "can. The spectral step buys fewer such events, a coordinate crossing rather than a\n  linkage reordering. The lower turnover above is an empirical result, not a continuity\n  theorem.</p>")
rep(B, "The analytical companion — <em>When the Out-of-Sample-Optimal Schur Portfolio Lies Between HRP and Minimum Variance</em> — is <a",
       "The analytical companion, <em>When the Out-of-Sample-Optimal Schur Portfolio Lies Between HRP and Minimum Variance</em>, is <a")
rep(B, "schur.microprediction.org</a> — the Schur bridge.", "schur.microprediction.org</a>, the Schur bridge.")

print(f"{len(log)} replacements applied")
