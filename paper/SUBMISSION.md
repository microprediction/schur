# Submission plan: arXiv, then Mathematical Finance

## Paper

*When the Out-of-Sample-Optimal Schur Portfolio Lies Between HRP and
Minimum Variance* — `neither-end-of-the-bridge.tex` (self-contained,
builds with `tectonic` or any standard pdflatex; no custom class).
Certificates: `neither_end_certificate.py` (Python stdlib only; prints exact
rational verifications of the paper's exact claims about the example
covariances). Interactive demonstrations:
https://schur.microprediction.org/min-var-demos.html

## Step 1 — arXiv

- Primary category: `q-fin.PM` (Portfolio Management). Optional cross-list:
  `q-fin.MF` (Mathematical Finance).
- MSC classes: 91G10, 62H12 (already in the paper).
- Upload: the single `.tex` (self-contained, embedded bibliography). Include
  the certificate script as an ancillary file (`anc/neither_end_certificate.py`).
- Suggested comments field:
  "Exact machine-verifiable certificates (Python stdlib) included as
  ancillary material; interactive demonstrations at
  schur.microprediction.org."
- Sanity: compiles under plain pdflatex (packages: amsmath, amssymb, amsthm,
  booktabs, geometry, microtype, hyperref, lmodern — all in TeX Live).

## Step 2 — Mathematical Finance (Wiley)

- Editor-in-Chief: Rama Cont. Scope statement: "mathematically rigorous
  style … methodological novelty and contribution to financial modelling."
- Submission: Wiley's author portal (Research Exchange) from the journal
  page at onlinelibrary.wiley.com/journal/14679965. Free-format LaTeX/PDF is
  accepted at initial submission.
- Include: manuscript PDF, the certificate script as supplementary material,
  and the cover letter below. Prior arXiv posting is compatible with Wiley's
  preprint policy; cite the arXiv number in the submission form once live.

## Cover letter draft

Dear Professor Cont,

Please consider the enclosed manuscript, "When the Out-of-Sample-Optimal Schur Portfolio Lies Between HRP and
Minimum Variance," for
publication in Mathematical Finance.

Schur-complementary allocation interpolates hierarchical risk parity and
minimum variance through a single coupling parameter, and has been adopted
in practice (it ships in the skfolio library). The manuscript asks where on
that bridge to sit when the covariance is estimated with error, and answers
in four layers: a structural theorem showing cross-block noise is invisible
at the hierarchical end and prices coupling at order gamma tau^2; a local
theorem at an exact minimum-variance endpoint showing positive marginal
noise sensitivity moves the optimum into the interior; an exchangeable
four-asset family solved exactly, whose unique optimizer is interior at
every noise level the theorem covers and decreases monotonically from full
coupling to the middle of the bridge; and a whole-matrix layer that drops
the cross-block restriction, with a sharp second-order criterion for
interiority, exact enumeration certificates under entrywise noise, and a
sample-covariance study in which the median-optimal coupling sweeps the
bridge with sample size. Exact examples mark the boundaries of what
holds in general, including a family in which hierarchical risk parity
becomes locally optimal at a finite noise level.

The paper's exact claims about the example covariances are verified in
rational arithmetic by a short, dependency-free Python script included as
supplementary material, and each theorem has an interactive demonstration at
schur.microprediction.org/min-var-demos.html.

The manuscript is not under consideration elsewhere. A preprint is posted on
arXiv as [ID].

Sincerely,
Peter Cotton

## Checklist before sending

- [ ] arXiv posted; ID inserted in cover letter and submission form.
- [ ] Certificate script runs on a clean Python 3 (stdlib only): `python3 neither_end_certificate.py`.
- [ ] PDF builds from the submitted tex byte-identically.
- [ ] min-var-demos.html live and linked from the paper's closing.
