# Submission plan: arXiv, then Mathematical Finance

## Paper

*Betwixt Minimum Variance and Hierarchical Risk Parity: Analytical Results
for the Schur Bridge* — `neither-end-of-the-bridge.tex` (self-contained,
builds with `tectonic` or any standard pdflatex; no custom class).
Certificates: `neither_end_certificate.py` (Python stdlib only; prints exact
rational verifications of every exact claim). Interactive theorem checks:
https://schur.microprediction.org/theorems.html

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

Please consider the enclosed manuscript, "Betwixt Minimum Variance and
Hierarchical Risk Parity: Analytical Results for the Schur Bridge," for
publication in Mathematical Finance.

Schur-complementary allocation interpolates hierarchical risk parity and
minimum variance through a single coupling parameter, and has been adopted
in practice (it ships in the skfolio library). The manuscript asks where on
that bridge to sit when the covariance is estimated with error, and answers
in three layers: a structural theorem showing cross-block noise is invisible
at the hierarchical end and prices coupling at order gamma tau^2; a local
theorem at an exact minimum-variance endpoint showing positive marginal
noise sensitivity moves the optimum into the interior; and an exchangeable
four-asset family solved exactly, whose unique optimizer is interior at
every admissible noise level and decreases monotonically from full coupling
to the middle of the bridge. Exact examples mark the boundaries of what
holds in general, including a family in which hierarchical risk parity
becomes locally optimal at a finite noise level.

Every exact claim in the paper is verified in exact rational arithmetic by a
short, dependency-free Python script included as supplementary material, and
each theorem has an interactive demonstration at
schur.microprediction.org/theorems.html.

The manuscript is not under consideration elsewhere. A preprint is posted on
arXiv as [ID].

Sincerely,
Peter Cotton

## Checklist before sending

- [ ] arXiv posted; ID inserted in cover letter and submission form.
- [ ] Certificate script runs on a clean Python 3 (stdlib only): `python3 neither_end_certificate.py`.
- [ ] PDF builds from the submitted tex byte-identically.
- [ ] theorems.html live and linked from the paper's footnote.
