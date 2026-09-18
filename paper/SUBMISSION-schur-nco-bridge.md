# Submission notes: the NCO bridge paper

*Nested Clustered Optimization Is One End of a Schur Bridge, and the
Interior Is Sometimes Provably Better* — `schur-nco-bridge.tex`,
self-contained (embedded bibliography, no figures), compiles under plain
pdflatex with zero warnings. Verification: `verify_schur_nco_bridge.py`
(NumPy plus the standard library; 33 checks, exact rational arithmetic for
the examples). Demonstrations: https://schur.microprediction.org/nco-demos.html

Already posted: SSRN 7480738, CC BY,
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7480738

## arXiv

- Pack: `schur-nco-bridge_arxiv.tar.gz` (the `.tex` and
  `anc/verify_schur_nco_bridge.py`). arXiv treats `anc/` as ancillary files.
- Primary category: `q-fin.PM` (Portfolio Management).
- Cross-lists: `stat.ME` (the Vecchia conditioning-set connection and the
  estimation-error analysis); `math.OC` (the bridge as a relaxation of the
  minimum-variance program).
- MSC classes: 91G10 (portfolio theory), 62H12 (multivariate estimation).
- License: CC BY 4.0, to match SSRN.
- Comments field: "13 pages. Verification script (NumPy, exact rational
  arithmetic for the examples) as ancillary material. Interactive
  demonstrations at schur.microprediction.org/nco-demos.html. Also SSRN 7480738."
- Abstract: paste the paragraph below.

Nested clustered optimization allocates within each cluster from the
cluster's own covariance block and then across the resulting cluster
portfolios. Block inversion says the unconstrained minimum-variance
portfolio has the same two-tier shape, with each block replaced by its Schur
complement against every other asset. Conditioning instead on one knot from
each other cluster truncates the conditioning set in the manner of a
Vecchia approximation, and we give the rank-one model of cross-cluster
dependence under which it is exact. Damping the complement by
$\gamma\in[0,1]$ then gives a bridge with nested clustered optimization at
$\gamma=0$ and the global optimum at $\gamma=1$, with no linear solve larger
than a cluster or the number of clusters. Under estimation error the optimal
$\gamma$ can be strictly interior and full coupling can remain optimal, and
we give the local theorem at the minimum-variance end with exact examples
of both, including a symmetric family in which the optimum is a closed form.

## Sanity checklist before upload

- `tar tzf schur-nco-bridge_arxiv.tar.gz` lists exactly the two files.
- `pdflatex schur-nco-bridge.tex` three times from the unpacked directory:
  no warnings, 13 pages.
- The `\date{}` in the file is the literal date of writing, not `\today`.
