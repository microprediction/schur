# schur

Schur complementary portfolios — a unification of hierarchical and optimisation-based portfolio
construction via block-matrix inversion (Schur complement).

**Paper:** [arXiv:2411.05807](https://arxiv.org/abs/2411.05807)
**Site:** [microprediction.github.io/schur/](https://microprediction.github.io/schur/) *(after Pages is enabled)*
**Recommended implementation:** [SkFolio tutorial](https://skfolio.org/auto_examples/clustering/plot_6_schur.html)
**Original implementation:** [`precise`](https://github.com/microprediction/precise) (`schurmanagers` module)
**Textbook coverage:** Palomar, [*Portfolio Optimization* §12.3.4](https://portfoliooptimizationbook.com/book/12.3-hierarchical-clustering-based-portfolios.html#HRP-vs-GMVP)

## Cite

```bibtex
@article{cotton2024schur,
  author  = {Cotton, Peter},
  title   = {Schur Complementary Allocation: A Unification of Hierarchical
             Risk Parity and Minimum Variance Portfolios},
  journal = {arXiv preprint arXiv:2411.05807},
  year    = {2024},
  url     = {https://arxiv.org/abs/2411.05807}
}
```

## Enabling the Pages site

Settings → Pages → Source: **GitHub Actions**. The workflow in `.github/workflows/pages.yml`
uploads `docs/` directly (no Jekyll). After the first push to `main` the workflow runs
automatically and the site is live within a minute.
