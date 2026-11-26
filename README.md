<h1 align="center">
  <picture>
    <source media="(max-width: 640px)" srcset="docs/assets/causilo-banner-compact.png">
    <img src="docs/assets/causilo-banner.png" alt="Causilo 1.5. A New Standard for Prediction." width="100%">
  </picture>
</h1>

<p align="center">Tabular classification and regression with a scikit-learn API.</p>

<p align="center">
  <a href="https://pypi.org/project/causilo/"><img src="https://img.shields.io/pypi/v/causilo?label=PyPI&amp;color=brightgreen&amp;style=flat" alt="Causilo package version on PyPI"></a>
  <a href="#documentation"><img src="https://img.shields.io/badge/Docs-Python-blue?style=flat" alt="Causilo Python documentation"></a>
  <a href="https://arxiv.org/abs/2609.22866"><img src="https://img.shields.io/badge/Technical%20report-arXiv-913f47?style=flat" alt="Causilo Technical Report on arXiv"></a>
  <br>
  <a href="#license"><img src="https://img.shields.io/badge/License-Code%20%2F%20weights-646b73?style=flat" alt="Code and model weights licenses"></a>
  <a href="https://nums.world/"><img src="https://img.shields.io/badge/Nums%20AI-Website-283442?style=flat" alt="Nums AI website"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/Python-3.10%E2%80%933.14-646b73?style=flat" alt="Python 3.10–3.14"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/PyTorch-2.13%2B-646b73?style=flat" alt="PyTorch 2.13 or newer"></a>
</p>

## Quick start

### Installation

Install [Causilo from PyPI](https://pypi.org/project/causilo/).

```sh
pip install causilo
```

If you already have Causilo, upgrade with `pip install --upgrade causilo`.

The first `fit` downloads the [public Causilo 1.5 checkpoint](https://huggingface.co/nums-ai/causilo-1.5) for the estimator's task. No Hugging Face account is required.

### Classification

```python
from causilo import CausiloClassifier

clf = CausiloClassifier()
clf.fit(X_train, y_train)
labels = clf.predict(X_test)
```

### Regression

```python
from causilo import CausiloRegressor

reg = CausiloRegressor()
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
```

**Causilo 1.5** uses the full training context by default. Set `mode="light"`
on either estimator for **Causilo 1.5 (Light)**. To sample from larger tables,
set `max_context=200_000`. Set `model_version="1.0"` to use **Causilo 1.0**.

`fit` prepares preprocessing and an inference context from your labeled table. `predict` uses that context with fixed model weights.

## Benchmarks

Overall rank and Elo using the Causilo 1.5.1 defaults.

| Benchmark | Causilo 1.5 | Causilo 1.5 (Light) |
| --- | ---: | ---: |
| [BeyondArena Core2K](https://huggingface.co/nums-ai/causilo-1.5/blob/ec07168e1df86ba4b008ab6c11fdc2056f1f8ffb/benchmarks/beyondarena/leaderboard.csv) | **#1 · 1,454.5** | #3 · 1,390.7 |
| [ScoringBench](https://huggingface.co/nums-ai/causilo-1.5/blob/ec07168e1df86ba4b008ab6c11fdc2056f1f8ffb/benchmarks/scoringbench/leaderboard.csv) | **#1 · 2,284.0** | #3 · 2,205.9 |
| [MulTaBench Image](https://huggingface.co/nums-ai/causilo-1.5/blob/ec07168e1df86ba4b008ab6c11fdc2056f1f8ffb/benchmarks/multabench-image/leaderboard.csv) | **#1 · 1,881.3** | #2 · 1,840.4 |
| [MulTaBench Text](https://huggingface.co/nums-ai/causilo-1.5/blob/ec07168e1df86ba4b008ab6c11fdc2056f1f8ffb/benchmarks/multabench-text/leaderboard.csv) | **#1 · 1,818.7** | #3 · 1,786.1 |
| [RamanBench](https://huggingface.co/nums-ai/causilo-1.5/blob/ec07168e1df86ba4b008ab6c11fdc2056f1f8ffb/benchmarks/ramanbench/leaderboard.csv) | **#1 · 1,569.7** | #2 · 1,540.5 |
| [STRABLE](https://huggingface.co/nums-ai/causilo-1.5/blob/ec07168e1df86ba4b008ab6c11fdc2056f1f8ffb/benchmarks/strable/leaderboard.csv) | **#1 · 1,876.1** | #2 · 1,825.5 |
| [TabBench-Bio](https://huggingface.co/nums-ai/causilo-1.5/blob/ec07168e1df86ba4b008ab6c11fdc2056f1f8ffb/benchmarks/tabbench-bio/leaderboard.csv) | **#1 · 1,251.0** | #3 · 1,229.0 |
| [TabBioMed](https://huggingface.co/nums-ai/causilo-1.5/blob/ec07168e1df86ba4b008ab6c11fdc2056f1f8ffb/benchmarks/tabbiomed/leaderboard-release.csv) | **#1 · 1,515.7** | #2 · 1,499.3 |
| [TALENT](https://huggingface.co/nums-ai/causilo-1.5/blob/ec07168e1df86ba4b008ab6c11fdc2056f1f8ffb/benchmarks/talent/leaderboard.csv) | **#1 · 1,979.0** | #3 · 1,931.3 |

[Detailed results and evaluation protocols](https://huggingface.co/nums-ai/causilo-1.5/blob/ec07168e1df86ba4b008ab6c11fdc2056f1f8ffb/benchmarks/README.md)

The TabArena evaluation for 1.5.1 is pending. [TabArena results, timing, and Pareto for 1.5.0](https://huggingface.co/nums-ai/causilo-1.5/tree/4ca2275bae85014a19422e8c18fcf8f24c5c14e1/benchmarks/tabarena) · [1.5.0 benchmark figure](https://github.com/nums-ai/causilo/blob/v1.5.0/docs/assets/benchmark-overview.png)

## Documentation

[Python inference](docs/inference.md) · [Checkpoints](docs/checkpoints.md) · [Examples](examples/)

## License

Code is licensed under [Apache-2.0](LICENSE).

Causilo 1.5 weights are licensed under the [Nums AI Model License](https://huggingface.co/nums-ai/causilo-1.5/blob/4ca2275bae85014a19422e8c18fcf8f24c5c14e1/LICENSE).

## Citation

If you use Causilo in research, please cite:

```bibtex
@misc{cho2026causilotechnicalreport,
  title={Causilo Technical Report},
  author={Minyong Cho and Minho Jeong and Dooho Lee and Jinmo Lee and Jaemin Yoo},
  year={2026},
  eprint={2609.22866},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2609.22866},
}
```
