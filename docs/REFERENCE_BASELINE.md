# Pinned author-reference comparison

`examples/compare_ppi_reference.py` runs the same numeric arrays through
`judgecal` and the authors' `ppi-python==0.2.3` scalar-mean APIs. This is a
numerical implementation comparison, not an independent demonstration of
coverage, accuracy, label savings, or research novelty.

The committed results are in the [reference comparison summary](../reports/reference/SUMMARY.md).

## Provenance and inspection

The baseline is the published [PyPI release 0.2.3](https://pypi.org/project/ppi-python/0.2.3/),
uploaded December 22, 2024 by the project maintained by Anastasios
Angelopoulos. It is tied to the [authors' PPI repository](https://github.com/aangelopoulos/ppi_py)
and [PPI++ paper](https://arxiv.org/abs/2311.01453). The release is pinned,
rather than a moving `main` checkout. The wheel, included license, scalar-mean
functions, tuning function, and import dependencies were inspected before
reference execution. No substantial reference code is vendored.

| Artifact | SHA-256 |
|---|---|
| `ppi_python-0.2.3-py3-none-any.whl` | `f20ea31f90af3339de0f4a48f556c281493e347f49a9fea3d0e164769fdf8b24` |
| Wheel's `ppi_py/ppi.py` | `3c329fb57ce2e73f2cb6e8696188a3415eabcb9f3720128f350378b369693a9c` |

The wheel's MIT license names Anastasios Angelopoulos (2023). Installation
retains the package's license. The script checks distribution version and
the inspected `ppi.py` hash before importing it, checks the imported source
location and API signatures, and records hashes of all installed package
Python files. The published wheel hash verifies the distribution artifact;
the script's source check alone is not a re-verification of the entire wheel.

Declared dependencies are NumPy, Numba, pandas, statsmodels, scikit-learn,
and gdown. SciPy is also imported and supplied through dependencies. There
are no model weights or GPU frameworks in this dependency list. Dataset
download helpers are not called. The optional script neither installs nor
downloads anything and requires no credentials.

## Isolated installation and run

Use a separate virtual environment. On this Windows project, nesting it
under `.venv/reference-env` keeps it within the existing ignored `.venv`
directory. Use `python -m venv .venv/reference-env`, then its Python for all
commands below. Equivalent commands work on other platforms with that
environment's Python path.

```text
python -m pip install "ppi-python @ https://files.pythonhosted.org/packages/8a/42/754bd7edf25e1cfa8b47001d94d71a4966a0f23e98727029ba4c58123127/ppi_python-0.2.3-py3-none-any.whl#sha256=f20ea31f90af3339de0f4a48f556c281493e347f49a9fea3d0e164769fdf8b24"
python -m pip install -e ".[dev]"
python examples/compare_ppi_reference.py --output reports/reference
python -m pytest tests/test_reference_contract.py -q
```

Only the reference distribution is pinned in the installation command;
resolved dependency versions are recorded in `environment.json`. The
script fails if a different reference version or source hash is installed.
Default tests require no reference installation: ordinary offline fixture
checks run, while the two reference integration tests explicitly skip when
the distribution is absent. An installed but wrong/broken reference fails
instead of being silently skipped.

## Exact public APIs used

```python
from ppi_py import ppi_mean_pointestimate, ppi_mean_ci

point = ppi_mean_pointestimate(Y, Yhat, Yhat_unlabeled, lam=lam)
low, high = ppi_mean_ci(
    Y, Yhat, Yhat_unlabeled, alpha=0.05,
    alternative="two-sided", lam=lam,
)
```

Arrays are numeric one-dimensional scores in `{0, 0.5, 1}`. `lam=0`, `0.25`,
and `1` test fixed coefficients. `lam=None` invokes the reference's automatic
tuning; the corresponding judgecal argument is `power="auto"`. The reference
does not return its coefficient through these public functions. The report
therefore labels it as a value computed from the pinned source formula,
validated through the actual public point and interval calls.

No cluster identifiers are passed: this checks the common iid scalar-mean
scope. It does not validate cluster adaptation against the author package.
Six retrospective fixtures include strong, weak, anticorrelated judges,
a small prediction-only pool, ties, and a perfect-judge algebra fixture.
Exact seeded arrays are exported. These are numerical examples, not
independent empirical replications of a statistical experiment.

## Expected matches and expected differences

For a fixed coefficient, both points equal
`mean(Y) + lam * (mean(F_U) - mean(F_L))`, up to floating-point arithmetic.
Their normal intervals can differ. Version 0.2.3 uses NumPy's `ddof=0`
variance for each residual/prediction interval term, while judgecal uses
`ddof=1`. The comparison verifies each convention separately rather than
forcing false interval equality.

For automatic tuning, let `n` and `N` be audit and prediction-only counts.
The inspected unweighted scalar reference computes

```text
reference_power = clip(
    mean((Y - mean(Y)) * (F_L - mean(F_L)))
    / [(1 + n/N) * var(concat(F_L, F_U), ddof=1)], 0, 1
).
```

Judgecal instead uses audit covariance with `ddof=1` divided by
`var(F_L, ddof=1) + (n/N) * var(F_U, ddof=1)`, clipped to the same interval.
Both point and interval differences can therefore be expected on identical
data. Neither convention is declared statistically superior here. The
script checks each implementation against its own explicit scalar algebra;
it retains all cross-implementation differences.

Constant pooled predictions make the reference's automatic denominator
zero. Those degenerate fixtures are outside this numerical contract; the
script rejects them explicitly instead of patching the reference or hiding
undefined results. Judgecal's zero-denominator fallback is tested separately.

## Evidence artifacts

- `cases.json`: all numeric arrays, seed, configuration, pinned wheel/source
  metadata, powers, scope, and tolerance.
- `comparisons.csv`: both returned points/intervals, powers, moments,
  differences, and individual convention-check outcomes for every case.
- `environment.json`: Python/platform, every installed distribution version,
  source fingerprints, and exact inspected API signatures.
- `summary.json` and `SUMMARY.md`: comparison counts, contract outcome, and
  maximum observed fixed/automatic differences.
- `artifact_hashes.json`: raw-byte SHA-256 hashes of the five evidence files.
  Generated text artifacts use UTF-8 LF on all platforms. Repository-source
  fingerprints normalize CRLF to LF; inspected reference files retain
  raw-byte hashes. The normalization is recorded explicitly.

The command exits nonzero if a numerical contract check fails. Successful
checks establish agreement with the specified conventions on these fixtures;
they do not establish byte-for-byte equivalence or statistical guarantees.
