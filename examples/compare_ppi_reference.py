"""Optional, offline numerical comparison with pinned author ppi-python 0.2.3.

Install the reference into an isolated environment before running this file.
This script never installs dependencies, downloads data, or calls model APIs.
The comparison validates conventions; it is not an accuracy/coverage study.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
from numbers import Integral
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.special import ndtri

from judgecal import prediction_powered_win_rate


REFERENCE_VERSION = "0.2.3"
REFERENCE_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/8a/42/"
    "754bd7edf25e1cfa8b47001d94d71a4966a0f23e98727029ba4c58123127/"
    "ppi_python-0.2.3-py3-none-any.whl"
)
REFERENCE_WHEEL_SHA256 = "f20ea31f90af3339de0f4a48f556c281493e347f49a9fea3d0e164769fdf8b24"
REFERENCE_PPI_SOURCE_SHA256 = "3c329fb57ce2e73f2cb6e8696188a3415eabcb9f3720128f350378b369693a9c"
FIXED_POWERS = (0.0, 0.25, 1.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha256(path: Path) -> str:
    """Normalize CRLF to LF for repository source evidence across checkouts."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 LF bytes consistently on Windows and POSIX."""
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def _scalar(value) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1 or not np.isfinite(array).all():
        raise ValueError("expected one finite scalar from reference API")
    return float(array.reshape(-1)[0])


def _labels(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) < 2 or not np.isin(scores, [0., .5, 1.]).all():
        raise ValueError("scores must be a one-dimensional array of 0, 0.5, 1 with length >= 2")
    return np.where(scores == 1, "A", np.where(scores == 0, "B", "tie"))


def build_seeded_cases(seed: int = 2026) -> list[dict]:
    """Retrospective numerical fixtures; exact generated arrays are exported."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    specs = (
        ("iid_strong", 200, 2000, .6, .95, .9),
        ("iid_weak", 80, 500, .6, .6, .5),
        ("iid_anticorrelated", 100, 300, .6, .15, .1),
        ("small_prediction_pool", 120, 30, .6, .95, .9),
    )
    cases = []
    for name, n, count_u, prevalence, sensitivity, specificity in specs:
        y = rng.binomial(1, prevalence, size=n).astype(float)
        f_l = rng.binomial(1, np.where(y == 1, sensitivity, 1 - specificity)).astype(float)
        y_u = rng.binomial(1, prevalence, size=count_u)
        f_u = rng.binomial(1, np.where(y_u == 1, sensitivity, 1 - specificity)).astype(float)
        cases.append({
            "case": name,
            "generator": {"type": "binary_iid", "prevalence": prevalence,
                          "sensitivity": sensitivity, "specificity": specificity},
            "human_labeled": y.tolist(), "judge_labeled": f_l.tolist(),
            "judge_unlabeled": f_u.tolist(),
        })
    y = np.array([1., .5, 0., .5, 1., 0., .5, 1.])
    f_l = np.array([1., 1., 0., .5, 1., 0., 0., .5])
    permutation = rng.permutation(len(y))
    cases.append({
        "case": "small_tie_fixture",
        "generator": {"type": "fixed_algebra_fixture_with_seeded_row_order"},
        "human_labeled": y[permutation].tolist(),
        "judge_labeled": f_l[permutation].tolist(),
        "judge_unlabeled": rng.permutation([1., 1., .5, 1., 0., .5, 1., 1., .5]).tolist(),
    })
    cases.append({
        "case": "perfect_judge_fixture",
        "generator": {"type": "fixed_algebra_fixture_with_seeded_row_order"},
        "human_labeled": np.tile([0., 1.], 10).tolist(),
        "judge_labeled": np.tile([0., 1.], 10).tolist(),
        "judge_unlabeled": rng.permutation(np.tile([0., 1.], 20)).tolist(),
    })
    return cases


def scalar_moments(y: np.ndarray, f_l: np.ndarray, f_u: np.ndarray) -> dict:
    """Independent scalar algebra for both implementations' conventions."""
    y, f_l, f_u = (np.asarray(values, dtype=float) for values in (y, f_l, f_u))
    for values in (y, f_l, f_u):
        _labels(values)
    if len(y) != len(f_l):
        raise ValueError("audit human and judge arrays must align")
    n, count_u = len(y), len(f_u)
    centered_product = float(np.dot(y - y.mean(), f_l - f_l.mean()))
    covariance_ddof0 = centered_product / n
    covariance_ddof1 = centered_product / (n - 1)
    per_pool_denominator = float(np.var(f_l, ddof=1) + n / count_u * np.var(f_u, ddof=1))
    pooled_variance = float(np.var(np.concatenate([f_l, f_u]), ddof=1))
    reference_denominator = (1 + n / count_u) * pooled_variance
    # Reference auto behavior with a zero denominator is outside this
    # numerical contract; do not silently patch the authors' implementation.
    reference_power = (float(np.clip(covariance_ddof0 / reference_denominator, 0, 1))
                       if reference_denominator > 0 else None)
    return {
        "n_labeled": n, "n_unlabeled": count_u,
        "covariance_ddof0": covariance_ddof0, "covariance_ddof1": covariance_ddof1,
        "per_pool_denominator": per_pool_denominator,
        "pooled_judge_variance_ddof1": pooled_variance,
        "reference_denominator": reference_denominator,
        "judgecal_auto_power_formula": float(np.clip(covariance_ddof1 / per_pool_denominator, 0, 1))
            if per_pool_denominator > 0 else 0.,
        "reference_auto_power_formula": reference_power,
    }


def load_reference():
    """Validate version and inspected source before importing reference code."""
    try:
        distribution = metadata.distribution("ppi-python")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("Optional baseline missing: install ppi-python==0.2.3 in an isolated environment.") from exc
    if distribution.version != REFERENCE_VERSION:
        raise RuntimeError(f"Expected ppi-python=={REFERENCE_VERSION}, found {distribution.version}")
    source = Path(distribution.locate_file("ppi_py/ppi.py")).resolve()
    if not source.is_file() or _sha256(source) != REFERENCE_PPI_SOURCE_SHA256:
        raise RuntimeError("Reference ppi.py does not match the inspected 0.2.3 wheel source hash")
    # The reference source sets a global warnings filter on import. Restore
    # caller filters so comparison and test warnings are not suppressed.
    with warnings.catch_warnings():
        reference = importlib.import_module("ppi_py")
    imported_source = Path(inspect.getsourcefile(reference.ppi_mean_pointestimate)).resolve()
    if imported_source != source:
        raise RuntimeError("Imported reference module is shadowed by a different source location")
    signatures = {
        name: str(inspect.signature(getattr(reference, name)))
        for name in ("ppi_mean_pointestimate", "ppi_mean_ci")
    }
    expected_point = ["Y", "Yhat", "Yhat_unlabeled", "lam", "coord", "w", "w_unlabeled", "lam_optim_mode"]
    expected_ci = ["Y", "Yhat", "Yhat_unlabeled", "alpha", "alternative", "lam", "coord", "w", "w_unlabeled", "lam_optim_mode"]
    if (list(inspect.signature(reference.ppi_mean_pointestimate).parameters) != expected_point
            or list(inspect.signature(reference.ppi_mean_ci).parameters) != expected_ci):
        raise RuntimeError("Reference scalar-mean API contract differs from the inspected release")
    return reference, {"version": distribution.version, "source_sha256": _sha256(source),
                       "api_signatures": signatures}


def compare_cases(cases: list[dict], reference, alpha: float = .05) -> pd.DataFrame:
    if isinstance(alpha, (bool, np.bool_)) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    quantile = float(ndtri(1 - alpha / 2))
    records = []
    for case in cases:
        y, f_l, f_u = (np.asarray(case[key], dtype=float) for key in
                       ("human_labeled", "judge_labeled", "judge_unlabeled"))
        moments = scalar_moments(y, f_l, f_u)
        if moments["reference_auto_power_formula"] is None:
            raise ValueError("Reference automatic coefficient is undefined for a constant pooled judge fixture")
        for power in (*FIXED_POWERS, "auto"):
            reference_argument = None if power == "auto" else power
            ref_power = moments["reference_auto_power_formula"] if power == "auto" else power
            expected_ours_power = moments["judgecal_auto_power_formula"] if power == "auto" else power
            ours = prediction_powered_win_rate(_labels(f_l), _labels(y), _labels(f_u), power=power, alpha=alpha)
            reference_point = _scalar(reference.ppi_mean_pointestimate(y, f_l, f_u, lam=reference_argument))
            reference_ci = reference.ppi_mean_ci(y, f_l, f_u, alpha=alpha, alternative="two-sided", lam=reference_argument)
            reference_low, reference_high = (_scalar(endpoint) for endpoint in reference_ci)
            reference_se = (reference_high - reference_low) / (2 * quantile)
            expected_reference_point = float(y.mean() + ref_power * (f_u.mean() - f_l.mean()))
            expected_ours_point = float(y.mean() + expected_ours_power * (f_u.mean() - f_l.mean()))
            expected_reference_variance = float(np.var(y - ref_power * f_l, ddof=0) / len(y)
                                                + ref_power ** 2 * np.var(f_u, ddof=0) / len(f_u))
            expected_ours_variance = float(np.var(y - expected_ours_power * f_l, ddof=1) / len(y)
                                           + expected_ours_power ** 2 * np.var(f_u, ddof=1) / len(f_u))
            checks = {
                "judgecal_power_matches_formula": np.isclose(ours.selected_power, expected_ours_power, atol=1e-12, rtol=1e-10),
                "judgecal_point_matches_formula": np.isclose(ours.point, expected_ours_point, atol=1e-12, rtol=1e-10),
                "reference_point_matches_formula": np.isclose(reference_point, expected_reference_point, atol=1e-12, rtol=1e-10),
                "judgecal_variance_matches_ddof1": np.isclose(ours.standard_error ** 2, expected_ours_variance, atol=1e-12, rtol=1e-10),
                "reference_variance_matches_ddof0": np.isclose(reference_se ** 2, expected_reference_variance, atol=1e-12, rtol=1e-10),
                "reference_interval_center_matches_point": np.isclose((reference_low + reference_high) / 2, reference_point, atol=1e-12, rtol=1e-10),
                "judgecal_interval_matches_normal_formula": np.allclose(
                    [ours.interval.low, ours.interval.high],
                    [ours.point - quantile * ours.standard_error, ours.point + quantile * ours.standard_error],
                    atol=1e-12, rtol=1e-10),
            }
            records.append({
                "case": case["case"], "power_mode": str(power), "alpha": alpha,
                **moments,
                "judgecal_selected_power": ours.selected_power,
                "reference_power_from_pinned_formula": ref_power,
                "power_difference": ours.selected_power - ref_power,
                "judgecal_point": ours.point, "reference_point": reference_point,
                "point_difference": ours.point - reference_point,
                "judgecal_standard_error": ours.standard_error,
                "reference_standard_error_from_ci": reference_se,
                "expected_judgecal_variance_ddof1": expected_ours_variance,
                "expected_reference_variance_ddof0": expected_reference_variance,
                "judgecal_ci_low": ours.interval.low, "judgecal_ci_high": ours.interval.high,
                "reference_ci_low": reference_low, "reference_ci_high": reference_high,
                "ci_low_difference": ours.interval.low - reference_low,
                "ci_high_difference": ours.interval.high - reference_high,
                **{name: bool(value) for name, value in checks.items()},
                "contract_passed": bool(all(checks.values())),
            })
    return pd.DataFrame(records)


def write_artifacts(output: Path, seed: int = 2026, alpha: float = .05) -> dict:
    reference, reference_metadata = load_reference()
    cases = build_seeded_cases(seed)
    comparisons = compare_cases(cases, reference, alpha)
    source = Path(inspect.getsourcefile(reference.ppi_mean_pointestimate)).resolve()
    source_hashes = {str(path.relative_to(source.parent)): _sha256(path)
                     for path in sorted(source.parent.rglob("*.py"))}
    configuration = {
        "schema_version": 1, "seed": seed, "alpha": alpha,
        "scope": "iid scalar-mean numerical convention comparison; not a performance or coverage benchmark",
        "protocol": "retrospective versioned fixtures; development and exploratory checks informed choices",
        "reference_distribution": "ppi-python", "reference_version": REFERENCE_VERSION,
        "reference_wheel_url": REFERENCE_WHEEL_URL, "reference_wheel_sha256": REFERENCE_WHEEL_SHA256,
        "reference_ppi_source_sha256": REFERENCE_PPI_SOURCE_SHA256,
        "powers": list(FIXED_POWERS) + ["auto"],
        "tolerance": {"atol": 1e-12, "rtol": 1e-10},
        "cases": cases,
    }
    installed = {dist.metadata["Name"]: dist.version for dist in metadata.distributions() if dist.metadata["Name"]}
    environment = {
        "python": sys.version, "platform": platform.platform(), "versions": dict(sorted(installed.items())),
        "reference": reference_metadata, "reference_package_source_sha256": source_hashes,
        "repository_source_hash_normalization": "CRLF replaced with LF before SHA-256; reference installed files remain raw-byte hashes",
        "judgecal_ppi_source_sha256": _source_sha256(Path(inspect.getsourcefile(prediction_powered_win_rate))),
        "comparison_script_sha256": _source_sha256(Path(__file__)),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _write_text(output / "cases.json", json.dumps(configuration, indent=2, allow_nan=False) + "\n")
    _write_text(output / "environment.json", json.dumps(environment, indent=2, allow_nan=False) + "\n")
    comparisons.to_csv(output / "comparisons.csv", index=False, float_format="%.17g", lineterminator="\n")
    fixed = comparisons[comparisons.power_mode != "auto"]
    automatic = comparisons[comparisons.power_mode == "auto"]
    passed = bool(comparisons.contract_passed.all())
    summary = {
        "cases": len(cases), "comparisons": len(comparisons),
        "contract_passed": passed,
        "fixed_max_abs_point_difference": float(fixed.point_difference.abs().max()),
        "auto_max_abs_point_difference": float(automatic.point_difference.abs().max()),
        "auto_max_abs_power_difference": float(automatic.power_difference.abs().max()),
    }
    _write_text(output / "summary.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
    _write_text(output / "SUMMARY.md",
        "# Pinned author-reference numerical comparison\n\n"
        f"Reference: `ppi-python=={REFERENCE_VERSION}`. Cases: {len(cases)}; comparisons: {len(comparisons)}. "
        f"All numerical convention checks passed: **{passed}**.\n\n"
        f"Maximum fixed-power point difference: {summary['fixed_max_abs_point_difference']:.3g}. "
        f"Maximum automatic-power point difference: {summary['auto_max_abs_point_difference']:.6g}; "
        f"maximum automatic coefficient difference: {summary['auto_max_abs_power_difference']:.6g}.\n\n"
        "Fixed-power points should match. Interval widths differ because judgecal uses ddof=1 "
        "per-pool variance and this reference uses ddof=0 interval moments. Automatic coefficients "
        "also differ: judgecal uses separate-pool ddof=1 moments, while the reference uses audit "
        "covariance with divisor n and pooled judge variance with ddof=1. Every observed difference "
        "is retained in comparisons.csv. The reference coefficient column is computed from the "
        "pinned source formula and checked through its public point/CI calls, not returned by the API.\n\n"
        "This validates numerical conventions on the exported fixtures; it does not establish "
        "statistical superiority, real-data accuracy, population coverage, or clustered-reference equivalence. "
        "The script performed no downloads or model calls. See cases.json and environment.json "
        "for exact arrays, source fingerprints, API signatures, and installed versions.\n",
    )
    artifacts = ("cases.json", "environment.json", "comparisons.csv", "summary.json", "SUMMARY.md")
    _write_text(output / "artifact_hashes.json", json.dumps({
        "algorithm": "sha256", "normalization": "raw bytes",
        "artifacts": {name: _sha256(output / name) for name in artifacts},
    }, indent=2, allow_nan=False) + "\n")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/reference"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--alpha", type=float, default=.05)
    args = parser.parse_args(argv)
    try:
        summary = write_artifacts(args.output, args.seed, args.alpha)
    except (RuntimeError, ValueError) as exc:
        parser.exit(2, f"Reference comparison failed: {exc}\n")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["contract_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
