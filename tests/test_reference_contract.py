"""Offline fixture checks plus explicitly optional pinned-author integration."""
import importlib.metadata as metadata
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest


HELPERS = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples" / "compare_ppi_reference.py"))


def test_seeded_fixtures_are_reproducible_and_serializable_without_reference():
    cases = HELPERS["build_seeded_cases"]()
    assert cases == HELPERS["build_seeded_cases"]()
    assert cases != HELPERS["build_seeded_cases"](2027)
    assert len(cases) == 6
    assert len({case["case"] for case in cases}) == 6
    assert json.loads(json.dumps(cases, allow_nan=False)) == cases
    for case in cases:
        y, f_l, f_u = [np.array(case[key]) for key in
                       ("human_labeled", "judge_labeled", "judge_unlabeled")]
        assert len(y) == len(f_l)
        for values in (y, f_l, f_u):
            assert len(values) >= 2
            assert np.isin(values, [0, .5, 1]).all()
        assert HELPERS["scalar_moments"](y, f_l, f_u)["reference_auto_power_formula"] is not None


def test_pooled_and_per_pool_conventions_have_distinct_hand_calculated_coefficients():
    y = np.array([1., .5, 0., .5])
    f_l = np.array([1., 1., 0., 0.])
    f_u = np.array([1., 0., .5, .5])
    moments = HELPERS["scalar_moments"](y, f_l, f_u)
    # Audit covariance ddof=0 is 1/8; ddof=1 is 1/6.
    # Separate judge variance denominator: 1/3 + 1/6 = 1/2.
    # Concatenated judge variance is (3/2)/7=3/14.
    assert moments["covariance_ddof0"] == pytest.approx(1 / 8)
    assert moments["covariance_ddof1"] == pytest.approx(1 / 6)
    assert moments["judgecal_auto_power_formula"] == pytest.approx(1 / 3)
    assert moments["reference_auto_power_formula"] == pytest.approx(7 / 24)


@pytest.mark.parametrize("seed", [-1, True, 1.5, "2026"])
def test_invalid_seed_rejected_without_reference(seed):
    with pytest.raises(ValueError):
        HELPERS["build_seeded_cases"](seed)


@pytest.fixture(scope="module")
def reference():
    try:
        metadata.version("ppi-python")
    except metadata.PackageNotFoundError:
        pytest.skip("Optional integration requires pinned ppi-python==0.2.3; default suite stays offline.")
    # Wrong version, unexpected source hash, or broken dependencies must fail
    # visibly rather than silently skipping an installed but invalid baseline.
    return HELPERS["load_reference"]()[0]


def test_optional_author_reference_matches_its_documented_conventions(reference):
    comparisons = HELPERS["compare_cases"](HELPERS["build_seeded_cases"](), reference)
    assert len(comparisons) == 24
    assert comparisons.contract_passed.all()
    fixed = comparisons[comparisons.power_mode != "auto"]
    np.testing.assert_allclose(fixed.point_difference, 0, atol=1e-12)
    assert (fixed.judgecal_standard_error >= fixed.reference_standard_error_from_ci - 1e-12).all()
    automatic = comparisons[comparisons.power_mode == "auto"]
    assert automatic.power_difference.abs().max() > 1e-5
    # Different automatic coefficients are expected, not evidence of a bug.
    assert automatic.point_difference.abs().max() > 1e-5


def test_optional_reference_writes_complete_machine_readable_evidence(reference, tmp_path):
    summary = HELPERS["write_artifacts"](tmp_path)
    assert summary["contract_passed"]
    assert summary["comparisons"] == 24
    assert {path.name for path in tmp_path.iterdir()} == {
        "cases.json", "environment.json", "comparisons.csv", "summary.json", "SUMMARY.md", "artifact_hashes.json"
    }
    configuration = json.loads((tmp_path / "cases.json").read_text(encoding="utf-8"))
    environment = json.loads((tmp_path / "environment.json").read_text(encoding="utf-8"))
    assert configuration["reference_version"] == "0.2.3"
    assert environment["reference"]["source_sha256"] == HELPERS["REFERENCE_PPI_SOURCE_SHA256"]
    assert len(environment["reference_package_source_sha256"]) > 1
    assert "ppi-python" in environment["versions"]
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert len(hashes["artifacts"]) == 5
    for name, expected in hashes["artifacts"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == expected
        assert b"\r\n" not in (tmp_path / name).read_bytes()
