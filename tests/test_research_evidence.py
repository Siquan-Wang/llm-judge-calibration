"""Pinned research facts, exact selectors and a nonmutating reproducibility check."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def evidence_module():
    path = Path(__file__).resolve().parents[1] / 'examples/research_evidence.py'
    specification = importlib.util.spec_from_file_location('research_evidence_cli', path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def copy_inputs(module, destination):
    names = set(module.PINNED_LINEAGE_INPUTS)
    names.update(specification['repository_path'] for specification in module.PINNED_INPUTS.values())
    for name in module.PINNED_REPORT_MANIFESTS:
        names.add(name)
        manifest = json.loads((module.ROOT / name).read_text(encoding='utf8'))
        for artifact in manifest['artifacts_sha256']:
            if artifact == 'REPORT.md' or artifact.endswith('config.json'):
                names.add((Path(name).parent / artifact).as_posix())
    for name in names:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(module.ROOT / name, path)
    return destination


def test_exact_rows_selectors_portability_and_coverage_of_all_findings(evidence_module):
    module = evidence_module
    result = module.build_evidence()
    assert [entry['id'] for entry in result['findings']] == [f'E{i:02d}' for i in range(1, 12)]
    assert len(result['sources']) == 15
    serialized = json.dumps(result, allow_nan=False)
    assert 'absolute_path' not in serialized and 'created_utc' not in serialized
    assert str(module.ROOT) not in serialized
    for entry in result['findings']:
        assert entry['estimand'] and entry['denominator_and_aggregation']
        assert entry['dependency_and_overlap_lineage'] and entry['uncertainty']
        for item in entry['csv_evidence']:
            source = result['sources'][item['source_id']]
            path = module.ROOT / source['repository_path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256']
            data = pd.read_csv(path, float_precision='round_trip')
            for name, selection in item['row_filter'].items():
                data = data.loc[data[name].isin(selection['in'])] if 'in' in selection else data.loc[data[name].eq(selection['eq'])]
            assert len(data) == item['selected_rows']
            if 'values' in item:
                expected = module._clean(data[item['metric_columns']].to_dict('records'))
                assert item['values'] == expected
            else:
                assert 'values_policy' in item  # Full-data reduction is explicitly declared.


def test_quantitative_claims_have_correct_target_and_denominator(evidence_module):
    evidence = evidence_module.build_evidence()
    findings = {entry['id']: entry for entry in evidence['findings']}
    assert findings['E01']['derived_values']['lower_than_human_cells'] == 3
    assert [row['lower_mae_pairs'] for row in findings['E01']['derived_values']['pairwise_lower_mae_counts']] == [10, 9, 11]
    assert findings['E02']['derived_values']['changed_scores'] == 305
    assert findings['E02']['derived_values']['pair_budget_gap_sign_changes'] == 5
    assert findings['E03']['derived_values']['tuned_normal_covered'] == 524
    assert findings['E03']['derived_values']['tuned_normal_zero_width'] == 353
    assert findings['E04']['derived_values']['width_narrower_count'] == 1
    assert findings['E04']['derived_values']['width_denominator'] == 12
    assert findings['E05']['derived_values']['signed_minus_positive_mse'] < 0
    assert findings['E06']['derived_values']['signed_minus_positive_mse'] > 0
    assert findings['E05']['derived_values']['reused_llmbar_comparison']['denominator'] == 27
    targets = {row['method']: row for row in findings['E07']['csv_evidence'][0]['values']}
    assert targets['pool_tuned']['population_rmse'] > targets['population_tuned']['population_rmse']
    assert targets['pool_tuned']['pool_rmse'] < targets['population_tuned']['pool_rmse']
    wrong = findings['E08']['derived_values']
    assert wrong['wrong_fraction_among_agreements'] == 176 / 205
    assert wrong['agreement_fraction'] == 205 / 319
    assert findings['E09']['derived_values']['both_reference_wrong_count'] == 160
    reversal = findings['E10']['derived_values']['gpt4o_budget40_conflicting_loss_metrics']
    assert reversal['audit_residual_mae'] < reversal['reference_audit_mae']
    assert reversal['audit_residual_rmse'] > reversal['reference_audit_rmse']
    repeated = findings['E11']['csv_evidence'][0]['values']
    assert [row['coverage'] for row in repeated] == [.89, .596]
    assert all(row['repetitions'] == 500 for row in repeated)
    assert all(row['repetitions'] == 1000 for row in findings['E11']['csv_evidence'][1]['values'])


def test_deterministic_exports_and_nonmutating_check(evidence_module, tmp_path):
    module = evidence_module
    module.main(['--output-directory', str(tmp_path)])
    names = ['research_evidence.json', 'RESEARCH_EVIDENCE.md']
    before = {name: ((tmp_path / name).read_bytes(), (tmp_path / name).stat().st_mtime_ns) for name in names}
    module.main(['--output-directory', str(tmp_path), '--check'])
    assert before == {name: ((tmp_path / name).read_bytes(), (tmp_path / name).stat().st_mtime_ns) for name in names}
    other = tmp_path / 'second'
    module.main(['--output-directory', str(other)])
    assert {name: content[0] for name, content in before.items()} == {name: (other / name).read_bytes() for name in names}
    report = (tmp_path / 'RESEARCH_EVIDENCE.md').read_text(encoding='utf8')
    assert '52.4%' in report and '89.0%' in report and '59.6%' in report
    assert all(f'## E{i:02d}:' in report for i in range(1, 12))
    (tmp_path / 'RESEARCH_EVIDENCE.md').write_text('incorrect summary', encoding='utf8')
    with pytest.raises(ValueError, match='generated evidence differs'):
        module.main(['--output-directory', str(tmp_path), '--check'])
    assert (tmp_path / 'RESEARCH_EVIDENCE.md').read_text() == 'incorrect summary'


@pytest.mark.parametrize('mode', ['csv', 'self_declared_hash', 'manifest', 'report', 'configuration', 'lineage'])
def test_source_and_self_declared_manifest_tampering_rejected(evidence_module, tmp_path, mode):
    module = evidence_module
    copied = copy_inputs(module, tmp_path)
    if mode in ('csv', 'self_declared_hash'):
        path = copied / 'reports/research/summary.csv'
        path.write_bytes(path.read_bytes() + b'\n')
        if mode == 'self_declared_hash':
            sidecar = copied / 'reports/research/reproducibility.json'
            contents = json.loads(sidecar.read_text())
            contents['artifacts_sha256']['summary.csv'] = hashlib.sha256(path.read_bytes()).hexdigest()
            sidecar.write_text(json.dumps(contents), encoding='utf8')
    elif mode == 'manifest':
        path = copied / 'reports/finite-sample/reproducibility.json'
        path.write_bytes(path.read_bytes() + b'\n')
    elif mode == 'report':
        path = copied / 'reports/rewardbench-audit/REPORT.md'
        path.write_bytes(path.read_bytes() + b'changed interpretation\n')
    elif mode == 'configuration':
        path = copied / 'reports/research/simulation_config.json'
        path.write_bytes(path.read_bytes() + b'\n')
    else:
        path = copied / 'reports/llmbar/labels.csv'
        path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='differs'):
        module.build_evidence(copied)


def test_alternative_root_has_identical_portable_evidence(evidence_module, tmp_path):
    copied = copy_inputs(evidence_module, tmp_path)
    assert evidence_module.build_evidence(copied) == evidence_module.build_evidence()
