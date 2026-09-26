"""Build or verify a deterministic crosswalk of eleven published research findings.

No model, network request or source-artifact rewrite is performed. CSV and
historical report-manifest byte hashes are pinned independently in this file.
Historical code fingerprints describe those runs; they are not asserted to
match the continually evolving current implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_COMMIT = 'b4bdeab08481a25b529d81fa909f96aab10f368f'
PINNED_INPUTS = {'mt_fixed_summary': {'repository_path': 'reports/research/summary.csv',
                      'sha256': 'e7e4f6286fb44b89a660d747b11f187a42bfe9bd529eb95e16b68bf18f27a00f',
                      'rows': 12},
 'reference_changes': {'repository_path': 'reports/sensitivity/reference_changes.csv',
                       'sha256': 'f995f0f084d5308a4c7d88e0be885a608c76c89ecb11b9e5ff51193b813dcb98',
                       'rows': 1814},
 'reference_pair_gaps': {'repository_path': 'reports/sensitivity/paired_method_gaps.csv',
                         'sha256': '0b93e78e8a1078cf31e142ff6b80badbbc7ae99a084aab38c5585df20b4cec7a',
                         'rows': 90},
 'reference_summary': {'repository_path': 'reports/sensitivity/summary.csv',
                       'sha256': '2a7e4a79a1333f92a62be7db1962c9e83491e66464803702d3a7a8279f8b8e2d',
                       'rows': 24},
 'finite_summary': {'repository_path': 'reports/finite-sample/summary.csv',
                    'sha256': '1bd5f5b609865247bd68739bb98a957768fa019303e5153c0cdfa58350b248a5',
                    'rows': 112},
 'signed_simulation_summary': {'repository_path': 'reports/signed-power/simulation_summary.csv',
                               'sha256': '25287d171c50227bb944ed46cbd3e4bd52cc61f883630c57336b0deb2bee1999',
                               'rows': 72},
 'estimand_simulation_summary': {'repository_path': 'reports/estimand/simulation_summary.csv',
                                 'sha256': 'e85107c61972ca91bf62a40ef6df2b88c690591724b5b1cca76fff4d882f3a55',
                                 'rows': 144},
 'llmbar_diagnostics': {'repository_path': 'reports/llmbar-audit/corpus_diagnostics.csv',
                        'sha256': 'd75a0c99c914570b59c3db1990ca5b89fff4492f7441b1d1f0536b80dd306b4d',
                        'rows': 21},
 'rewardbench_diagnostics': {'repository_path': 'reports/rewardbench-audit/corpus_diagnostics.csv',
                             'sha256': '7530f854aee99edc2da7509d04b6f411dd6c4016d48b325ab312ab14369370b9',
                             'rows': 52},
 'rewardbench_summary': {'repository_path': 'reports/rewardbench-audit/summary.csv',
                         'sha256': 'fa4870c417e8f200a4a53991f4f7877645bfd35cf3fb2ede789f8d7451a0059b',
                         'rows': 108},
 'mt_fixed_per_pair': {'repository_path': 'reports/research/per_pair.csv',
                       'sha256': 'b5936a5089cd1ac2fc226ada485b980a751ab90859924b37e901ba370725b4aa',
                       'rows': 180},
 'signed_llmbar_summary': {'repository_path': 'reports/signed-power/llmbar_summary.csv',
                           'sha256': 'e237b97b21750bd808f42b3d5a3d61b93b8ff7d6ab780dd74ee9d352697da788',
                           'rows': 135},
 'llmbar_summary': {'repository_path': 'reports/llmbar-audit/summary.csv',
                    'sha256': 'dd85dcbeb25c167944e81bf845ba91cdb71e4df21b8f341a922db997a4113688',
                    'rows': 108},
 'stress_summary': {'repository_path': 'reports/stress/summary.csv',
                    'sha256': '7100b9923f55de25b40dc0852dff526ec34de2657af4a9494ad6ffebce7c356e',
                    'rows': 168},
 'research_simulation_summary': {'repository_path': 'reports/research/simulation_summary.csv',
                                 'sha256': '1bb5ac6e8a7519666f93440249463d5f72cf6ad4e789c2ff151340d52db082b9',
                                 'rows': 28}}
PINNED_REPORT_MANIFESTS = {'reports/research/reproducibility.json': 'a63d170f0ccc0ddddb5a05215cd79fa59597a964cd480487235a597b42692e26',
 'reports/sensitivity/reproducibility.json': 'e79d7eda781a2cfec82c13c922614991c594c574010d5e621ccf6ba0e58d88d8',
 'reports/finite-sample/reproducibility.json': '05f22c16822087d2842768389787b7015a4e8df67253f259755ab7f3b3bb6fff',
 'reports/signed-power/reproducibility.json': 'e6ae7ea21d794d1db5095713083cd5bcad5ce7e1a2c0fdf779bb8ff433bab854',
 'reports/estimand/reproducibility.json': 'f0ef7e2816ac026798003bc7449abff4341bad2c8b48007aca666e66bd77b5ea',
 'reports/llmbar-audit/reproducibility.json': '25e525937416f68f7feed67f4cddf36f8228a9c7a41213eb4e34e5dfb3e404ee',
 'reports/rewardbench-audit/reproducibility.json': '6efd6c230b0d4dc1c4f1174366d8e3bda9cc9d63048ad8d65d1140062819bd8d',
 'reports/stress/reproducibility.json': '8347d637ebea67905ce9ec38ed40319895341017cdac94528698df35e0a2ae67'}
PINNED_LINEAGE_INPUTS = {'reports/mtbench/labels.csv': 'ac0b5b074a4c0ecef978983ee7ab7c64c82f94c4eea38def24c68938d96ab943',
 'reports/llmbar/labels.csv': 'aafff984b410bd1f60e0a3196a10957c5a1f241caf8c85c4c391d8b1e2ef1f97',
 'reports/rewardbench/labels.csv': '7dc7d8aad832f470a2831df5011fd7e8e1c670996a225b5d91c81f760350120f'}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs(root=ROOT):
    """Verify fixed byte pins and selected CSV membership in report manifests."""
    root = Path(root)
    manifests = {}
    for relative, expected in PINNED_REPORT_MANIFESTS.items():
        path = root / relative
        if _sha256(path) != expected:
            raise ValueError(f"pinned report manifest differs: {relative}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        # Report and configuration are also part of the cited historical claim.
        for name in (name for name in manifest.get("artifacts_sha256", {}) if name == "REPORT.md" or name.endswith("config.json")):
            expected_artifact = manifest.get("artifacts_sha256", {}).get(name)
            if not expected_artifact or _sha256(path.parent/name) != expected_artifact:
                raise ValueError(f"report artifact differs from pinned manifest: {path.parent.name}/{name}")
        manifests[path.parent.relative_to(root).as_posix()] = manifest
    for specification in PINNED_INPUTS.values():
        relative = specification["repository_path"]
        path = root / relative
        if _sha256(path) != specification["sha256"]:
            raise ValueError(f"pinned CSV differs: {relative}")
        manifest = manifests[path.parent.relative_to(root).as_posix()]
        if manifest.get("artifacts_sha256", {}).get(path.name) != specification["sha256"]:
            raise ValueError(f"CSV not certified by report manifest: {relative}")
    for relative, expected in PINNED_LINEAGE_INPUTS.items():
        if _sha256(root/relative) != expected:
            raise ValueError(f"pinned lineage input differs: {relative}")


def _clean(obj):
    if isinstance(obj, dict):
        return {str(key): _clean(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(value) for value in obj]
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def build_evidence(root=ROOT):
    """Extract declared rows and derived facts after independent byte-pin checks."""
    ROOT = Path(root)
    verify_inputs(ROOT)
    COMMIT = REPORT_COMMIT
    BASE = f"https://github.com/Siquan-Wang/llm-judge-calibration/blob/{COMMIT}/"
    sources, findings = {}, []
    def source(name,path):
        file=ROOT/path;data=pd.read_csv(file,float_precision='round_trip')
        if len(data) != PINNED_INPUTS[name]['rows']:
            raise ValueError(f'CSV row count differs: {path}')
        sources[name]={'repository_path':path,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                       'rows':len(data),'immutable_link':BASE+path, 'report_manifest':str(Path(path).parent/'reproducibility.json').replace('\\','/'), 'report_manifest_sha256':PINNED_REPORT_MANIFESTS[(Path(path).parent/'reproducibility.json').as_posix()]}
        return data


    def select(source_id,data,filters,columns):
        mask=pd.Series(True,index=data.index)
        for key,rule in filters.items():
            mask &= data[key].isin(rule['in']) if 'in' in rule else data[key].eq(rule['eq'])
        selected=data.loc[mask,columns]
        return {'source_id':source_id,'row_filter':filters,'selected_rows':len(selected),
                'metric_columns':columns,'values':selected.to_dict('records')}


    def add(identifier,title,report,estimand,denominator,evidence,derived,lineage,uncertainty,limits):
        findings.append({'id':identifier.split('_')[0],'key':identifier,'title':title,'report_path':report,'report_link':BASE+report,
            'estimand':estimand,'denominator_and_aggregation':denominator,'csv_evidence':evidence,
            'derived_values':derived,'dependency_and_overlap_lineage':lineage,'uncertainty':uncertainty,
            'interpretation_limits':limits})


    MT={'corpus':'MT-Bench human-vote snapshot; older six-model output panel and 80 selected questions',
        'input_csv':'reports/mtbench/labels.csv','input_sha256':hashlib.sha256((ROOT/'reports/mtbench/labels.csv').read_bytes()).hexdigest(),
        'related_reports':['reports/research','reports/sensitivity','reports/mtbench'],
        'dependence':'Same questions recur across model pairs, both turns, nested budgets and 30 splits. Sensitivity is a reanalysis of identical comparisons/splits, not independent replication. Historical reports/mtbench has a different complementary-pool design and must not be pooled with fixed-target results.'}
    LLM={'corpus':'Certified LLMBar 419 comparisons, 418 exact instructions, three historical two-order judge caches',
         'input_csv':'reports/llmbar/labels.csv','input_sha256':hashlib.sha256((ROOT/'reports/llmbar/labels.csv').read_bytes()).hexdigest(),
         'related_reports':['reports/llmbar-audit','reports/signed-power','reports/estimand','reports/rewardbench-audit'],
         'dependence':'Signed and estimand empirical extensions reuse the same LLMBar panel and historical splits. RewardBench includes the same 419-comparison source component with different cached judges. Cohorts, methods and repeated splits are not independent evidence.'}
    RB={'corpus':'RewardBench: 2,985 comparisons / 2,733 exact prompts, two dated OpenAI 2024 single-choice caches',
        'input_csv':'reports/rewardbench/labels.csv','input_sha256':hashlib.sha256((ROOT/'reports/rewardbench/labels.csv').read_bytes()).hexdigest(),
        'related_reports':['reports/rewardbench-audit','reports/llmbar-audit','reports/signed-power','reports/estimand','reports/research'],
        'dependence':'NonLLMBar excludes 419 known LLMBar comparisons and shares no exact prompt with that component; All overlaps both. Remaining sources still include MT-Bench-derived comparisons, so NonLLMBar is not certified as wholly independent benchmark lineage. Both target directions use the same two judgments/reference labels; 30 splits and budgets overlap.'}

    mt=source('mt_fixed_summary','reports/research/summary.csv')
    table=mt.pivot(index='labeled_fraction',columns='method',values='mean_absolute_error')
    gaps=[{'labeled_fraction':float(b),'tuned_minus_human_mae':float(r.ppi_tuned-r.human_only),
           'tuned_minus_fixed_mae':float(r.ppi_tuned-r.ppi)} for b,r in table.iterrows()]
    add('E01_mtbench_fixed_target','Tuned correction improves aggregate fixed-target MAE at all three budgets',
        'reports/research/REPORT.md','MAE against each model-pair/split realized held-out plurality A-score; categorical A/tie/B scores 1/.5/0.',
        {'cells':3,'model_pairs':15,'seeds_per_pair':30,'trial_losses_per_method_budget':450,'evaluation_question_fraction':.25,'audit_fraction_denominator':'Total eligible questions within each pair','aggregation':'Equal mean over450pair/split errors per method/budget; these are dependent losses.'},
        [select('mt_fixed_summary',mt,{},list(mt.columns))],
        {'operation':'For each labeled_fraction subtract human_only or ppi mean_absolute_error from ppi_tuned; count strict negative differences.',
         'per_budget_gaps':gaps,'lower_than_human_cells':sum(x['tuned_minus_human_mae']<0 for x in gaps),'lower_than_fixed_cells':sum(x['tuned_minus_fixed_mae']<0 for x in gaps),'cell_denominator':3},MT,
        {'type':'Descriptive dependent-split averages; no iid MCSE or significance claim'},
        ['This is aggregate direction, not improvement for every pair/split.','Existing population intervals do not target realized-pool error coverage.'])

    changes=source('reference_changes','reports/sensitivity/reference_changes.csv')
    g=source('reference_pair_gaps','reports/sensitivity/paired_method_gaps.csv')
    sens=source('reference_summary','reports/sensitivity/summary.csv')
    aligned=g.pivot(index=['model_a','model_b','labeled_fraction'],columns='reference_definition',values='tuned_minus_human_mae')
    flips=aligned.loc[np.sign(aligned.mean_vote)!=np.sign(aligned.plurality)].reset_index()
    aggregate=sens.pivot(index=['reference_definition','labeled_fraction'],columns='method',values='mean_absolute_error')
    add('E02_reference_definition','Reference definition changes 305 comparison scores and 5 of 45 pair/budget method-gap signs',
        'reports/sensitivity/REPORT.md','Separate realized held-out means of human plurality scores versus equally comparison-weighted mean human vote scores; neither is a pooled-vote target or latent truth.',
        {'comparisons':len(changes),'recorded_votes':int(changes.n_human_votes.sum()),'single_vote_comparisons':int(changes.n_human_votes.eq(1).sum()),'matched_pair_budget_cells':len(aligned),'reference_budget_cells':len(aggregate),'seeds_per_pair':30},
        [{'source_id':'reference_changes','row_filter':{},'selected_rows':len(changes),'metric_columns':['n_human_votes','score_change'],'values_policy':'All 1,814 rows are aggregated by the exact operations below; no omitted-score filter.'},
         select('reference_pair_gaps',g,{},list(g.columns)),select('reference_summary',sens,{'method':{'in':['human_only','ppi_tuned']}},['reference_definition','labeled_fraction','method','mean_absolute_error','trials'])],
        {'operation':'Count score_change!=0; mean(abs(score_change)); align by model_a/model_b/labeled_fraction, count np.sign(mean_vote_gap)!=np.sign(plurality_gap), with exactzero signs and no tolerance.',
         'changed_scores':int(changes.score_change.ne(0).sum()),'mean_absolute_score_change':float(changes.score_change.abs().mean()),
         'pair_budget_gap_sign_changes':len(flips),'changed_pair_budget_rows':flips.to_dict('records'),
         'tuned_lower_mae_reference_budget_cells':int(aggregate.ppi_tuned.lt(aggregate.human_only).sum())},MT,
        {'type':'Descriptive target sensitivity; repeated question/pair/split dependence, no independent-trial SE'},
        ['Lower error under one reference does not make that reference superior.','Single-vote observations cannot reveal within-comparison annotator disagreement.'])

    finite=source('finite_summary','reports/finite-sample/summary.csv')
    fcols=['scenario','method','truth','repetitions','n_labeled','n_unlabeled','rmse','rmse_mcse','coverage','coverage_mcse','coverage_mc_low','coverage_mc_high','mean_width','mean_width_mcse','zero_width_draws','fraction_full_domain_covered','theorem_applicable']
    synthetic_finite={'study':'Finite-sample retrospective follow-up','configuration':'reports/finite-sample/config.json','seed':2028,
        'relation':'Synthetic data, not corpus observations. Shared draws across 8 methods within each scenario. The 12 IID cells and 2 intentional assumption-violation cells; only specified IID cells support finite-theorem applicability. The same mechanism family recurs in earlier/later studies with distinct experiment seeds; do not pool as new domains.'}
    add('E03_normal_undercoverage','Near-boundary lower point RMSE coexists with 52.4% normal coverage',
        'reports/finite-sample/REPORT.md','Population binary outcome mean p=.95; IID frozen proxy sensitivity .95/specificity .90; nominal 95% population interval.',
        {'scenario':'iid_rare_strong_n0020','repetitions_per_method':1000,'audit_rows':20,'prediction_rows':200,'methods_share_draws':True},
        [select('finite_summary',finite,{'scenario':{'eq':'iid_rare_strong_n0020'},'method':{'in':['human_normal','ppi_tuned_normal','human_hoeffding','ppi_eb_grid']}},fcols)],
        {'operation':'Compare target-matched RMSE, coverage and untruncated width in the selected same-draw scenario. Coverage counts are coverage*1000.',
         'tuned_normal_covered':int(round(finite.loc[(finite.scenario=='iid_rare_strong_n0020')&(finite.method=='ppi_tuned_normal'),'coverage'].iloc[0]*1000)),'tuned_normal_zero_width':int(finite.loc[(finite.scenario=='iid_rare_strong_n0020')&(finite.method=='ppi_tuned_normal'),'zero_width_draws'].iloc[0]),'human_hoeffding_covered':int(round(finite.loc[(finite.scenario=='iid_rare_strong_n0020')&(finite.method=='human_hoeffding'),'coverage'].iloc[0]*1000)),'eb_grid_full_domain_count':int(round(finite.loc[(finite.scenario=='iid_rare_strong_n0020')&(finite.method=='ppi_eb_grid'),'fraction_full_domain_covered'].iloc[0]*1000))},synthetic_finite,
        {'type':'1,000 independent simulation replications; paired methods. Coverage_mc_low/high are pointwise95percent exact-binomial intervals; no simultaneous claim.'},
        ['Normal interval coverage is empirical, not guaranteed at this audit size.','Finite intervals rely on IID/transport/boundedness assumptions and may be uninformatively wide.'])

    iid=finite.loc[finite.scenario_family.eq('iid')]
    theorem=iid.loc[iid.theorem_applicable]
    width=iid.pivot(index='scenario',columns='method',values='mean_width')
    width_rows=[{'scenario':k,'grid_mean_width':float(v.ppi_eb_grid),'human_hoeffding_mean_width':float(v.human_hoeffding),'grid_minus_human_hoeffding_width':float(v.ppi_eb_grid-v.human_hoeffding),'narrower':bool(v.ppi_eb_grid<v.human_hoeffding)} for k,v in width.iterrows()]
    add('E04_finite_width_tradeoff','Conservative finite intervals cover frequently but grid EB beats human Hoeffding width in only 1 of 12 IID settings',
        'reports/finite-sample/REPORT.md','Known population binary mean in 12 prespecified IID settings; interval raw untruncated width, not clipped domain width or deployment label efficiency.',
        {'iid_scenarios':12,'theorem_applicable_method_scenario_cells':len(theorem),'replications_per_cell':1000,'nominal_interval_coverage':.95},
        [select('finite_summary',finite,{'theorem_applicable':{'eq':True}},fcols),
         select('finite_summary',finite,{'scenario_family':{'eq':'iid'},'method':{'in':['ppi_eb_grid','human_hoeffding','human_eb']}},['scenario','method','repetitions','mean_width','mean_width_mcse','width_difference_vs_human_family','width_difference_vs_human_family_mcse'])],
        {'operation':'Across theorem_applicable=True rows take min/max observed coverage; pivot IID means byscenario and compare grid EB width with standalone human Hoeffding.',
         'coverage_min':float(theorem.coverage.min()),'coverage_max':float(theorem.coverage.max()),'width_narrower_count':sum(x['narrower'] for x in width_rows),'width_denominator':len(width_rows),'per_scenario_widths':width_rows},synthetic_finite,
        {'type':'Each cell has pointwise coverage Monte Carlo interval and mean-width MCSE. Human-Hoeffding comparison is descriptive across families; same-family paired-width MCSE is separately exported against human EB.'},
        ['Observed coverage does not prove the underlying finite-sample theorem.','The12-cell denominator excludes deliberate dependence and shift violations.','A width greater than 1 does not necessarily contain the whole [0,1] domain.'])

    signed=source('signed_simulation_summary','reports/signed-power/simulation_summary.csv')
    scols=['scenario','method','prevalence','profile','n_labeled','n_unlabeled','repetitions','rmse','rmse_mcse','mean_selected_power','coverage','coverage_mc_low','coverage_mc_high','mse_difference_vs_positive','mse_difference_vs_positive_mcse']
    for identifier,profile,title in [('E05_signed_inverse','inverse','Signed tuning exploits inverse signal at a small audit budget'),('E06_signed_noise','uninformative','Allowing negative powers can increase observed error when the proxy carries no signal')]:
        scenario=f'iid_p050_{profile}_n0020'
        row=signed.loc[signed.scenario.eq(scenario)&signed.method.eq('ppi_signed')].iloc[0]
        base=signed.loc[signed.scenario.eq(scenario)&signed.method.eq('ppi_tuned')].iloc[0]
        add(identifier,title,'reports/signed-power/REPORT.md','Population binary outcome prevalence p=.5; paired comparison of population-variance tuning over [0,1] versus [-1,1].',
            {'scenario':scenario,'audit_rows':20,'prediction_rows':200,'independent_replications':1000,'sensitivity':.05 if profile=='inverse' else .60,'specificity':.10 if profile=='inverse' else .40},
            [select('signed_simulation_summary',signed,{'scenario':{'eq':scenario},'method':{'in':['human_only','ppi_tuned','ppi_signed']}},scols)],
            {'operation':'Subtract positive-range RMSE from signed RMSE; paired MSE contrast/MCSE are direct source columns computed on shared draws.',
             'signed_minus_positive_rmse':float(row.rmse-base.rmse),'signed_minus_positive_mse':float(row.mse_difference_vs_positive),'paired_mse_difference_mcse':float(row.mse_difference_vs_positive_mcse)},
            {'configuration':'reports/signed-power/config.json','seed':2029,'dependence':'One scenario from 18 synthetic settings; all 4 methods share each draw. Design motivated by earlier LLMBar negative association; empirical LLMBar extension is repeated evidence, while this finding uses synthetic outcomes only. Mechanism family overlaps other simulations; no cross-study pooling.'},
            {'type':'Independent simulation replications with same-draw paired MSE MCSE; RMSE MCSE is also retained.'},
            ['These are observed finite-sample effects for the declared law, not uniform dominance or deployment savings.','A larger coefficient range can fit audit noise; lower point risk does not establish normal interval validity.'])

    est=source('estimand_simulation_summary','reports/estimand/simulation_summary.csv')
    ecols=['scenario','method','profile','prevalence','n_labeled','n_unlabeled','repetitions','oracle_population_power','oracle_pool_power','mean_selected_power','population_rmse','population_rmse_mcse','pool_rmse','pool_rmse_mcse','population_mse_difference_vs_population_tuned','population_mse_difference_vs_population_tuned_mcse','pool_mse_difference_vs_population_tuned','pool_mse_difference_vs_population_tuned_mcse']
    add('E07_estimand_tradeoff','The same switch worsens population RMSE while improving realized-pool RMSE',
        'reports/estimand/REPORT.md','Two explicitly separate targets: analytic population prevalence p=.5, and each draw\'s realized held-out binary mean. Compare methods within each target, never compare unlike-target RMSE as a superiority metric.',
        {'scenario':'iid_p050_positive_n0200_r005','audit_rows':200,'prediction_rows':100,'N_over_n':.5,'replications':1000,'proxy_sensitivity':.95,'proxy_specificity':.90},
        [select('estimand_simulation_summary',est,{'scenario':{'eq':'iid_p050_positive_n0200_r005'},'method':{'in':['population_tuned','pool_tuned']}},ecols)],
        {'operation':'Read each target-specific RMSE for population_tuned and pool_tuned; paired contrasts are pool_tuned squared loss minus population_tuned squared loss within the same target/draw. Positive population contrast and negative pool contrast establish the opposite descriptive directions.'},
        {'configuration':'reports/estimand/config.json','seed':2030,'dependence':'All 4 point rules and both targets use identical draws; target scores are correlated, not independent experiments. The 36 settings reuse earlier proxy-law families at additional pool ratios. Empirical LLMBar extension elsewhere repeats the original corpus and is not included in this synthetic finding.'},
        {'type':'1,000 independent draws; paired MSE contrasts have separate within-target MCSEs. Oracle powers are diagnostics, never fitted-method inputs.'},
        ['Same-audit estimated coefficients need not achieve their oracle criterion at small n.','IID random-pool intervals are marginal over both pools, not conditional guarantees for a frozen empirical corpus.'])

    ld=source('llmbar_diagnostics','reports/llmbar-audit/corpus_diagnostics.csv')
    add('E08_consistently_wrong','ChatGPT agrees across orders on 176 reference-wrong adversarial comparisons',
        'reports/llmbar-audit/REPORT.md','Historical ChatGPT original-order canonical choice correctness against curated LLMBar instruction-following reference; valid two-order agreement is a separate gold-free proxy.',
        {'cohort_comparisons':319,'exact_instructions':318,'valid_agreement_comparisons':205,'consistently_wrong_numerator':176,'reference_correct_numerator':90},
        [select('llmbar_diagnostics',ld,{'cohort':{'eq':'Adversarial'},'judge':{'eq':'ChatGPT'}},list(ld.columns))],
        {'operation':'Use all 319 adversarial comparisons in accuracy and agreement rates; among 205 valid agreements,176 are reference-incorrect. Distinguish denominator 319 from 205.',
         'wrong_fraction_among_agreements':float(ld.loc[(ld.cohort=='Adversarial')&(ld.judge=='ChatGPT'),'agree_but_wrong'].iloc[0]/ld.loc[(ld.cohort=='Adversarial')&(ld.judge=='ChatGPT'),'agree_valid_count'].iloc[0]),'agreement_fraction':float(ld.loc[(ld.cohort=='Adversarial')&(ld.judge=='ChatGPT'),'order_agreement'].iloc[0]),'original_order_correctness_fraction':float(ld.loc[(ld.cohort=='Adversarial')&(ld.judge=='ChatGPT'),'forward_accuracy'].iloc[0])},LLM,
        {'type':'Complete fixed-corpus counts, not a random deployment sample or confidence interval'},
        ['Agreement is not accuracy or latent truth.','Adversarial source construction includes model-based filtering; judge comparisons inherit selection and dated cache limitations.'])

    rd=source('rewardbench_diagnostics','reports/rewardbench-audit/corpus_diagnostics.csv')
    rs=source('rewardbench_summary','reports/rewardbench-audit/summary.csv')
    add('E09_shared_proxy_asymmetry','One shared RewardBench agreement signal corresponds to two different reference accuracies',
        'reports/rewardbench-audit/REPORT.md','Equal-comparison-weighted correctness relative to heterogeneous RewardBench references, versus cross-model canonical-choice agreement; not official weighted leaderboard score.',
        {'primary_cohort':'NonLLMBar','comparisons_per_target':2566,'prompt_groups':2315,'target_directions':2,'reference_labels_shared':True},
        [select('rewardbench_diagnostics',rd,{'cohort':{'eq':'NonLLMBar'}},list(rd.columns))],
        {'operation':'Read identical cross_judge_agreement for two targets and their distinct reference_accuracy; multiply rates by 2,566 only to recover integer counts.',
         'agreement_count':int(rd.loc[(rd.cohort=='NonLLMBar')].agree_count.iloc[0]),'both_reference_wrong_count':int(rd.loc[(rd.cohort=='NonLLMBar')].agree_but_wrong.iloc[0])},RB,
        {'type':'Fixed-corpus descriptive counts; target directions share both caches and labels'},
        ['Two models belong to one family, and both caches are historical2024versions.','Full-corpus diagnostics describe outcomes but never enter proxy construction or coefficient fitting.'])

    primary=rs.loc[rs.cohort.eq('NonLLMBar')]
    rt=primary.pivot(index=['target_judge','labeled_fraction'],columns='method',values='mean_absolute_error')
    counts={}
    for method in ('ppi','ppi_tuned','ppi_signed','audit_residual'):
        diff=rt[method]-rt.human_only;tie=np.isclose(diff,0,rtol=0,atol=1e-12)
        counts[method]={'lower_mae_cells':int(((diff<0)&~tie).sum()),'tied_cells':int(tie.sum()),'higher_mae_cells':int(((diff>0)&~tie).sum()),'denominator':len(diff)}
    full,small='openai/gpt-4o-2024-08-06','openai/gpt-4o-mini-2024-07-18'
    add('E10_rewardbench_primary_gaps','Tuning improves primary audit-baseline MAE descriptively, but not every alternative or cell',
        'reports/rewardbench-audit/REPORT.md','Per-target error against realized held-out strict reference accuracy; MAE averages 30 dependent fixed-corpus splits within each target/budget cell.',
        {'primary_cohort':'NonLLMBar','target_judges':2,'audit_group_fractions':[.2,.4,.6],'cells_per_method_contrast':6,'seeds_per_cell':30,'evaluation_group_fraction':.25},
        [select('rewardbench_summary',rs,{'cohort':{'eq':'NonLLMBar'}},list(rs.columns))],
        {'operation':'For each target/budget subtract human_only mean_absolute_error; ties abs(delta)<=1e-12. Counts concern6 cells, not 6 independent experiments.',
         'method_counts_vs_reference_only':counts,
         'signed_minus_raw_mae_gpt4o_budget20':float(rt.loc[(full,.2),'ppi_signed']-rt.loc[(full,.2),'raw_proxy']),
         'audit_residual_minus_reference_mae_mini_budget40':float(rt.loc[(small,.4),'audit_residual']-rt.loc[(small,.4),'human_only'])},RB,
        {'type':'Descriptive paired overlapping-split means, no iid MCSE or significance claim'},
        ['Signed tuning can lose to the zero-label raw proxy at a particular target/budget even when it improves against reference-only audit.','Audit-residual tuning retains an adverse primary cell.','All cells and failures retained; no claim of uniform gain or priced annotation savings.'])

    # Additional manuscript contrasts use the same evidence IDs and full source pins.
    by_id = {item['id']: item for item in findings}
    per_pair = source('mt_fixed_per_pair', 'reports/research/per_pair.csv')
    pt = per_pair.pivot(index=['model_a', 'model_b', 'labeled_fraction'], columns='method', values='mean_absolute_error')
    by_id['E01']['csv_evidence'].append(select('mt_fixed_per_pair', per_pair, {'method': {'in': ['human_only', 'ppi_tuned']}}, list(per_pair.columns)))
    by_id['E01']['derived_values']['pairwise_lower_mae_counts'] = [
        {'labeled_fraction': float(b), 'lower_mae_pairs': int(group.ppi_tuned.lt(group.human_only).sum()), 'pair_denominator': len(group)}
        for b, group in pt.groupby(level='labeled_fraction')]
    signed_empirical = source('signed_llmbar_summary', 'reports/signed-power/llmbar_summary.csv')
    st = signed_empirical.pivot(index=['cohort', 'judge', 'labeled_fraction'], columns='method', values='mean_absolute_error')
    difference = st.ppi_signed - st.ppi_tuned
    ties = np.isclose(difference, 0, rtol=0, atol=1e-12)
    by_id['E05']['csv_evidence'].append(select('signed_llmbar_summary', signed_empirical, {'method': {'in': ['ppi_tuned', 'ppi_signed']}}, ['cohort', 'judge', 'labeled_fraction', 'method', 'mean_absolute_error', 'trials']))
    by_id['E05']['derived_values']['reused_llmbar_comparison'] = {
        'lower_mae_cells': int(((difference < 0) & ~ties).sum()), 'ties': int(ties.sum()),
        'higher_mae_cells': int(((difference > 0) & ~ties).sum()), 'denominator': len(difference),
        'tolerance': 1e-12, 'scope': 'Same historical panel/splits; exploratory repeat analysis, not external replication'}
    by_id['E05']['dependency_and_overlap_lineage']['empirical_extension'] = LLM
    by_id['E05']['dependency_and_overlap_lineage']['dependence'] = 'Primary contrast uses one of 18 synthetic settings with shared method draws. The supplementary 11/27 LLMBar comparison repeats the historical panel and splits; it is exploratory, not external replication. No synthetic and empirical losses are pooled.'
    by_id['E07']['csv_evidence'].append(select('estimand_simulation_summary', est, {
        'scenario': {'eq': 'iid_p095_positive_n0020_r005'}, 'method': {'in': ['population_tuned', 'pool_tuned']}},
        ['scenario', 'method', 'repetitions', 'n_labeled', 'n_unlabeled', 'pool_rmse', 'pool_rmse_mcse', 'pool_coverage', 'pool_coverage_mc_low', 'pool_coverage_mc_high']))
    legacy = source('llmbar_summary', 'reports/llmbar-audit/summary.csv')
    selected = legacy.loc[legacy.cohort.eq('Adversarial') & legacy.judge.isin(['ChatGPT', 'LLaMA2'])]
    lt = selected.pivot(index=['judge', 'labeled_fraction'], columns='method', values='mean_absolute_error')
    by_id['E08']['csv_evidence'].append(select('llmbar_summary', legacy, {'cohort': {'eq': 'Adversarial'}, 'judge': {'in': ['ChatGPT', 'LLaMA2']}}, list(legacy.columns)))
    by_id['E08']['derived_values']['adversarial_fallback'] = {
        'zero_mean_nonnegative_tuned_power_cells': int(selected.loc[selected.method.eq('ppi_tuned'), 'mean_selected_power'].eq(0).sum()),
        'fixed_ppi_higher_mae_cells': int(lt.ppi.gt(lt.human_only).sum()), 'denominator': len(lt),
        'reason_all_splits_zero': 'Every selected power is constrained nonnegative; exact zero mean entails every retained selected power is zero.'}
    pp = primary.loc[primary.target_judge.eq(full) & primary.labeled_fraction.eq(.4)].set_index('method')
    by_id['E10']['derived_values']['gpt4o_budget40_conflicting_loss_metrics'] = {
        'target_judge': full, 'labeled_fraction': .4,
        'reference_audit_mae': float(pp.loc['human_only', 'mean_absolute_error']),
        'audit_residual_mae': float(pp.loc['audit_residual', 'mean_absolute_error']),
        'reference_audit_rmse': float(pp.loc['human_only', 'root_mean_squared_error']),
        'audit_residual_rmse': float(pp.loc['audit_residual', 'root_mean_squared_error'])}
    gains = rt.human_only - rt.ppi_signed
    by_id['E10']['derived_values']['signed_mae_gain_range_vs_audit'] = [float(gains.min()), float(gains.max())]
    by_id['E10']['derived_values']['signed_and_positive_summary_mae_equal'] = bool(rt.ppi_signed.equals(rt.ppi_tuned))

    stress = source('stress_summary', 'reports/stress/summary.csv')
    shifted = source('research_simulation_summary', 'reports/research/simulation_summary.csv')
    common_columns = ['scenario', 'method', 'validity_scope', 'truth', 'repetitions', 'n_labeled', 'n_unlabeled', 'audit_units', 'target_units', 'bias', 'bias_mcse', 'rmse', 'coverage', 'coverage_mcse', 'mean_width', 'mean_selected_power']
    add('E11_dependence_and_shift', 'Grouping and transport are separate requirements',
        'reports/stress/REPORT.md', 'Population outcome mean under an explicit synthetic law; intended-target coverage under correct clusters versus deliberate row-IID misuse, or under deliberate audit-to-target shift.',
        {'cluster_replications': 500, 'independent_audit_clusters': 8, 'repeated_rows_per_cluster': 4,
         'audit_rows': 32, 'prediction_clusters': 80, 'prediction_rows': 320,
         'shift_replications_per_law': 1000, 'shift_audit_rows': 200, 'shift_prediction_rows': 2000},
        [select('stress_summary', stress, {'scenario': {'in': ['cluster_g008_correct', 'cluster_g008_naive_iid']}, 'method': {'eq': 'ppi_tuned'}}, common_columns + ['draw_group', 'inference_mode']),
         select('research_simulation_summary', shifted, {'scenario': {'in': ['prevalence_shift', 'conditional_error_shift']}, 'method': {'eq': 'ppi_tuned'}}, common_columns)],
        {'operation': 'Compare tuned coverage on identical cluster_g008 draws; read tuned bias and intended-target coverage separately for each shift law. No pooling of 500- and 1,000-replication studies.',
         'secondary_report': BASE + 'reports/research/REPORT.md'},
        {'cluster_configuration': 'reports/stress/config.json', 'cluster_seed': 2027,
         'shift_configuration': 'reports/research/simulation_config.json', 'shift_seed': 2026,
         'dependence': 'Grouped and naive variants share identical generated draws, but variance/tuning calculations differ. Shift scenarios are distinct declared laws, not empirical corpora or independent domain replications.'},
        {'type': 'Scenario-specific simulation MCSE; 500 shared cluster draws versus 1,000 draws for each shift law. Zero observed coverage does not imply known exactly-zero coverage probability.'},
        ['Eight clusters do not guarantee a reliable normal approximation even after clustering.',
         'These are intended assumption violations of basic correction, not a refutation of separately designed shift-robust methods.',
         'The source native_coverage field is deliberately not used to redefine the intended target under shift.'])

    return _clean({
        "schema": "research-evidence-crosswalk-v1",
        "repository": "Siquan-Wang/llm-judge-calibration",
        "report_commit": COMMIT,
        "generator": "examples/research_evidence.py",
        "source_policy": "Final CSVs parsed with round_trip precision; independent CSV and historical report-manifest byte pins verified. Links identify the immutable report commit.",
        "units": "Rates, errors and widths are fractions; displayed percentages or percentage points multiply by100. MSE and paired MSE differences are squared fractions.",
        "aggregation_policy": "No pooling across targets, corpora, reference definitions or synthetic laws. Empirical splits are dependent; simulation MC uncertainty is scenario-specific.",
        "selection": "Eleven retrospective headline findings; named illustrative scenarios do not stand for the whole grid. Adverse findings and explicit denominators are retained.",
        "pinned_report_manifests": PINNED_REPORT_MANIFESTS,
        "pinned_lineage_inputs": PINNED_LINEAGE_INPUTS,
        "sources": sources,
        "findings": findings,
    })


def render_markdown(evidence):
    """Compact reading guide; structured selectors/full values remain in JSON."""
    entries = {entry["id"]: entry for entry in evidence["findings"]}
    lines = ["# Reproducible research evidence", "",
             "Eleven retrospective findings, each tied to pinned CSV bytes, an exact selector and a named target. "
             "The [structured crosswalk](research_evidence.json) retains metric names, exact values, denominators, "
             "uncertainty and dataset lineage. These studies do not introduce a new estimator or prove general label savings.", "",
             "Rates/errors below use percentages or percentage points (pp) as labeled. Paired MSE contrasts use squared fractions. "
             "Simulation Monte Carlo uncertainty applies within the declared scenario; fixed-corpus split averages have no iid MCSE.", ""]
    for identifier, entry in entries.items():
        evidence_rows = [row for item in entry["csv_evidence"] for row in item.get("values", [])]
        derived = entry["derived_values"]
        if identifier == "E01":
            rows = entry["csv_evidence"][0]["values"]
            lookup = {(row["labeled_fraction"], row["method"]): row["mean_absolute_error"] for row in rows}
            result = "; ".join(f"{int(100*b)}% budget: {100*lookup[b,'human_only']:.3f} to {100*lookup[b,'ppi_tuned']:.3f} pp" for b in (.2,.4,.6))
            result += ". Tuned MAE is lower at 3/3 budgets versus both reference-only audit and fixed-power PPI."
        elif identifier == "E02":
            result = (f"{derived['changed_scores']}/1,814 comparison scores change; mean absolute score change "
                      f"{100*derived['mean_absolute_score_change']:.3f} pp. Pair-level tuned-minus-audit MAE changes sign "
                      f"in {derived['pair_budget_gap_sign_changes']}/45 cells; neither reference is thereby superior.")
        elif identifier == "E03":
            tuned = next(row for row in evidence_rows if row['method']=='ppi_tuned_normal')
            result = (f"At p=.95,n=20,N=200: tuned normal RMSE {100*tuned['rmse']:.3f} pp, coverage "
                      f"{100*tuned['coverage']:.1f}% (pointwise Monte Carlo interval "
                      f"{100*tuned['coverage_mc_low']:.2f}–{100*tuned['coverage_mc_high']:.2f}%), "
                      f"with {tuned['zero_width_draws']}/1,000 zero-width intervals. Point accuracy does not validate coverage.")
        elif identifier == "E04":
            result = (f"Observed coverage is {100*derived['coverage_min']:.1f}–{100*derived['coverage_max']:.1f}% "
                      f"across 60 finite-theorem-applicable method/scenario cells. Grid EB has smaller untruncated mean width "
                      f"than human Hoeffding in {derived['width_narrower_count']}/{derived['width_denominator']} IID settings.")
        elif identifier in ("E05", "E06"):
            rows = {row['method']:row for row in entry['csv_evidence'][0]['values']}
            result = (f"At p=.5,n=20,N=200, positive-to-signed RMSE changes from "
                      f"{100*rows['ppi_tuned']['rmse']:.3f} to {100*rows['ppi_signed']['rmse']:.3f} pp. "
                      f"Paired signed-minus-positive MSE is {derived['signed_minus_positive_mse']:+.6f} "
                      f"with MCSE {derived['paired_mse_difference_mcse']:.6f} across 1,000 shared-draw replications.")
        elif identifier == "E07":
            rows = {row['method']:row for row in entry['csv_evidence'][0]['values']}
            a,b=rows['population_tuned'],rows['pool_tuned']
            result = (f"At p=.5,n=200,N=100 with a positive proxy: population RMSE "
                      f"{100*a['population_rmse']:.3f} to {100*b['population_rmse']:.3f} pp; "
                      f"realized-pool RMSE {100*a['pool_rmse']:.3f} to {100*b['pool_rmse']:.3f} pp. "
                      "These are two within-target comparisons, with opposite directions.")
        elif identifier == "E08":
            row=evidence_rows[0]
            result = (f"ChatGPT on 319 Adversarial comparisons: original-order reference accuracy "
                      f"{100*row['forward_accuracy']:.1f}%, agreement {100*row['order_agreement']:.1f}%; "
                      f"{row['agree_but_wrong']}/{row['agree_valid_count']} valid agreements choose the reference-incorrect response.")
        elif identifier == "E09":
            rows=sorted(evidence_rows,key=lambda row:row['target_judge'])
            result = (f"On 2,566 NonLLMBar comparisons, agreement is {100*rows[0]['cross_judge_agreement']:.2f}% "
                      f"for both directions; target accuracies are {100*rows[0]['reference_accuracy']:.2f}% and "
                      f"{100*rows[1]['reference_accuracy']:.2f}%. Both judges agree incorrectly on "
                      f"{derived['both_reference_wrong_count']} comparisons.")
        elif identifier == "E10":
            counts=derived['method_counts_vs_reference_only']
            result = ("Lower-MAE cells versus reference-only audit: " + ", ".join(f"{method} {counts[method]['lower_mae_cells']}/6" for method in counts)
                      + f". Adverse examples remain: signed-minus-raw MAE {100*derived['signed_minus_raw_mae_gpt4o_budget20']:+.3f} pp "
                      + f"for GPT-4o at 20%; residual-minus-audit MAE {100*derived['audit_residual_minus_reference_mae_mini_budget40']:+.3f} pp for mini at 40%.")
            reversal = derived['gpt4o_budget40_conflicting_loss_metrics']
            result += (f" For GPT-4o at 40%, residual tuning changes MAE "
                       f"{100*reversal['reference_audit_mae']:.3f} to {100*reversal['audit_residual_mae']:.3f} pp, "
                       f"but RMSE {100*reversal['reference_audit_rmse']:.3f} to {100*reversal['audit_residual_rmse']:.3f} pp.")
        else:
            rows = {row['scenario']: row for row in evidence_rows}
            result = (f"On 500 shared draws with 8 audit clusters repeated 4 times, tuned coverage is "
                      f"{100*rows['cluster_g008_correct']['coverage']:.1f}% with grouped inference and "
                      f"{100*rows['cluster_g008_naive_iid']['coverage']:.1f}% with naive row independence. "
                      f"Prevalence-shift tuned bias/coverage are {rows['prevalence_shift']['bias']:.4f}/"
                      f"{100*rows['prevalence_shift']['coverage']:.1f}%; conditional-error-shift values are "
                      f"{rows['conditional_error_shift']['bias']:.4f}/{100*rows['conditional_error_shift']['coverage']:.1f}% "
                      "across 1,000 replications per shift law. These scenarios are not pooled.")
        lines += [f"## {identifier}: {entry['title']}", "", result, "",
                  f"**Target:** {entry['estimand']}", "",
                  f"[Complete source report]({entry['report_link']}). **Selectors and metrics:**", ""]
        for item in entry['csv_evidence']:
            source=evidence['sources'][item['source_id']]
            selector=json.dumps(item['row_filter'],sort_keys=True,separators=(',',':'))
            lines += [f"- [{source['repository_path']}]({source['immutable_link']}): `{selector}`; "
                      f"{item['selected_rows']} rows; metric columns and exact values in `{identifier}` of the JSON."]
        lines += ["", "**Dependence:** " + entry['dependency_and_overlap_lineage'].get('dependence',entry['dependency_and_overlap_lineage'].get('relation','See structured lineage.')), ""]
    lines += ["## Verify or rebuild", "", "```bash", "python examples/research_evidence.py --check",
              "python examples/research_evidence.py --output-directory docs", "```", "",
              "`--check` verifies inputs and compares both generated files without writing. "
              "Any CSV, cited report/configuration, report-manifest or lineage-input byte mismatch fails; "
              "changing a self-declared sidecar hash cannot bypass the independent pins. Rebuilds are offline and deterministic.", "",
              "Historical report source-code fingerprints describe their original runs, not the current evolving source tree. "
              "Immutable source-report links refer to commit `" + evidence['report_commit'] + "`.", ""]
    return '\n'.join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory',type=Path,default=ROOT/'docs')
    parser.add_argument('--check',action='store_true',help='verify exact existing outputs without mutation')
    args=parser.parse_args(argv)
    evidence=build_evidence()
    output={'research_evidence.json':(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'),
            'RESEARCH_EVIDENCE.md':render_markdown(evidence).encode('utf-8')}
    if args.check:
        for name,payload in output.items():
            path=args.output_directory/name
            if not path.exists() or path.read_bytes()!=payload:
                raise ValueError(f'generated evidence differs: {path}')
        print('Verified 11 findings and both committed evidence artifacts without changes.')
    else:
        args.output_directory.mkdir(parents=True,exist_ok=True)
        for name,payload in output.items():
            (args.output_directory/name).write_bytes(payload)
        print('Wrote 11 source-pinned findings and their compact evidence crosswalk.')


if __name__=='__main__':
    main()
