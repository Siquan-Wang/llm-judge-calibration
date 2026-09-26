# Fixed finite-corpus accuracy: retrospective protocol

This study estimates either cached judge's recorded-reference accuracy over a
complete fixed corpus after labeling a uniform sample of prompt groups. Its
confidence target is the **full corpus's equally weighted comparison mean**.
It is separate from the earlier held-out point-error audit and the package's
population-mean and random-pool prediction intervals.

The design is retrospective, informed by source inspection and previous studies,
and is not preregistered. The configuration and implementation are frozen before
the final full study. This protocol specifies no observed findings. It applies
classical difference estimation and established finite-population concentration
bounds; it introduces no new estimator principle or coverage theorem.

## Fixed data and outcome

Use the committed [RewardBench labels](../reports/rewardbench/labels.csv) and
[provenance](../reports/rewardbench/dataset.json), pinned by SHA256:

```text
labels.csv
7dc7d8aad832f470a2831df5011fd7e8e1c670996a225b5d91c81f760350120f
dataset.json
3ff9bd62f5a939c645b1a18e75cacf83bbe4e37a3d6e26747980894f993062f2
```

Keep only `NonLLMBar`, defined by excluding source subset names beginning with
`llmbar-`. Retain all 2,566 comparisons in its 2,315 exact prompt groups and all
18 remaining source subsets. Analyze both designated judges:

- `openai/gpt-4o-2024-08-06`
- `openai/gpt-4o-mini-2024-07-18`

The same comparison can appear once per judge in the long input; this is a
two-judge panel, not 5,132 distinct comparisons. For target judge `j`, the binary
outcome is `Y_ij = 1{canonical_choice_ij == reference_choice_i}`. The agreement
proxy is `F_i = 1{canonical_choice_iA == canonical_choice_iB}`, the same in both
directions. Compute it from choices and validate any redundant stored fields.
It contains no reference orientation, and agreement need not imply correctness.

The [RewardBench protocol](REWARDBENCH_PROTOCOL.md) records the exact upstream
data and cache revisions, full-content joins, binary-cache restriction, and
reference semantics. The references combine response-quality, instruction,
safety, code and mathematics criteria; they are not uniformly human votes or
latent truth. The target is not the weighted official leaderboard score.
The dated caches are reused without new model calls or annotation. The existing
[source notice](../reports/rewardbench/NOTICE.md) continues to apply: public
availability and derived numeric outputs do not establish a new license for
the original cached judgments. This study does not redistribute source text.

## Sampling design and identity

Write `G=2315`, `M=2566`, and `n_g` for each known group's row count. Uniformly
sample exactly `k` groups without replacement and reveal all their outcomes:

| Nominal fraction of all groups | Audit groups `floor(b G)` |
| --- | ---: |
| 0.20 | 463 |
| 0.60 | 1,389 |
| 0.90 | 2,083 |

There is no held-out evaluation partition. All corpus proxies and memberships
are known; unsampled references are available only to the retrospective scorer,
not to any estimator or its coefficient selection. The realized row-label cost
`sum_{g in S} n_g` varies with the sampled groups. These fractions are group
budgets, not row budgets, annotation times or dollar costs. Stopping after a
variable number of groups to reach a row budget would change the design.

Generate 2,000 replication permutations with master seed `2032`. Replication
`r` uses `numpy.random.Philox(numpy.random.SeedSequence([2032, r]))` through a
NumPy generator, with `r=0,...,1999`, to permute lexicographically sorted group
IDs. Use that same full permutation for both judges, all four methods and all
three budgets. Prefixes make budgets nested within a replication. Separate
replication streams represent independent random design draws; sharing within
a replication is deliberate pairing. Save the NumPy version and RNG contract.

The frame catalog has the exact structure

```text
{
  "schema_version": 1,
  "row_ids": [lexicographically sorted comparison IDs],
  "groups": [
    {"instruction_id": ID, "row_indices": [sorted row-catalog indices]}, ...
  ]
}
```

Sort the group records lexicographically by `instruction_id`. The frame SHA256
hashes UTF-8 bytes of `json.dumps(catalog, sort_keys=True, separators=(',', ':'),
ensure_ascii=True)` with no trailing newline. It includes only structural IDs
and membership, not gold outcomes or proxies. The source-input hashes separately
pin those values. The frame hash identifies the inputs; it is not an extra seed
component or an outcome-dependent choice of seed.

Save one complete group-index permutation per replication, together with its
replication number and frame hash. For each replication/budget, also save
SHA256 hashes of selected group IDs and selected row IDs, each encoded as
lexicographically sorted IDs joined by newline with a terminal newline. This
permits exact sample replay without repeating long IDs in every trial.

## Estimator and target

For either proxy definition below, let `F_g=sum_{i in g} F_i` and
`Y_g=sum_{i in g} Y_i`. The target and candidate difference estimator are

```text
mu_C = sum_g Y_g / M
R_g(lambda) = Y_g - lambda F_g
theta_hat(lambda) = lambda sum_g F_g/M + G/(k M) sum_{g in S} R_g(lambda)
theta_hat(lambda) - mu_C = (G/M) [mean_S R(lambda) - mean_frame R(lambda)].
```

For each fixed coefficient, inclusion probability `k/G` makes this point
estimate design-unbiased. Its zero-coefficient case is the Horvitz–Thompson
expansion of sampled group totals. With unequal group sizes, it is not the
sampled-row ratio `sum_S Y_g/sum_S n_g` or an equal average of group means.
The correction uses the full known proxy total, including sampled groups;
there is no independent prediction pool and no prediction-pool error term.

There are four prespecified methods:

| ID | Coefficient and auxiliary information | Interval family |
| --- | --- | --- |
| `ht_hs` | Fixed zero; no auxiliary proxy | Hoeffding–Serfling |
| `ht_ebs` | Fixed zero; no auxiliary proxy | Empirical Bernstein–Serfling |
| `agreement_grid_ebs` | Choice agreement; grid `(0, .25, .5, .75, 1)` | Simultaneous empirical Bernstein–Serfling grid |
| `size_grid_ebs` | Constant row proxy `F_i=1`, hence `F_g=n_g`; same grid | Simultaneous empirical Bernstein–Serfling grid |

The size control uses no auxiliary judge or agreement information. Its known
total is `M`, and its correction adjusts for the sampled groups' realized size
imbalance. It still evaluates the designated judge's correctness outcome, so
that outcome requires the judge's cached choice. This control can distinguish
benefit from known group size from benefit attributable to agreement; it does
not imply a universal decomposition of those effects.

For each grid method, select the smallest **untruncated** radius, breaking an
exact tie toward the smaller coefficient. Both grids are fixed before sampled
references are inspected. They are separate methods: do not select whichever
proxy family or bound family happens to return the narrower interval. An
outcome-selected grid point need not be unbiased even though each fixed point
is design-unbiased. The simultaneous interval guarantee is a different claim.

## Finite-population bounds and allocation

Use Bardenet and Maillard (2015), *Concentration inequalities for sampling
without replacement*, Bernoulli 21(3), 1361–1385,
[DOI 10.3150/14-BEJ605](https://doi.org/10.3150/14-BEJ605), pinned
[arXiv v2](https://arxiv.org/abs/1309.4029v2). Source locations below use the
electronic reprint's pages, which differ from journal pagination.

Known frame information gives `R_g(lambda) in [-lambda F_g, n_g-lambda F_g]`.
Define the common enclosing range and finite-population correction by

```text
a_lambda = min_g (-lambda F_g)
b_lambda = max_g (n_g - lambda F_g)
L_lambda = b_lambda - a_lambda

rho(k,G) = 1-(k-1)/G                 if k <= G/2
           (1-k/G)(1+1/k)           if k > G/2.

v_hat_lambda = sum_{g in S} [R_g(lambda)-mean_S R(lambda)]^2 / k.
```

Use group count `k` and group residual totals, not a row-level variance or the
number of sampled rows. The support uses the whole known frame; an observed
sample range is not a substitute. Empirical variance uses **`ddof=0`**, following
[Eq. (26), reprint p. 19](https://arxiv.org/pdf/1309.4029v2#page=19).

At default `alpha=.05`, a method with `K` prespecified candidate coefficients
has the following two-sided radius:

```text
Hoeffding–Serfling:
  t_H = log(2 K / alpha)
  r_H(lambda) = (G/M) L_lambda sqrt[rho(k,G) t_H/(2 k)]

Empirical Bernstein–Serfling:
  t_E = log(10 K / alpha)
  kappa = 7/3 + 3/sqrt(2)
  r_E(lambda) = (G/M) {
      sqrt[2 rho(k,G) v_hat_lambda t_E/k]
      + kappa L_lambda t_E/k
  }.
```

[Corollary 2.5, p. 10](https://arxiv.org/pdf/1309.4029v2#page=10) bounds one
tail with failure `delta`. Apply it to both residual signs with
`delta=alpha/(2K)`. [Theorem 4.3, p. 21](https://arxiv.org/pdf/1309.4029v2#page=21)
bounds one tail with failure `5 delta`; assigning `delta=alpha/(10K)` yields
the empirical-Bernstein allocation. The latter constant is confirmed by
[Eqs. (31)–(32), p. 23](https://arxiv.org/pdf/1309.4029v2#page=23).

The baselines use `K=1`; each grid uses `K=5`. A union bound makes all candidate
intervals in one declared method cover simultaneously with probability at
least `1-alpha`, conditional on the fixed corpus and under the stated sampling
design. Any selection within that grid preserves interval coverage. The two
grid methods each have their own marginal guarantee; there is no claim of
joint 95% coverage across methods, judges or budgets, and no cross-family
selector. Selecting an unrestricted continuous coefficient or adding candidates
after examining outcomes is not justified by this allocation.

The common support always has `L_lambda >= max_g n_g = L_0`. For nonnegative
coefficients, `L_lambda = max_g[n_g + lambda(max_h F_h-F_g)]`. Thus a
radius-minimizing range-only grid containing zero cannot beat its zero
candidate at the same allocation. This is why no tuned Hoeffding grid is
included. A smaller residual variance can help the empirical-Bernstein radius,
but the range term and grid penalty can offset it. No efficiency gain is assumed.

## Census and interpretation

Report a separate census check for both judges and all four methods: eight
rows with `k=G`, all `M` references observed, the exact corpus mean and zero
radius. Compute the mean directly. Although `rho(G,G)=0`, the displayed
empirical-Bernstein range term remains positive; returning radius zero is an
exact-data special case, not a property of that formula. Do not mix the census
rows into the 2,000-replication summaries.

These are sampling-design confidence intervals for the fixed recorded corpus.
No within-group independence or iid superpopulation is needed. Uniform
fixed-size group sampling, correct complete membership and complete outcomes
for every selected group are essential. Within-frame outcome dependence,
including crossed dependence, does not violate this sampling-design argument.
The design does not address
nonuniform selection, missing labels, outcome-dependent groups, deployment
shift or erroneous references. It establishes no conditional guarantee for a
remaining held-out pool, annotation savings, current-model performance, or
accuracy beyond this corpus.

All methods receive the same references for a given replication/budget. Gold
references are shared across designated judges rather than counted as newly
acquired twice. The agreement method uses the two complete choice caches;
the baselines and size control need the designated choices for sampled outcomes
but no auxiliary full-corpus choices for their formulas. Cached-input counts,
scoring-only reference access and new calls are separate quantities. No new
model call is made. Label counts here are retrospective information budgets,
not demonstrated acquisition cost reductions.

## Outputs and checks

Run the fixed study with

```bash
python examples/finite_corpus_study.py --output reports/reproduced/finite-corpus --plot
```

The canonical report is [reports/finite-corpus/REPORT.md](../reports/finite-corpus/REPORT.md).
At defaults retain all `2000 * 3 * 2 * 4 = 48000` trial rows, 24 method/cell
summaries, and 36 wide paired-contrast rows covering all six unordered method
pairs in each of six judge/budget cells. Paired differences use the shared
replications; record the subtraction direction and Monte Carlo uncertainty.
Do not interpret separate method MCSEs as the uncertainty of their difference.

Export untruncated points, endpoints and widths, selected coefficients, actual
label counts, and every candidate's coefficient, radius, point, range and
empirical residual variance. Keep the frame catalog, one full permutation per
replication, split hashes, exact configuration, source/output fingerprints,
software versions, separate census table, deterministic report and plots.

For numerical coverage scoring only, use
`low-tol <= truth <= high+tol`, with
`tol=8*float64_epsilon*max(1,abs(low),abs(high),abs(truth))`, consistently across
all methods. This is a floating-point membership convention, not an inferential
correction. Saved raw endpoints and truth permit strict-membership reconstruction;
points, radii, widths and losses are unchanged. Report any changed membership
counts when validating final results.

Report coverage with Monte Carlo uncertainty, including pointwise exact
binomial intervals at observed coverage zero or one. Repeated random designs
provide Monte Carlo replication conditional on one corpus; they do not provide
2,000 independently sampled benchmarks. Preserve all draws and report all
prespecified cells, including unfavorable results. Verify the group-to-row
scaling, both correction branches, `ddof=0`, finite-grid allocation and census
identity independently. These implementation checks accompany, and do not
replace, the established finite-population theorem.
