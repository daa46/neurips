# Frozen HRMS specification

Status: method specification frozen before inspecting application rankings.
The post-freeze public-TE realization is recorded at the end of this document;
it does not amend the estimands or decision rule.

## Scope and assumptions

Let `m in {1,...,M}` index a fixed candidate-configuration library and let
`e in {1,...,E}` index environment types observed during development. The target
distribution is assumed to be a mixture `q` of these represented environment
types. HRMS does **not** claim a guarantee for an arbitrary novel environment.

Every validation observation has a loss for every candidate and belongs to an
independent root cluster. Root clusters, not observations, are the resampling and
train/select/test splitting units. Candidate definitions and all training-only
calibration procedures are frozen before selection losses are inspected.

## Estimands

The primary observation-weighted risk is

    R[m,e] = sum_i w[i] loss[i,m] / sum_i w[i],  for observations i in e.

The cluster-balanced sensitivity estimand first averages within every
environment-by-root-cluster cell and then gives each represented root cluster in
that environment equal weight.

Pairwise differences are `Delta[m,k,e] = R[m,e] - R[k,e]`. Negative values favor
`m`. They are always estimated from paired losses on the same observations.

## Paired simultaneous calibration

HRMS defaults to a delete-root jackknife-pseudovalue bootstrap-t calibration. If
every root cluster belongs to exactly one environment, the resampling blocks are
the environments: roots are treated independently within each block, preserving
the observed root count `C_g`. If any root spans environments, there is instead
one global block containing all `C_g` roots and all environments. Deletion and
resampling are then global, so every occurrence of a spanning root is retained or
removed jointly. The same operations always apply to every candidate.

For root `c` in block `g` and every environment `e` represented by that block,
let `Rhat_minus_c[m,e]` be the risk estimator after deleting that root. Define
the risk and paired-contrast pseudovalues

    Z[c,m,e] = C_g Rhat[m,e] - (C_g - 1) Rhat_minus_c[m,e]
    P[c,m,k,e] = Z[c,m,e] - Z[c,k,e].

For a given coordinate, the original jackknife standard error is

    s[m,k,e] = sd_c(P[c,m,k,e]) / sqrt(C_g).

The pseudovalues are centered coordinatewise. In bootstrap replicate `b`, HRMS
draws `C_g` root rows with replacement within every block, computes their mean
centered contrast `Pbar_b[m,k,e]`, and recomputes the replicate-specific standard
error `s_b[m,k,e]` from those same sampled rows. The replicate contributes

    T_b = max_(m<k,e: s[m,k,e]>0) abs(Pbar_b[m,k,e] / s_b[m,k,e]),

where the maximum is also taken jointly across resampling blocks. A positive-
scale coordinate with zero replicate scale and nonzero numerator contributes
infinity. The critical value is the empirical `(1-alpha)` quantile (using the
higher order statistic) of the `T_b` values. The default bounds are centered at
the original paired risk estimate:

    L[m,k,e] = max(loss_min-loss_max, Delta_hat[m,k,e] - c s[m,k,e])
    U[m,k,e] = min(loss_max-loss_min, Delta_hat[m,k,e] + c s[m,k,e]).

Thus Brier-loss contrasts are clipped to `[-1,1]`. Only candidates declared to
be structural duplicates before selection inference may have their contrasts
fixed to zero, and `fit` verifies exact equality of their observation-wise loss
vectors before accepting the declaration. Observed equality alone never marks
an undeclared pair as structural. A non-duplicate coordinate with zero
jackknife standard error receives the full bounded-loss contrast interval and
triggers a diagnostic.

The upper and lower arrays are constructed for `m<k` and extended by exact
antisymmetry: `L[k,m,e] = -U[m,k,e]`, `U[k,m,e] = -L[m,k,e]`.

Raw root-cluster bootstrap-t (`raw_bootstrap_t`, with replicate-specific
cluster-sandwich standard errors) and the ordinary fixed-denominator bootstrap
max-t (`fixed_max_t`) are retained only as prespecified calibration ablations.
Raw root-cluster replicates are also used for the selection-stability diagnostic:
the reported value is the frequency with which each candidate minimizes
point-estimated robust regret in those raw replicates. It is not a nested HRMS
refit, confidence level, or posterior probability, and the raw replicates are not
the default interval envelope. A global raw replicate that omits all roots
contributing to an environment is undefined and raises a diagnostic rather than
being redrawn or discarded.
The fitted result exposes the exact choice as `result.calibration_method`; the
default calibration API is `jackknife_bootstrap_t` and its reported method string
is `jackknife-pseudovalue cluster bootstrap-t`.

## Mixture uncertainty and regret

For a strictly positive nominal mixture `p`,

    Q_rho(p) = {q >= 0: sum(q)=1, sum_e (q[e]-p[e])^2 / p[e] <= rho}.

For a target mixture `q`, candidate `m` has library regret

    G[m,q] = max_k sum_e q[e] Delta[m,k,e].

Its robust regret is `Gamma[m] = max_{q in Q} G[m,q]`. Given simultaneous
componentwise bounds, HRMS reports

    lower[m] = max_k max_{q in Q} q dot L[m,k,:]
    upper[m] = max_k max_{q in Q} q dot U[m,k,:].

The default model minimizes `upper[m]`. If upper certificates tie within the
declared numerical tolerance, HRMS minimizes point-estimated robust regret among
the tied candidates, then takes the first canonical model identifier (identifiers
are ordered by Python type name and string representation). The robust model
confidence set is

    C = {m: lower[m] <= min_j upper[j]}.

## Conditional guarantees

On the joint event `L <= Delta <= U` for all pairs and environments:

1. `lower[m] <= Gamma[m] <= upper[m]` for every candidate.
2. Every truly robust-optimal candidate is in `C`; every candidate outside `C`
   is strictly suboptimal for the declared ambiguity set.
3. The minimum-upper-bound candidate belongs to `C`.

These statements are deterministic conditional on simultaneous coverage. They
do not assert finite-sample exactness of the cluster bootstrap. Calibration,
confidence-set coverage, and small-cluster behavior must be evaluated in
simulation.

If `m*` minimizes true robust regret and
`eta[m*] = max_(q,k) sum_e q[e](U[m*,k,e]-Delta[m*,k,e])`, the selected model
satisfies `Gamma[selected] <= Gamma[m*] + eta[m*]` on the joint event. A `+epsilon`
statement therefore requires `epsilon` to bound the *upper slack*, not merely the
half-width of a symmetric interval.

For every fixed mixture `q`, at least one candidate has zero regret. Under mixture
uncertainty and rank reversals, however, the minimum robust regret across fixed
candidates can be strictly positive.

## Required governance

- Split at the highest independent root cluster before fitting any model.
- Fit, tune, early-stop, and calibrate candidates using training data only.
- Freeze model identifiers, hyperparameters, seeds, checkpoint hashes, and
  training-data hashes before reading selection rankings.
- Use selection losses once for HRMS and untouched test losses once for final
  evaluation.
- Never tune the ambiguity radius on test labels.

## Immutable file interface

The statistical package receives a long selection-loss table with exactly one
row per `(observation_id, model_id)` and the columns `environment_id`,
`root_cluster_id`, `loss`, and optional positive `sample_weight`. Every
observation must contain the complete frozen model library. Loss tables for the
untouched test split use the identical schema but are never passed to `fit`.

The candidate registry records `model_id`, model family, representation,
pooling, regularization, seed, training-only calibration method, checkpoint
content hash, and training-data content hash. A registry with placeholders is a
planning document, not a frozen library.

Root-cluster split exports record `observation_id`, `root_cluster_id`, `split`,
`split_seed`, and a canonical `split_hash`. The split hash is a SHA-256 digest of
the deduplicated root-cluster-to-split mapping and is invariant to row order.

Each command-line run writes a result manifest containing input/output SHA-256
digests, numerical-software versions, and all selection parameters. Generated
paper values must be traceable to one such manifest.

## Primary analysis defaults

- Brier loss, with probabilities calibrated on training-only data.
- Uniform nominal environment mixture.
- `alpha=0.05`; 1,999 cluster bootstrap replicates in the frozen application
  runs (2,000 was the planning target).
- Radius path `{0, 0.1, 0.25, 0.5, 1.0, full simplex}`.
- Observation-weighted risks primary; cluster-balanced risks a prespecified
  sensitivity analysis.
- Distribution shift may change only the prevalence of represented environments;
  within-environment loss distributions are assumed stable.

## Computational complexity

Let `P=M(M-1)/2` and let `H=sum_g C_g |E_g|`, where `E_g` is the set of
environments in resampling block `g`. Thus `H=sum_e C_e` for environment-nested
roots and `H=C E` for one global spanning-root block. After `O(NM)` loss
aggregation, the current vectorized default calibration costs `O(B H P)` time
and `O(H P + B E P)` working memory (the latter includes stored raw-bootstrap
pairwise replicates used for stability). For each ambiguity radius, the robust
certificates require `O(M^2 E^2)` worst-case time with the current active-set
chi-squared support solver and `O(E)` working memory per support query.

## Post-freeze public-TE realization

The completed application predicts association between an hg38 TE locus and an
ENCODE SCREEN V4 pELS/dELS cCRE. This is a derived biochemical-annotation
benchmark. A positive is not necessarily a causal regulatory TE copy, and a
negative is not-cCRE-annotated rather than proven inactive.

The performance-blind homology gate excluded `Alu`, `ERV1`, `ERVL-MaLR`, and
`hAT-Charlie` and retained the represented families `ERVL`, `L1`, `L2`, and
`MIR`. The final cohort contains 22,590 loci in 11,295 matched positive/control
pairs and 10,920 transitive roots. Roots are the union of 80%-identity,
80%-bidirectional-coverage MMseqs2 threshold-graph components for full oriented
TE copies and 512-bp oriented input windows, plus matched-design-pair edges.
The 60/20/20 split contains 13,702/4,450/4,438 train/select/test observations
and 6,552/2,184/2,184 roots. The strict under-5% root-dominance gate applies to
selection and test, the partitions used for selection uncertainty and final
evaluation; all four retained environments pass it and have both labels and at
least 20 roots in each held-out partition. No root, TE component, window
component, or design pair crosses a split.

The frozen library has 24 HRMS-eligible CPU candidate configurations and two
separate diagnostic-only controls. Selection used observation-weighted Brier
loss, `alpha=0.05`, 1,999 root-bootstrap replicates, seed 42, and a uniform
chi-squared ball with default radius `rho=0.25`. It selected
`lr_w128_k6_c0p1`; the upper robust-regret certificate is 0.012609 and the
confidence set contains all 24 candidates. The finite held-out test reference
has robust regret 0.001383 at that radius. All five primary selection rules
chose the same configuration, while the test-optimal configuration differs by
TE family. These outcomes support rank variation and non-identification of a
unique robust winner; they do not establish an application-level regret
advantage for HRMS.

The run is anchored by the following local content hashes:

| Artifact | SHA-256 |
| --- | --- |
| Performance-blind gate evidence, `data/te_public/gate_evidence_combined_v2.csv` | `76b4903f84ce92026bf5e625700c2a06d59ba03dba580b0581799f29138a2dfe` |
| Final cohort, `data/te_public/final_clustered_v2/labeled_loci_split.parquet` | `5de6f64300211d22965cb51beb68ac28a11eecb58299f4e3a45a4b239b59fb08` |
| Cluster/split manifest, `data/te_public/final_clustered_v2/cluster_split_manifest.json` | `5627592b8c8ae5366fae7bcb48e882828f50b65d6cee2c354aa00d36dd097ddb` |
| Split table, `data/te_public/final_clustered_v2/splits.parquet` | `09672bb4cd1bb43fc24a87b9857ec23246f56c228188e8a10139b1dcfc9a26d2` |
| Candidate-training manifest, `results/te_public/training/training_manifest.json` | `91d5db064e9114feabd5ddc3f5142b9b3d3b61977b9376804aefa20f8a8aae7a` |
| Sealed-selection manifest, `results/te_public/evaluation/selection/selection_manifest.json` | `67b642fe412b9ce575f09978b29d2bf245a87c75e8ca154ae2bf37225a332ac4` |
| Test-evaluation manifest, `results/te_public/evaluation/test_evaluation/evaluation_manifest.json` | `7badc1cc689009bd44225ade891e408b161ec4bc1d9b315ea4be2d1897e569df` |

These are local working-tree artifacts, not a claim that a public archival copy
or DOI already exists.
