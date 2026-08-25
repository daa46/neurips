# HRMS anonymous artifact candidate

This local artifact candidate accompanies *Reliable Model Selection Under
Hierarchical Dependence and Unknown Environment Mixtures*. The anonymous bundle
contains the HRMS package, frozen simulation configurations, tests, paper
source, aggregate application results, content-addressed manifests, and the
self-contained final clustered TE cohort. It does **not** redistribute
CivilComments text, the public ENCODE/UCSC source downloads, or row-level test
prediction/loss tables. No public archive URL or DOI is claimed at this stage.

## Environment and fast verification

Python 3.11 or newer and `uv` are required.

```bash
uv sync --locked --extra dev --extra paper
make reproduce PYTHON=.venv/bin/python OUTPUT_DIR=results/reproduction
```

The reproduction target runs the complete test and lint suite, validates a
synthetic frozen registry, exercises HRMS end to end, renders smoke figures,
and writes a nested SHA-256 manifest. The smoke study verifies integration; it
is intentionally smaller than the paper experiments.

Build and independently verify the deterministic submission archive with:

```bash
.venv/bin/python scripts/package_anonymous_artifact.py
```

The packager uses an explicit allowlist, checks its internal SHA-256 manifest,
extracts and imports the package in a temporary directory, and rejects local
identity or home-directory strings.

## Frozen headline simulations

The executable configurations reject conflicting statistical command-line
overrides. The decision study supports deterministic process-level parallelism;
changing `--jobs` does not alter seeds or row order.

```bash
.venv/bin/python simulations/coverage_study.py \
  --config configs/simulation/coverage_headline.json \
  --output-dir results/coverage_final_paired

.venv/bin/python scripts/write_manifest.py \
  --root results/coverage_final_paired \
  --output results/coverage_final_paired/manifest.json \
  --input frozen_config=configs/simulation/coverage_headline.json

.venv/bin/python simulations/decision_study.py \
  --config configs/simulation/decision_headline.json \
  --output-dir results/decision_headline --jobs 6

.venv/bin/python scripts/make_simulation_figure.py \
  --coverage-summary results/coverage_final_paired/coverage_summary.csv \
  --decision-summary results/decision_headline/decision_summary.csv \
  --output-stem paper/figures/simulation
```

The local high-resolution result trees include the full per-repetition CSVs,
summaries, effective configurations, seed rules, and hashes. The coverage study
uses 500 repetitions for each of four ICC settings and 999 bootstrap
replicates. The decision study uses 500 repetitions for each of six
rank-reversal severities and 999 bootstrap replicates. On the reported
eight-core CPU, the serial coverage command takes about 70 minutes; the
six-worker decision command takes about eight minutes.

## Public TE--cCRE experiment

This is a derived TE-associated enhancer-like cCRE prediction task. A positive
hg38 TE locus overlaps at least half of one ENCODE SCREEN V4 pELS/dELS cCRE; a
matched negative has no cCRE of any class within 250 bp. These are annotation
labels, not evidence that positives are causal regulators or that negatives are
biologically inactive.

The predeclared, performance-blind homology gate retained four represented
RepeatMasker families (`ERVL`, `L1`, `L2`, and `MIR`) and excluded `Alu`,
`ERV1`, `ERVL-MaLR`, and `hAT-Charlie`. The final cohort has 22,590 loci,
11,295 matched pairs, and 10,920 roots. The 60/20/20 root split has
13,702/4,450/4,438 train/select/test observations and
6,552/2,184/2,184 roots. All retained selection and test cells have both labels,
at least 20 roots, and largest-root fraction strictly below 5%; no homology
root, full-TE component, 512-bp-window component, or matched design pair crosses
a split.

The exact local run is anchored by these hashes:

| Artifact | SHA-256 |
| --- | --- |
| Frozen task config, `configs/te/public_ccre.json` | `e0ac97cfb5531438a0ada9737fea14a56db8846280419623408298ccab00a23f` |
| Gate evidence, `data/te_public/gate_evidence_combined_v2.csv` | `76b4903f84ce92026bf5e625700c2a06d59ba03dba580b0581799f29138a2dfe` |
| Frozen cohort, `data/te_public/final_clustered_v2/labeled_loci_split.parquet` | `5de6f64300211d22965cb51beb68ac28a11eecb58299f4e3a45a4b239b59fb08` |
| Cluster/split manifest, `data/te_public/final_clustered_v2/cluster_split_manifest.json` | `5627592b8c8ae5366fae7bcb48e882828f50b65d6cee2c354aa00d36dd097ddb` |
| Split table, `data/te_public/final_clustered_v2/splits.parquet` | `09672bb4cd1bb43fc24a87b9857ec23246f56c228188e8a10139b1dcfc9a26d2` |
| Candidate config, `configs/te/candidates.public.json` | `9459211536274fda8abdeb51dbd5b8da9c5baf2d7ce502432ff47915b98cee10` |
| Training manifest, `results/te_public/training/training_manifest.json` | `91d5db064e9114feabd5ddc3f5142b9b3d3b61977b9376804aefa20f8a8aae7a` |
| Frozen registry, `results/te_public/training/model_registry.csv` | `bc00df37a19ab82bda085f4602bd4789b1597021abc4cde9be10936d3265067a` |
| Sealed-selection manifest, `results/te_public/evaluation/selection/selection_manifest.json` | `67b642fe412b9ce575f09978b29d2bf245a87c75e8ca154ae2bf37225a332ac4` |
| Test-evaluation manifest, `results/te_public/evaluation/test_evaluation/evaluation_manifest.json` | `7badc1cc689009bd44225ade891e408b161ec4bc1d9b315ea4be2d1897e569df` |
| Descriptive-metrics manifest, `results/te_public/evaluation/descriptive_metrics/descriptive_metrics_manifest.json` | `073ca903090fd6b0336014413c1dd49ae855b62ce440573df86ef45f608c19db` |

Given the frozen clustered cohort, the following commands reproduce candidate
training, persist selection before the evaluator first semantically reads or
aggregates the already-exported test-loss table, compute reporting-only metrics,
and render the paper figure. Run them from a clean working tree if the
sealed output directories already exist.

```bash
.venv/bin/python -B applications/te_regulation/train_public_candidates.py \
  --cohort data/te_public/final_clustered_v2/labeled_loci_split.parquet \
  --config configs/te/candidates.public.json \
  --output-dir results/te_public/training

.venv/bin/python -B applications/te_regulation/evaluate_public.py \
  --selection-losses results/te_public/training/losses_select.parquet \
  --test-losses results/te_public/training/losses_test.parquet \
  --registry results/te_public/training/model_registry.csv \
  --training-manifest results/te_public/training/training_manifest.json \
  --output-dir results/te_public/evaluation \
  --radius 0.25 --alpha 0.05 --bootstrap 1999 --seed 42 --mixtures 512

.venv/bin/python -B applications/te_regulation/report_public_metrics.py \
  --predictions-select results/te_public/training/predictions_select.parquet \
  --predictions-test results/te_public/training/predictions_test.parquet \
  --registry results/te_public/training/model_registry.csv \
  --training-manifest results/te_public/training/training_manifest.json \
  --evaluation-manifest \
    results/te_public/evaluation/test_evaluation/evaluation_manifest.json \
  --output-dir results/te_public/evaluation/descriptive_metrics \
  --diagnostic-registry \
    results/te_public/training/diagnostic_model_registry.csv \
  --diagnostic-predictions-select \
    results/te_public/training/diagnostic_predictions_select.parquet \
  --diagnostic-predictions-test \
    results/te_public/training/diagnostic_predictions_test.parquet

MPLCONFIGDIR=/tmp/hrms-matplotlib-cache \
  .venv/bin/python -B scripts/make_te_figure.py \
  --output-stem paper/figures/te_public
```

The 24-candidate primary analysis selected `lr_w128_k6_c0p1` at every reported
radius. At `rho=0.25`, its simultaneous upper robust-regret certificate is
0.012609, raw root-bootstrap point-selection stability is 0.894, and the model
confidence set contains all 24 candidates. Its finite held-out test robust
regret is 0.001383; the test-optimal fixed configuration at that radius is
`lr_w128_k6_c1`. All five primary selectors chose the same configuration, but
the lowest-Brier candidate differs across the four held-out TE families. Thus
the application documents rank variation and insufficient evidence for a
unique winner; it does not show an application-level regret advantage for HRMS.

Raw-to-cohort reconstruction uses the pinned public inputs and staged CLIs under
`applications/te_regulation/`; the frozen protocol and source hashes are in
`notes/PUBLIC_TE_BENCHMARK.md` and `configs/te/public_ccre.json`. The combined
gate-evidence file is a required performance-blind input to the final extraction
and records the failed pilot/production gates. The generated anonymous archive
explicitly includes the final aggregate result tables, manifests, gate evidence,
and frozen clustered cohort even though `data/te_public/` and `results/` are
ignored by Git. These local materials are not yet represented by a public
archival record.

## CivilComments-derived experiment

CivilComments is used under CC0-1.0 via the exact source commit and SHA-256 in
the bundled manifests. The preparation script creates disjoint exact-one-
identity environments and carries `article_id` as the highest available root
identifier. Candidate fitting and Platt calibration split official training
data at the article-root level. Validation selection is sealed before the test
evaluator accepts labels.

The local working tree provides aggregate candidate metrics, the frozen
registry, training timings, selection outputs, sealed-test summaries, and the
IID-comment-root sensitivity analysis. Observation-level comment text,
predictions, and loss tables are excluded. To reconstruct them from a licensed
local source copy, run the CLIs in this order and inspect each `--help` page for
paths:

```bash
.venv/bin/python applications/civilcomments/prepare_data.py --help
.venv/bin/python applications/civilcomments/train_candidates.py --help
.venv/bin/python applications/civilcomments/select_models.py --help
.venv/bin/python applications/civilcomments/evaluate.py --help
.venv/bin/python applications/civilcomments/ablate_iid_roots.py --help
```

Primary selection uses 1,999 global article-root bootstrap replicates,
`alpha=0.05`, and `rho=0.25`. Test labels must not be supplied to the evaluator
until the selection manifest exists and records `test_labels_accessed=false`.

## Paper build

The manuscript uses the official NeurIPS 2026 workshop style in anonymous mode.
From `paper/`, compile with a recent Tectonic or a standard XeLaTeX/BibTeX
installation. The checked PDF is US Letter, has embedded fonts, contains eight
pages of main text, and places references and the official checklist afterward.

## Scope

The theorem is deterministic conditional on a simultaneous pairwise coverage
event. Bootstrap calibration is empirical rather than finite-sample exact. The
target must be a mixture of represented environments, the candidate library
must be frozen, and article-level independence in CivilComments remains an
assumption. See `SPEC.md` and Section 6 of the paper for the full limitations.
