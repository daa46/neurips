# HRMS: Hierarchical Robust Model Selection

HRMS selects from a **frozen library** of probabilistic models when validation
observations are clustered, performance varies across represented environments,
and the future mixture of those environments is unknown. It reports both a
default model and a confidence set when the available independent evidence does
not justify a unique winner.

The statistical pipeline is:

1. estimate paired environment-specific loss differences;
2. resample independent root clusters while preserving the pairing across models;
3. construct one simultaneous max-statistic envelope over every pair and environment;
4. optimize the resulting lower and upper regret surfaces over a chi-square mixture ball;
5. return the minimum-upper-certificate model and all candidates not excluded from
   robust optimality.

## Quick start

```python
import pandas as pd

from hrms import HRMS
from hrms.ambiguity import ChiSquareBall

losses_select = pd.read_csv("examples/synthetic_losses.csv")
selector = HRMS(
    environment_col="environment_id",
    cluster_col="root_cluster_id",
    observation_col="observation_id",
    model_col="model_id",
    loss_col="loss",
    ambiguity=ChiSquareBall(center="uniform", radius=0.25),
    calibration="jackknife_bootstrap_t",
    n_bootstrap=2_000,
    random_state=42,
)

result = selector.fit(losses_select)
print(result.selected_model)
print(result.model_confidence_set)
```

The long-form loss table must contain exactly one row per observation and model.
Every observation must have predictions from the same registered candidate
library. See [SPEC.md](SPEC.md) for estimands, guarantees, and data governance.

## Install and reproduce

Python 3.11 or newer is required. The lockfile contains the exact environment;
the package itself has only NumPy, pandas, and SciPy as runtime dependencies.

```bash
uv sync --locked --extra dev --extra paper
make reproduce PYTHON=.venv/bin/python
```

The second command runs all tests and lint checks, regenerates the committed
synthetic input, runs HRMS, executes the lightweight simulation grid, renders
descriptive figures, and writes `results/reproduction/manifest.json` with a
SHA-256 digest for every input and output. It normally completes in seconds; it
is an integration check, not the high-resolution paper experiment. Override
`BOOTSTRAP` and `REPETITIONS` for a larger run.

Individual targets are also available:

```bash
make test PYTHON=.venv/bin/python
make example PYTHON=.venv/bin/python
make smoke PYTHON=.venv/bin/python
make figures PYTHON=.venv/bin/python
```

To run an arbitrary table from a repository checkout without installing HRMS:

```bash
.venv/bin/python scripts/run_hrms.py path/to/losses_select.csv \
  --model-registry path/to/model_registry.csv \
  --output results/my_analysis --bootstrap 2000 --seed 42
```

The output directory contains environment risks, tidy pairwise bounds, regret
certificates, the robustness path, selection-stability frequencies, cluster
diagnostics, a JSON summary, and a content-addressed manifest. Selection
stability is the raw root-cluster-bootstrap frequency of each point
robust-regret winner. It does not rerun the simultaneous HRMS procedure inside
each replicate and is neither a confidence level nor a posterior probability.
The summary and manifest record the chosen calibration API; the default is
`jackknife_bootstrap_t`, with `raw_bootstrap_t` and `fixed_max_t` retained for
prespecified sensitivity analyses.

## Input contracts

`losses_select.csv` or `losses_select.parquet` is long-form and contains:

| Column | Meaning |
| --- | --- |
| `observation_id` | Unique record identifier |
| `environment_id` | Represented environment type |
| `root_cluster_id` | Highest independent resampling unit |
| `model_id` | Frozen candidate-configuration identifier |
| `loss` | Finite bounded loss; Brier loss is primary |
| `sample_weight` | Optional positive observation weight |

Every observation must occur exactly once for every candidate. Environment,
root cluster, and weight must be invariant across the candidate rows for an
observation. The schema validator rejects incomplete pairing, duplicate rows,
null identifiers, non-finite losses, and non-positive weights.

Pairs passed through `structural_duplicate_pairs` must be distinct frozen
candidate identifiers known before selection inference. HRMS verifies that each
declared pair has exactly equal loss values at every paired observation before
fixing its contrast interval to zero; a mismatch aborts the fit. Equality in the
loss table alone never causes automatic duplicate treatment.

The strict candidate registry contains `model_id`, `model_family`,
`representation`, `pooling`, `regularization`, `seed`, `calibration`,
`checkpoint_hash`, and `training_data_hash`. Hashes must be hexadecimal content
digests; placeholder values are rejected. The synthetic registry under
`examples/` is complete and valid.

Train/select/test assignments contain `observation_id`, `root_cluster_id`,
`split`, `split_seed`, and `split_hash`. The final field is one canonical SHA-256
digest of the complete root-cluster assignment, repeated on each row. Use
`hrms.splits.assert_no_cluster_leakage` before candidate fitting and again before
final evaluation.

Wide prediction exports are the immutable interface to training code. They
contain observation/environment/cluster metadata, a label, an optional weight,
and one `prediction__<model_id>` column per registered candidate. Conversion to
the long loss contract happens once after training-only calibration; the current
statistical package deliberately does not fit or recalibrate prediction models.

```bash
.venv/bin/python scripts/build_loss_table.py predictions_select.parquet \
  model_registry.csv --output losses_select.parquet
```

This conversion validates the registry hashes, complete candidate set, unique
observations, binary labels, probability range, and positive weights, then writes
its own input/output hash manifest.

## Artifact status

The statistical implementation, synthetic generators, solvers, invariance
tests, and lightweight reproduction path are complete. The full test suite is
run by the reproduction command. Headline calibration and decision studies are
frozen in `configs/simulation/` and bundled under
`results/coverage_final_paired/` and `results/decision_headline/`; each run records its effective
configuration, source hashes, output hashes, and seed rule. Generated `results/`
files remain ignored by version control, so an anonymous release must explicitly
include the cited summaries and manifests rather than relying on a repository
checkout to contain them.

The derived CivilComments pipeline under `applications/civilcomments/` has been
run through a sealed-test evaluation. Its training, selection, test-evaluation,
and IID-root sensitivity manifests are under `results/civilcomments/`. Downloaded
or derived comment text is not redistributed; `data/civilcomments/` and the large
prediction/loss exports remain ignored. Model fitting uses only the official
training partition and HRMS consumes the frozen validation losses. Although
immutable test-loss exports necessarily exist after prediction scoring, the
selection/evaluation path does not semantically inspect or aggregate them until
the selection bundle is persisted and hashed; this is a procedural seal, not
cryptographic access control.

The public TE application is also complete. It is a **derived annotation
benchmark**: the label is association of an hg38 TE locus with an ENCODE SCREEN
V4 enhancer-like cCRE, not causal regulatory activity, and a negative is
not-cCRE-annotated rather than proven inactive. A performance-blind homology
gate retained four represented RepeatMasker families (`ERVL`, `L1`, `L2`, and
`MIR`). The final root-disjoint cohort has 22,590 loci (11,295 matched pairs)
and 10,920 transitive homology roots, split into 13,702/4,450/4,438
train/select/test observations. The immutable cohort and split hashes are
recorded in [ARTIFACT.md](ARTIFACT.md).

The frozen CPU library contains 24 candidate configurations plus two
diagnostic-only controls. With `alpha=0.05`, 1,999 root bootstrap replicates,
and `rho=0.25`, HRMS selected `lr_w128_k6_c0p1` with an upper robust-regret
certificate of 0.012609. The robust model confidence set contained all 24
candidates; it therefore does **not** support a unique robust winner. All five
primary point and uncertainty-aware selectors chose the same configuration in
this application, so the TE result is evidence about environment-specific rank
variation and honest selection uncertainty, not an empirical claim that HRMS
reduced regret here. The finite held-out test reference gives robust regret
0.001383 at `rho=0.25`, within the reported certificate.

Large source, derived-data, prediction, and result trees are intentionally
ignored by version control. The current working tree contains the sealed local
outputs, but no public artifact archive or DOI is claimed. A release must
explicitly package the aggregate tables and manifests identified in
[ARTIFACT.md](ARTIFACT.md); a plain repository checkout is not sufficient.
# neurips-aug-28
