# ML4H 2026 revision plan

Status: proposed plan, based on the current HRMS implementation and manuscript as
of 2026-08-25.

## Target and scope

- Primary target: ML4H 2026 Proceedings, Area 1 (Models and Methods).
- Fallback: ML4H Findings if the adaptive method is still preliminary at the
  submission freeze.
- Venue note: ML4H 2026 is a standalone symposium held before NeurIPS, not a
  NeurIPS workshop; the current NewInML/NeurIPS formatting is therefore invalid.
- Submission deadline: 2026-09-10, 11:59 PM AoE.
- Recommended paper title: **Adaptive Validation of Biomedical Models under
  Biological Heterogeneity**.
- Recommended method name: **A-HRMS** for the adaptive extension; retain **HRMS**
  for the existing fixed-sample procedure.

The revision should preserve the validated HRMS core and add one defensible
adaptive-validation contribution. It should not attempt to deliver a new
biomedical dataset, a general information-theoretic lower bound, and a new
ambiguity-set theory on the same deadline.

## Selected proposals

### Required for the revision

1. Reframe the problem as allocation of a limited biomedical validation budget
   across heterogeneous biological groups, rather than generic model selection.
2. Make the acquisition unit an independent biological root. Acquiring a root
   reveals the paired loss vector for every frozen candidate on all observations
   in that root.
3. Add an adaptive A-HRMS policy that chooses which eligible root or environment
   to evaluate next, updates a robust-model confidence set, and supports an
   epsilon-robust stopping rule.
4. Replace fixed-sample bootstrap inference in the adaptive path with
   time-uniform confidence sequences. The existing bootstrap HRMS implementation
   remains the fixed-sample comparator.
5. Retain the current deployment-mixture adversary and interpret it explicitly
   as uncertainty in future biological composition. Separate this from the
   acquisition policy: the worst deployment mixture need not be the most
   informative place to collect another root.
6. Run a retrospective acquisition study on the TE selection roots while keeping
   the test split sealed until the policy and stopping decision are fixed.
7. Compare acquisition policies and report cost--certainty frontiers, not only
   final regret.
8. Analyze TE-family ranking heterogeneity by candidate sequence view, context
   length, k-mer size, homology-root structure, and GC/length diagnostics without
   making causal biological claims.
9. Move CivilComments out of the main narrative. Retain it in the appendix as a
   generality check only if space permits.
10. Rewrite the title, abstract, introduction, contributions, limitations, and
    figures around adaptive biomedical validation.

### Conditional addition

Add a simple box-constrained or prevalence-range ambiguity-set sensitivity only
if its limits can be justified by an external biological source. Do not invent
"biologically plausible" bounds after viewing candidate performance. The
chi-square ambiguity set remains the primary analysis otherwise.

### Deferred

- A full information-theoretic characteristic-time lower bound and proof of
  asymptotic optimality.
- A second biomedical application requiring new labels, assays, adjudication, or
  restricted clinical data.
- Claims that the retrospective TE replay demonstrates prospective laboratory or
  annotation-cost savings.
- Renaming the Python package or rewriting the stable HRMS API.

These are credible follow-up directions, but making them submission requirements
would put rigor and completion at risk.

## Method specification

### Estimand and reveal model

- Freeze the candidate library, ambiguity set, acquisition policy, stopping rule,
  and evaluation protocol before running headline experiments.
- Use root-balanced environment risks for the first adaptive implementation. This
  keeps each acquired independent root bounded and avoids silently treating a
  variable-size cluster as IID evidence.
- At step `t`, select one unrevealed root using only previously revealed selection
  data and permissible root metadata. Reveal every candidate's paired Brier losses
  for that root.
- If a root contains observations from multiple environments, reveal it once and
  update every affected environment. Never split a homology/matched-design root.
- Keep the final test losses inaccessible to acquisition, stopping, hyperparameter
  choice, and policy selection.

### Anytime-valid uncertainty and stopping

Implement simultaneous, bounded, root-level confidence sequences for every
candidate-pair--environment coordinate. Allocate error across coordinates once,
then propagate the intervals through the existing robust-regret optimization.

Support two predeclared stopping modes:

1. singleton: stop when one robust model remains;
2. epsilon certificate: stop when every surviving candidate is certified within
   `epsilon` robust regret of the current optimum.

The epsilon rule should be primary because the existing TE data do not identify a
singleton under fixed-sample HRMS. Report budget exhaustion as a valid outcome;
the policy must be allowed to conclude that the evidence is insufficient.

### Acquisition policy

Use a transparent policy based on the uncertainty coordinates that currently
separate plausible robust winners:

1. form the current robust-model confidence set;
2. identify pairwise coordinates contributing to contenders' upper regret
   certificates;
3. score environments by certificate influence multiplied by current uncertainty;
4. acquire an unrevealed root uniformly within the highest-scoring eligible
   environment, with deterministic tie-breaking.

Do not call the policy information-theoretically optimal without a proof. Describe
it as a certificate-directed acquisition rule.

### Baselines

- uniform root sampling;
- sampling proportional to the nominal environment prevalence;
- round-robin environment sampling;
- maximum-width uncertainty sampling;
- worst-environment sampling;
- static allocation estimated from an independent pilot or simulation truth;
- fixed-sample HRMS at matched budgets.

The static oracle may be reported only as an evaluation reference, never as a
deployable baseline.

## Experimental plan

### Simulation

Extend the current rank-reversal generator with sequential root revelation and
vary:

- intracluster correlation and root-size imbalance;
- model-ranking reversal strength;
- environment prevalence;
- ambiguity radius;
- validation budget and epsilon.

Primary metrics:

- simultaneous coverage under adaptive stopping;
- probability the robust population optimum remains in the set;
- roots and observations required to reach an epsilon certificate;
- non-stopping rate at the maximum budget;
- confidence-set size versus cost;
- held-out population robust regret at stopping.

Use paired Monte Carlo seeds across policies and report uncertainty for differences.

### Public TE replay

- Treat the existing selection roots as an initially hidden acquisition pool.
- Reveal whole roots according to each policy; evaluate all 24 candidates together.
- Use the frozen test split exactly once after each policy's selection result has
  been persisted.
- Report root count and observation count because root sizes differ.
- Plot acquisition by TE family over time, confidence-set size, epsilon-certificate
  width, and held-out robust regret.
- Relate allocation patterns to observed family-specific model rankings, while
  clearly labeling the study as retrospective.

### Biological interpretation

Test descriptive associations between family-specific ranks and registered model
features (window size, k-mer length, learner family) plus cohort diagnostics (GC,
TE length, and root concentration). Correct for multiple comparisons or keep the
analysis explicitly exploratory. Do not infer evolutionary or regulatory mechanisms
from association alone.

## Code changes

Proposed additions:

```text
src/hrms/confidence_sequences.py
src/hrms/acquisition.py
src/hrms/active.py
experiments/active/simulation_study.py
experiments/active/te_replay.py
configs/active/simulation.json
configs/active/te_replay.json
tests/test_confidence_sequences.py
tests/test_acquisition.py
tests/test_active.py
```

Keep `src/hrms/selector.py` as the fixed-sample API. Export a separate `ActiveHRMS`
class only after its stopping and data-reveal contracts are tested. Add one CLI for
replayable active studies rather than overloading `hrms-select`.

Required engineering checks:

- deterministic results across worker counts and reruns;
- no test-data path accepted by acquisition-policy code;
- whole-root atomicity and no duplicate reveal;
- bounded-loss and candidate-pairing validation at every update;
- checkpoint/resume without changing the acquisition sequence;
- manifests containing configuration, code, input, output, and stopping hashes;
- all existing tests remain green.

## Paper rewrite

### Main-text structure

1. Biomedical validation-budget problem and concrete acquisition units.
2. Related work: robust model selection, adaptive experimental design, sequential
   inference, and evaluation under health-population shift.
3. Fixed-sample HRMS in compact form.
4. A-HRMS reveal model, confidence sequences, acquisition rule, and stopping rule.
5. Validity statement and assumptions.
6. Sequential simulations.
7. Retrospective TE acquisition and biological heterogeneity.
8. Limitations, ethics, and intended use.

Replace the main-text CivilComments section and table with the adaptive TE result.
Keep detailed fixed-sample derivations, CivilComments, additional baselines, and
protocol tables in the appendix.

### Claims to avoid

- Do not say A-HRMS reduces real assay costs based only on replayed public labels.
- Do not reuse fixed-sample bootstrap coverage claims after adaptive stopping.
- Do not describe cCRE overlap as causal regulation or strict controls as inactive.
- Do not imply that a full confidence set means candidate equivalence.
- Do not claim clinical deployment readiness, safety certification, or support for
  unseen environment types.

## ML4H 2026 format conversion

Use the official 2026 template, not the current NeurIPS/NewInML style.

- Proceedings: 8 content pages at submission, excluding references and appendices;
  one extra content page is allowed only for the camera-ready version.
- Findings fallback: 4 content pages, excluding references and appendices.
- Switch to the official JMLR/PMLR two-column template and set
  `\mlhtrack{proceedings}` (or `findings`) with review mode enabled.
- Keep the paper anonymous and retain review line numbers.
- Keep manuscript content in one `.tex` file as requested by the template; retain
  references in `references.bib` and figures as external assets.
- Remove `neurips_2026.sty`, `\workshoptitle`, and the NeurIPS checklist.
- Add the required `Data and Code Availability` paragraph immediately after the
  abstract. Any linked repository or supplement must be anonymous during review.
- Add the required IRB paragraph. The authors must verify and state the correct
  determination; do not infer an exemption merely because inputs are public.
- Add keywords and confirm all figures/tables are readable in two-column format.
- Compile with PDFLaTeX and the official template, then verify page count,
  anonymization, embedded fonts, links, references, and absence of template warnings.

Official requirements:

- [ML4H 2026 Call for Participation](https://ml4h.ahli.cc/submit/call-for-papers/)
- [ML4H 2026 FAQ](https://ml4h.ahli.cc/resources/faqs/)
- [ML4H 2026 writing guidelines](https://ml4h.ahli.cc/resources/writing-guidelines/)
- [Official ML4H 2026 LaTeX template](https://www.overleaf.com/latex/templates/machine-learning-for-healthcare-symposium-ml4h-2026-template/sqgwhtyswgcy)

## Repository naming and presentation

Recommended post-revision repository name: **`active-hrms`**.

Use **`hrms`** instead if the adaptive extension does not pass the go/no-go gate.
Do not use a date- or venue-specific name for the public research repository.

Keep the Python distribution and import namespace `hrms`; changing those names would
break the existing API without scientific benefit. If A-HRMS lands:

- change the README heading and project description to adaptive biomedical model
  validation;
- document fixed-sample `HRMS` and sequential `ActiveHRMS` separately;
- rename anonymous bundle labels from `hrms-anonymous-artifact` to
  `active-hrms-anonymous-artifact`;
- update `docs/artifact.md`, paper titles, archive metadata, and figure metadata;
- remove NewInML/NeurIPS-specific paper files and wording;
- use an anonymous supplemental archive or anonymous repository during review;
- decide and document the public code license before camera-ready release.

Renaming the GitHub repository does not require renaming source imports. Update the
remote and local directory only after the paper/method naming decision is frozen.

## Schedule and decision gates

### 2026-08-25 to 2026-08-28: specification and format

- freeze the adaptive estimand, confidence-sequence construction, acquisition rule,
  stopping rule, and headline configurations;
- import the official ML4H template and produce a compiling skeleton;
- draft the new introduction, claims, and mandatory availability/IRB statements.

### 2026-08-29 to 2026-09-03: implementation

- implement and unit-test sequential uncertainty, acquisition, and stopping;
- run small simulation and TE replay checks;
- audit that adaptive code cannot access test losses.

### Go/no-go: 2026-09-03

Proceed with A-HRMS as the headline only if:

- adaptive-stopping coverage is empirically compatible with the target level;
- the method has deterministic, tested code;
- at least one non-oracle baseline comparison is complete;
- the TE replay uses only selection information before stopping.

Otherwise submit the validated fixed-sample HRMS method with a biomedical framing,
present adaptive acquisition as future work, and use the ML4H Findings track if the
result is primarily a preliminary direction.

### 2026-09-04 to 2026-09-08: headline runs and writing

- freeze results and manifests;
- create the cost--certainty and TE-allocation figures;
- complete the eight-page Proceedings manuscript and appendix;
- run independent claim-to-artifact checks.

### 2026-09-09 to 2026-09-10: submission audit

- verify the official template and page limit;
- scan the paper, metadata, code bundle, PDF properties, and paths for identity;
- confirm author list, track, subject area, data modality, and reciprocal reviewer;
- upload early enough to validate the OpenReview-rendered submission.

## Completion criteria

- The paper compiles with the official ML4H 2026 template and satisfies the selected
  track's page limit and mandatory statements.
- The adaptive procedure has an explicit reveal model and time-uniform uncertainty;
  no fixed-sample interval is used to justify adaptive stopping.
- Simulations report coverage and acquisition cost with Monte Carlo uncertainty.
- The TE replay preserves root integrity and test sealing.
- Every headline number is reproducible from a content-addressed artifact.
- The repository installs locally, all tests and lint checks pass, and no generated
  caches or private/raw data are included.
- Claims distinguish demonstrated results, retrospective evidence, and future work.
