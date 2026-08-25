# Frozen public TE benchmark protocol

Status: data-construction protocol frozen before candidate rankings are observed.

## Task and scope

The benchmark predicts whether a human transposable-element (TE) locus is
associated with an enhancer-like ENCODE candidate cis-regulatory element
(pELS/dELS). This is a biochemical annotation task, not a claim of causal or
experimentally validated endogenous regulation. Negative examples are described
as *not cCRE-annotated*, never as biologically inactive.

The unit of observation is one hg38 TE segment. The represented environments
are the broad RepeatMasker families `Alu`, `L1`, `MIR`, `L2`, `ERVL-MaLR`,
`hAT-Charlie`, `ERV1`, and `ERVL`. A family is removed only if the predeclared
homology/split gate fails, never because of candidate performance.

## Pinned public inputs

| Input | SHA-256 (unless noted) |
| --- | --- |
| `GRCh38-cCREs.bed` | `a292aea64edea7efbef275140b78116efdef1bea5b729dc9e5513421dc868dda` |
| `hg38_2020_rmsk.bed` | `e0b8f6c70766c15d418e6933f610b8c1497cb62b46f3d1efaa56b4b80c171f2a` |
| `rmskOutCurrent.txt.gz` | `3c940b46b1074698386e4cb6ff50c7ac2bdb1dd5c6669a6ea57c9ae7798810dc` |
| `hg38.2bit` | MD5 `dcc3ea27079aa6dc3f9deccd7275e0f8` |

Sources are the SCREEN Registry V4, UCSC Repeat Browser 2020, the UCSC hg38
annotation database, and the UCSC GRCh38 reference sequence. All genomic
coordinates are zero-based and end-exclusive.

## Labels and controls

1. Retain canonical chromosomes 1--22 and X, TE length at least 100 bp, and
   interspersed TE classes; exclude simple, low-complexity, satellite, and RNA
   repeats.
2. A TE is positive when a single pELS/dELS cCRE has at least 50% of its bases
   overlapped by that TE. This follows the published ENCODE-TE construction.
   Multiple qualifying cCREs are resolved by maximum overlap fraction, then
   overlap length, then canonical cCRE identifier.
3. A strict negative TE has no overlap between its interval expanded by 250 bp
   and any Registry V4 cCRE class.
4. Positives and controls are paired without replacement within exact
   RepeatMasker `repName` and chromosome, with full oriented TE-copy length and
   GC used for matching. Sampling is deterministic and capped by subfamily so
   one young repeat does not dominate an environment. Model-window GC is audited
   after the pair-specific negative anchor has been defined, and a GC/length-only
   classifier is reported as a diagnostic rather than a deployment candidate.
5. The primary benchmark targets at most 3,000 positive/control pairs per
   environment (48,000 observations). A larger performance-blind candidate pool
   may be extracted before homology gating.

For a positive, the 512-bp model input is centered on its chosen cCRE midpoint.
Its matched negative uses the same strand-aware normalized position within the
TE. Inputs and full TE-copy sequences are uppercased and oriented in the TE
5-prime-to-3-prime direction; minus-strand records are reverse-complemented.
Boundary windows not exactly 512 bp are dropped and counted.

## Homology, splits, and hard gate

MMseqs2 nucleotide connected components are computed globally at 80% identity
and 80% bidirectional coverage for (a) full TE-copy sequences and (b) final
512-bp model windows. The two equivalence graphs are unioned. Entire resulting
root components are assigned to 60/20/20 train/select/test partitions before
candidate fitting.

Every retained environment must satisfy all of the following in both selection
and test partitions:

- both labels are present;
- at least 20 independent roots (50 preferred);
- the largest root contributes under 5% of observations;
- no root crosses partitions;
- no 80/80 match crosses partitions.

Sensitivity analyses use 70% and 90% identity thresholds, observation- versus
root-balanced risks, cCRE overlap thresholds of 50% and 80%, and negative
buffers of 0, 250, and 500 bp. These alternatives do not alter the frozen
primary construction after candidate rankings are visible.

## Frozen candidate-library family

The CPU-first library contains approximately 18--24 configurations spanning
128-, 256-, and 512-bp centered views and the oriented full TE copy; character
k-mer count/TF--IDF representations; regularized logistic and at least one
distinct calibrated linear probabilistic learner. Hyperparameters and seeds are
registered before selection predictions are examined. GC/length and family-only
models are diagnostics, not deployment candidates. Calibration uses an
internal training-root-only split.

The immutable interface consists of a hashed candidate registry, root-disjoint
split table, and wide select/test probability exports subsequently converted to
Brier-loss tables by the existing HRMS validator.

## Pre-training amendment log

- **2026-08-20, before homology splitting or model fitting:** clarified that
  the frozen GC caliper applies to the full oriented TE copy, not the 512-bp
  input window. A control window's coordinate depends on the normalized anchor
  transferred from its matched positive, so it does not exist independently at
  the time candidate controls are compared. The pipeline instead records and
  reports paired 512-bp window-GC differences after matching. No candidate
  predictions or performance rankings existed when this clarification was made.
