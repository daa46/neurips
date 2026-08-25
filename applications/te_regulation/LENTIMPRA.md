# K562 lentiMPRA secondary validation

This adapter builds a TE-annotated, continuous-response table from the processed K562
large-scale lentiMPRA release for ENCODE experiment
[`ENCSR382BVV`](https://www.encodeproject.org/functional-characterization-experiments/ENCSR382BVV/).
It does not use raw FASTQ files.

The scientific claim is deliberately narrow: `log2(RNA/DNA)` measures sequence-intrinsic
regulatory activity averaged over random lentiviral integration contexts in K562. It is not a
label for endogenous-locus causal regulation.

## Pinned public inputs

The processed tables accompany *Massively parallel characterization of transcriptional
regulatory elements* ([DOI 10.1038/s41586-024-08430-9](https://doi.org/10.1038/s41586-024-08430-9),
PMC11903340, CC BY 4.0). Exact URLs, byte counts, MD5s, SHA-256 digests, archive-member paths,
worksheet names, headers, and row counts are frozen in
[`configs/te/lentimpra_k562.json`](../../configs/te/lentimpra_k562.json).

The source audit found:

- design worksheet `K562 large-scale`: 243,780 data rows;
- activity worksheet `K562_summary_data`: 226,254 data rows with three replicate
  `log2(RNA/DNA)` values and their released mean;
- 17,526 designed elements absent from the processed activity summary after assay processing/QC;
- no per-element p-value, q-value, or 5%-FDR call in any processed K562 worksheet.

The article reports aggregate counts at 5% FDR, but those counts cannot be mapped back to
individual elements from the released processed tables. Consequently,
`activity_active_fdr05` is always serialized as null and
`activity_call_status=not_materialized_in_processed_tables`. A blank is never interpreted as an
inactive element.

TE class, family, and subfamily come from the pinned public UCSC hg38
`rmskOutCurrent.txt.gz`. An assayed genomic interval is retained only when one allowed
RepeatMasker record covers at least 50% of that interval. Overlaps from separate records are not
combined. Ties are resolved using the frozen rule in the configuration.

## Reproduce

Install the optional workbook reader:

```bash
uv sync --extra te
```

Download or verify both pinned inputs:

```bash
python applications/te_regulation/prepare_lentimpra.py download
```

The processed-table archive is approximately 342 MB and the RepeatMasker source is
approximately 170 MB. Existing files are reused only after exact byte-count and hash checks.

Build the derived artifacts:

```bash
python applications/te_regulation/prepare_lentimpra.py prepare \
  --output-dir data/te_lentimpra/processed
```

The command verifies the archive and its two workbook members, streams the pinned worksheets,
attaches RepeatMasker annotations, and writes deterministic immutable outputs:

- `lentimpra_k562_te_elements.csv.gz`: one assayed element orientation per row, including the
  released replicate and mean activity values, assay sequence, coordinates, and selected TE hit;
- `lentimpra_k562_te_activity_long.csv.gz`: one element orientation by released activity
  statistic (`replicate_1`, `replicate_2`, `replicate_3`, or `mean`);
- `lentimpra_k562_preflight.json`: source and artifact hashes, construction rules, counts,
  environment counts, schemas, response summary, claim boundary, and readiness gates.

The pipeline deliberately does not create train/select/test partitions. Both derived tables have
a null `root_cluster_id`, and the preflight declares `ready_for_split=false` and
`ready_for_hrms=false`. Global sequence-homology clustering must assign the independent root
units before any split. Model predictions and a validation loss must then be defined separately;
reporter activity itself is not an HRMS loss tensor.

## Focused tests

The fixture-only tests do not download or process the real cohort:

```bash
pytest -q tests/test_prepare_lentimpra.py
```

They cover the inclusive 50% overlap boundary, disallowed repeat classes, forward/reverse assay
orientations, deterministic tie-breaking, exact schema failure, continuous-response preservation,
null FDR semantics, immutable output bytes, and the HRMS readiness gate.
