# Derived CivilComments environment-mixture task

This application retains comments with exactly one of the eight WILDS
identity-category indicators at or above 0.5: male, female, LGBTQ, Christian,
Muslim, other religion, Black, or White. It preserves the mirror's official
`train`, `val`, and `test` assignments. Toxicity is binarized at 0.5 and model
selection uses Brier loss. This is a derived environment-mixture experiment,
not an official WILDS leaderboard evaluation.

The source mirror is CC0-licensed and content-addressed in
`data/civilcomments/source_metadata.json`. The highest defensible dependence
unit is `article_id`: comments in one article share topic and discussion
context. Because some articles span identity environments, HRMS resamples
articles globally and retains every environment-specific record from a sampled
article. No article crosses the official train/validation/test split.

The registered 272,542,670-byte source can be downloaded resumably with:

```bash
mkdir -p data/raw/civilcomments
curl -fL --retry 5 --continue-at - \
  -o data/raw/civilcomments/all_data_with_identities.csv.part \
  https://huggingface.co/datasets/shlomihod/civil-comments-wilds/resolve/main/all_data_with_identities.csv
mv data/raw/civilcomments/all_data_with_identities.csv.part \
  data/raw/civilcomments/all_data_with_identities.csv
sha256sum data/raw/civilcomments/all_data_with_identities.csv
```

The expected SHA-256 is
`403e638c83a225d738a937ff98b61fd0631e30f710d57928c7766d413526b77f`.

From the repository root:

```bash
uv sync --extra civilcomments
.venv/bin/python applications/civilcomments/prepare_data.py
.venv/bin/python applications/civilcomments/train_candidates.py
.venv/bin/python applications/civilcomments/select_models.py --bootstrap 1999
.venv/bin/python applications/civilcomments/evaluate.py
.venv/bin/python applications/civilcomments/ablate_iid_roots.py
```

The 12-candidate frozen library comprises word unigram, word unigram/bigram,
character 3--5-gram, and character 4--6-gram TF--IDF representations, each with
logistic-regression `C` in `{0.1, 1, 10}`. A deterministic 15% internal slice of
the official training article roots is reserved for Platt calibration, with
candidate group splits chosen to balance environment-by-label cells. Neither the
official validation nor test split is used for fitting, tuning, or calibration.
The training stage receives test text and metadata but no test labels. Selection
is persisted with a UTC timestamp and content hashes before the separate test
evaluation command is permitted to open the sealed label artifact.

The final command is explicitly post-test and diagnostic: it deliberately
misspecifies each comment as an independent root to quantify the effect of
ignoring article dependence. It never replaces the sealed primary selection.
