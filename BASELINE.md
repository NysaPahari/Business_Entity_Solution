# Business Entity Resolution

Complete local pipeline: normalization, disk-backed candidate retrieval, character TF-IDF/fuzzy features, LightGBM, threshold calibration, independent validation, and submission packaging. The original challenge statement is preserved in `PROBLEM_STATEMENT.md`.

The included baseline scored **0.87115 macro F0.5** on 3,000 independent audit
entities, with **80.93% blocking recall**. Its test outputs cover all **1,732,544**
Source 1 entities. These are local audit metrics, not a leaderboard score.

## Setup and reproduction

Use **64-bit Python 3.12**. Run commands from this project directory:

```sh
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.normalize --dataset-dir dataset
python -m src.pipeline --stage all --memory 2GB --threads 2
python utils/validate_submission.py --streaming --check-ids --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
python -m src.pipeline --stage package
```

On this workstation, use `.pipeline-venv/Scripts/python.exe` instead of `python`: that is the installed standard Windows environment. The older normalization-only `.venv` uses an MSYS layout.

Keep original train/test TSVs and training ground truth under `dataset/train` and `dataset/test`. Skip normalization if the six `_normalized.tsv` files already exist. See `NORMALIZATION.md` for their schema.

This is a substantial batch job over 24,229,173 records. Allow many GB of temporary disk space and potentially hours on a laptop. DuckDB spills joins/sorts to disk. `--memory` limits DuckDB, not the entire process; feature extraction also uses RAM.

Inference joins 10,000-query batches and uses a bounded queue of three feature
workers by default. Set `--workers 1` for minimum memory usage. Worker count does
not change the candidate selection or predictions.

The default classifier uses **12,000 training, 3,000 calibration, and 3,000 audit S1 entities**, searched against all training S2/S3 references. Every test S1 entity is processed. This is a reproducible baseline, not a claim that every training entity was used for fitting.

## Commands and outputs

| Command / artifact | Purpose |
| --- | --- |
| `python -m src.pipeline --stage train` | Ingest/index training records, fit, calibrate, audit |
| `python -m src.pipeline --stage predict` | Process the entire test set and validate |
| `python -m src.pipeline --stage validate` | Strict coverage, format, membership, duplicates and subset checks |
| `python -m src.pipeline --stage package` | Assemble the submission code folder and zip |
| `artifacts/model.joblib` | TF-IDF vectorizers, LightGBM and threshold |
| `artifacts/metrics.json` | Actual scores, blocking recall and feature importance |
| `artifacts/split_manifest.tsv` | Selected S1 IDs and cohorts |
| `artifacts/data_profile.json` | Source/country counts, missing fields and mean text lengths |
| `output/candidate_pairs.tsv` | Exactly the candidate set passed to the classifier |
| `output/matching_results.tsv` | Predicted matches, always a subset of candidates |
| `output/validation.json` | Strict validation result |
| `dist/business_entity_resolution_submission.zip` | Final deliverable |

Use `python -m src.pipeline --help` for options. Defaults are recorded in `config.json`; CLI defaults are authoritative. Use a new `--work-dir` when changing input files, seed, cohort sizes or index parameters. Candidate retrieval resumes after its last committed batch. Inference restarts and atomically replaces completed output files. A changed model requires fresh inference.

## Method

1. Generic normalization preserves originals and uses no country-specific address parser or external lookup.
2. Inverted keys combine canonical names, significant name tokens, address terms/exact addresses, name-token/address-number combinations, and 12 smallest hashed name four-grams. Keys occurring in more than 128 references are omitted; weighted overlap retrieves up to 32 candidates.
3. Actual 2-4 character n-gram TF-IDF cosine, token-sort similarity, address/numeric overlap and a small country-equality boost rerank to 20 candidates. Character retrieval is approximate, not exact global nearest-neighbor search. Country never hard-filters a candidate.
4. LightGBM features include name/address token Jaccard, token-sort/set ratios, normalized Levenshtein, TF-IDF cosine, exact flags, length ratios, presence flags, numeric overlap/conflict, country equality/presence and retrieval strength. Zero, one or many matches may be predicted independently.
5. Deterministic S1 hash buckets allocate 80% to training, 10% to threshold calibration and 10% to a separate audit, with capped deterministic samples per cohort. TF-IDF and the classifier fit on training-cohort candidate text/labels. Threshold selection maximizes exact macro F0.5 on calibration; audit labels never select the threshold. Singletons receive 1 only for empty predictions.

Blocking recall includes every ground-truth match, even unretrieved ones. No positives are injected into candidates. `oracle_macro_f05` is the ceiling within the final candidates.

## Importable API

```python
from src.normalize import normalize_record
from src.model import MatchingModel
from src.evaluate import entity_f05, macro_f05

model = MatchingModel.load('artifacts/model.joblib')
records = {
    'S1-demo': normalize_record({'business_name': 'Acme Ltd', 'business_address': '12 Rd', 'country': 'X'}),
    'S2-demo': normalize_record({'business_name': 'Acme Limited', 'business_address': '12 Road', 'country': 'X'}),
}
probability = model.score('S1-demo', 'S2-demo', records)
```

For exact parity with batch scores, include the candidate's `retrieval_score` from blocking; the standalone helper defaults it to zero. Load only trusted joblib files.

## Limits

For the subsequent error analysis and measured model/blocking experiments, see
[REFINEMENT.md](REFINEMENT.md) and the generated
[refinement report](artifacts/refinement/REFINEMENT_REPORT.md). Experimental
validation artifacts are separate from the baseline submission described here.

Approximate retrieval, high-frequency pruning and the top-20 cap can miss matches. Normalization cannot fix arbitrary typos or missing components. `St` can mean Street or Saint. France is absent from training, so the audit does not establish performance there. No leaderboard result is claimed or automatically uploaded.

All dependencies are pinned. LightGBM is an MIT-licensed tree model, with no downloaded neural weights. No external entity data, geocoding or business-registry services are used.
