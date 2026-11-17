# Entity-resolution error analysis and ablations

## Main findings

The original saved model scored **0.871150 macro F0.5** on the
same 3,000-entity audit cohort used throughout this comparison. It made
**212 false-positive pairs** and **2419 false-negative pairs**.
**1952** false negatives were absent from the final candidates.
Baseline blocking recall was **80.93%**,
well below the requested 97% check. Its candidate oracle ceiling was only
**0.914300 macro F0.5**. A 0.99 score
cannot be reached on this fixed candidate set, regardless of classifier quality.

Of the 1,952 blocking misses, 1,908 were absent from the 32-record retrieval pool
and 44 were removed by final reranking to 20. Raising the final cutoff alone is
therefore unlikely to fix most misses. Examples include transliteration gaps,
missing address components, and number/location conflicts between near-duplicate
business names. Labels are treated as authoritative; near-identical text is not
assumed to prove identity.

## Baseline audit breakdown

| Segment | S1 entities | FP pairs | FN pairs | Blocked FN pairs | Macro F0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| candidate_count: 0 | 16 | 0 | 36 | 36 | 0.187500 |
| candidate_count: 1 | 21 | 0 | 30 | 30 | 0.773810 |
| candidate_count: many | 2963 | 212 | 2353 | 1886 | 0.875531 |
| country: India | 1191 | 105 | 1161 | 966 | 0.831624 |
| country: US | 1809 | 107 | 1258 | 986 | 0.897172 |
| truth_type: real_match | 2832 | 189 | 2419 | 1952 | 0.871274 |
| truth_type: singleton | 168 | 23 | 0 | 0 | 0.869048 |

Full cross-segments and every error are saved in `audit_error_segments.tsv`,
`audit_errors.json`, and their calibration equivalents. These counts are pair
errors; an entity may have more than one. France has no training labels and is
not represented by these validation scores.

## Protocol

No audit labels enter classifier fitting, calibration fitting, threshold selection,
or hard-negative mining. The existing calibration cohort is partitioned by S1 ID:
one half fits isotonic regression and the other selects thresholds. The
matched-protocol raw-score control uses the same threshold-tuning half, isolating
calibration from the smaller tuning sample. Its small difference from the legacy
baseline is due to that changed tuning subset, not a new classifier.

Each requested change A-E is measured cumulatively in the requested order. All
use the identical final blocking candidates and original audit S1 IDs. The
separate F/G branch is permitted only because measured recall is below 97%.
F is compared with the matched-protocol baseline; G is compared with F.

The repeated audit scores are diagnostic comparisons on an exposed validation
set, not a fresh final holdout or a leaderboard result. Configuration selection
uses only the threshold-tuning calibration half; a new untouched holdout is
advisable before relying on small improvements.

## Measured results

| Experiment | Audit F0.5 | Delta vs preceding/control | Selection F0.5 | FP pairs | FN pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matched-protocol baseline | 0.871116 | -0.000034 | 0.879694 | 218 | 2411 |
| A: isotonic | 0.870261 | -0.000855 | 0.879177 | 259 | 2365 |
| B: exact / low-similarity rules | 0.870261 | +0.000000 | 0.879177 | 259 | 2365 |
| C: mutual-neighbor feature | 0.868375 | -0.001886 | 0.878287 | 335 | 2284 |
| D: segment thresholds | 0.867803 | -0.000571 | 0.878287 | 335 | 2286 |
| E: hard-negative mining | 0.871733 | +0.003930 | 0.880204 | 226 | 2428 |
| F: wider blocking, frozen model | 0.889978 | +0.018862 | 0.902481 | 289 | 2011 |
| G: wider blocking, refitted model | 0.888724 | -0.001253 | 0.903108 | 204 | 2157 |

- **A: isotonic:** Isotonic tied probability levels slightly reduced the best decision boundary; audit Brier changed from 0.009471 to 0.009555. Monotonic calibration does not create missing discrimination.
- **B: exact / low-similarity rules:** 1347 exact nonempty pairs across all selected cohorts were forced accepted; 0 pairs met the rejection floor. Existing decisions already handled these easy cases.
- **C: mutual-neighbor feature:** Cross-fitted forward/reverse agreement changed audit FP 259→335 and FN 2365→2284; it cannot restore pairs absent from blocking.
- **D: segment thresholds:** Tuned cutoffs were {'global': 0.44047619047619047, 'one': 0.8627450980392157, 'many': 0.44047619047619047}. The one-candidate segment is small; its separate cutoff did not improve the audit result in this run.
- **E: hard-negative mining:** Added 701 training-only hard negatives and upweighted 1126 training false positives; audit FP 335→226, FN 2286→2428.
- **F: wider blocking, frozen model:** Allowed keys with up to 1,024 occurrences and a 64-record retrieval pool, keeping final K=20. Audit blocking recall became 84.96%; classifier and TF-IDF were frozen.
- **G: wider blocking, refitted model:** Refitted the same classifier family on the wider training candidates, with unchanged cohort IDs and original training-only TF-IDF; this tests adaptation to the changed negative distribution.

## Decision and artifacts

The selection criterion chooses **g_wider_retrieval_retrained**, with selection F0.5
**0.903108** and audit F0.5
**0.888724**. This is a measured result, not a promise of 0.99.

- `audit_top_errors.md` and `calibration_top_errors.md`: top 15 false positives,
  top 15 blocking-prioritized false negatives, plus 15 scored false negatives.
  Each includes raw/normalized names and addresses, true label, and probability.
  Blocked pairs have **no production probability**; separately labelled
  diagnostic-only scores use zero retrieval strength and are not reported as
  actual inference results.
- `ablation_results.json`: thresholds, complete segment metrics, Brier scores,
  blocking recall and candidate-oracle ceilings for every trial.
- `training_false_positive_patterns.json` and `mined_training_negatives.tsv`:
  training-only mining evidence. No validation false positives are added to fit.
- `*_policy.joblib`, `*_model.joblib`, `rank_models.joblib`: experimental artifacts.
  Mutual-neighbor models require full candidate context and are not drop-in
  replacements for a pair-only scorer. Reverse retrieval searches all 2,206,821
  training S1 records; its nearest-neighbor result is approximate within the
  reverse retrieval pool. Three out-of-fold rankers prevent own-query label reuse
  in training mutual features. Ties are allowed; no one-to-one constraint is imposed.

The existing submission TSVs and baseline model are unchanged. New test-set
predictions have **not** been generated by these validation experiments.
