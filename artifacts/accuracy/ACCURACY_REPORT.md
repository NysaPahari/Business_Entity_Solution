# Accuracy follow-up: target 0.99

The calibration-selected configuration is **composite_k64_enhanced_more_training_rescore_k128**. Its audit macro F0.5 is
**0.938735**, versus the original baseline **0.871150**.
Target reached on this reused audit cohort: **False**.
No test labels exist here; no leaderboard score is claimed.

The one-time fresh audit of composite_k64_enhanced_more_training_rescore_k128 scored **0.945480 macro F0.5** on 3,000 new entities, with **94.62% blocking recall**. This is a separate cohort, not a same-split model delta. No model or cutoff was changed based on this result. The entity-bootstrap 95% interval is **[0.940448, 0.950217]** (2,000 resamples).

| Experiment | Calibration selection F0.5 | Audit F0.5 | Audit delta and comparator | Blocking recall | Candidate oracle F0.5 | FP | FN |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| wide_k64_base | 0.904542 | 0.892020 | +0.003296 vs g_wider_retrieval_retrained | 86.43% | 0.945596 | 230 | 2085 |
| wide_k20_enhanced | 0.916166 | 0.908113 | +0.015429 vs wide_k20_capacity | 84.96% | 0.939677 | 149 | 1929 |
| wide_k64_enhanced | 0.916940 | 0.907035 | -0.001078 vs wide_k20_enhanced | 86.43% | 0.945596 | 188 | 1861 |
| wide_k20_capacity | 0.905451 | 0.892684 | +0.003960 vs g_wider_retrieval_retrained | 84.96% | 0.939677 | 198 | 2099 |
| composite_k64_base | 0.929304 | 0.919634 | +0.027614 vs wide_k64_base | 93.68% | 0.977425 | 299 | 1425 |
| composite_k20_capacity | 0.930945 | 0.921116 | +0.028432 vs wide_k20_capacity | 91.46% | 0.971725 | 269 | 1443 |
| composite_k20_enhanced | 0.945344 | 0.934490 | +0.013374 vs composite_k20_capacity | 91.46% | 0.971725 | 213 | 1236 |
| composite_k64_enhanced | 0.944892 | 0.936340 | +0.001850 vs composite_k20_enhanced | 93.68% | 0.977425 | 177 | 1238 |
| composite_k64_enhanced_more_training | 0.949906 | 0.938861 | +0.002521 vs composite_k64_enhanced | 93.68% | 0.977425 | 169 | 1186 |
| composite_k64_enhanced_more_training_rescore_k64 | 0.949906 | 0.938861 | +0.000000 vs composite_k64_enhanced_more_training | 93.68% | 0.977425 | 169 | 1186 |
| composite_k64_enhanced_more_training_rescore_k128 | 0.950000 | 0.938735 | -0.000126 vs composite_k64_enhanced_more_training | 94.37% | 0.979919 | 176 | 1171 |

## Changes and interpretation

- `base` uses the existing 21 features and 250-tree model.
- `capacity` keeps those 21 features but uses 600 trees and 63 leaves, controlling
  for the capacity increase also used by `enhanced`.
- `enhanced` adds legal-suffix-free name similarity, partial text similarity,
  vowel-stripped token similarity, numeric coverage/order/conflicts with leading
  zeros collapsed, and address similarity after removing numbers. Empty fields
  do not become perfect matches. These are generic features, not country rules.
- `wide` uses the prior max_df=1024, pool=64 index. `composite` adds name/address
  token intersections, address-token pairs, and spelling-skeleton keys with and
  without numbers; max_df=4096 and pool=128. It searches the same full reference
  universe and keeps the same train/calibration/audit query IDs. K is the final
  label-free reranking limit before model scoring.
- Final candidate lists are not supplemented with ground-truth positives. Blocking
  recall includes unretrieved true matches, and the candidate oracle assumes
  perfect classification of only those pairs actually retrieved.
- `more_training` adds 30,000 new training-bucket queries, giving 42,000 training
  entities with identical calibration/audit IDs. `rescore` keeps the classifier
  frozen and tests larger final candidate sets with only the cutoff retuned.

All model fits use the training cohort only. Raw probability cutoffs and
configuration selection use the same calibration tuning half as the previous
experiments. The repeatedly inspected audit is a diagnostic set, not a fresh
final holdout. Performance for unseen France is not established by these scores.

## Reproduce and use

Run from the project root after the baseline and refinement prerequisites:

```powershell
.pipeline-venv/Scripts/python.exe -m src.push_accuracy --strategy wide
.pipeline-venv/Scripts/python.exe -m src.push_accuracy --strategy composite
.pipeline-venv/Scripts/python.exe -m src.more_training
.pipeline-venv/Scripts/python.exe -m src.full_pool
.pipeline-venv/Scripts/python.exe -m src.accuracy_report
```

Each model is saved as `artifacts/accuracy/<experiment>.joblib`. Enhanced models
load through `MatchingModel.load` and embed their enhanced extractor. Batch
inference must use the reported strategy, max_df, pool size and K. Composite
indexing uses one thread and a 4 GB DuckDB limit; use a fresh work directory when
changing index settings. Cached experiments are for these exact inputs/settings.

The selected model is also copied to `artifacts/accuracy/selected/model.joblib`,
alongside `inference_config.json`. To generate new test outputs with this
selection, use `python -m src.pipeline --stage predict` with
`--artifact-dir artifacts/accuracy/selected`, a fresh `--work-dir`, a separate
`--output-dir`, and the four recorded retrieval settings (`strategy` maps to
`--blocking-strategy`, with `wide` meaning `baseline`). For composite retrieval,
use `--memory 4GB --threads 1 --retrieval-batch-size 2000 --workers 1` initially.
This is a substantial full-dataset batch job; the command is not run by the
validation experiments.

Full test predictions for this selected model are complete and strictly validated. Submit `output_improved/matching_results.tsv`; its actual candidates are in `output_improved/candidate_pairs.tsv`. `output_improved/run_info.json` records model/input provenance and output SHA-256 hashes. The old `output/` files and baseline submission ZIP are preserved.

After all configuration choices, `python -m src.fresh_audit` evaluates the selected
model once on 3,000 additional audit-bucket entities. It freezes configuration and
refuses a repeat after its result is exposed. See `fresh_audit.json` when present;
its score is not used to revise thresholds or select another model.
