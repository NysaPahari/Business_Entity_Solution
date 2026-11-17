# Validation refinement handoff

The subsequent push toward 0.99 is recorded in
`artifacts/accuracy/ACCURACY_REPORT.md`. That report and
`artifacts/accuracy/selection.json` supersede the earlier refinement selection
when choosing among all completed experiments. The original test submission
remains unchanged until fresh inference is run.

`run_improved_predictions.ps1` runs the selected improved model on all test
records and validates `output_improved/matching_results.tsv` together with its
candidate file. Retrieval and scoring resume from saved checkpoints when inputs
are unchanged. Keep `.tmp` files and `inference_checkpoint.json` together during
an interrupted run; only final `.tsv` files with a successful `validation.json`
are ready for submission. The original `output/` folder is preserved.

Read `artifacts/refinement/REFINEMENT_REPORT.md` for measured results and the
decision. `ablation_results.json` contains full metrics, cutoffs, candidate recall,
and segment breakdowns. `audit_top_errors.md` shows concrete raw and normalized
examples; `audit_errors.json` contains every audit error.

Run from the project root using the pinned dependencies in `requirements.txt`:

```powershell
.pipeline-venv/Scripts/python.exe -m unittest discover -s tests -v
.pipeline-venv/Scripts/python.exe -m src.error_analysis
.pipeline-venv/Scripts/python.exe -m src.refinement
.pipeline-venv/Scripts/python.exe -m src.mutual
.pipeline-venv/Scripts/python.exe -m src.ablate
.pipeline-venv/Scripts/python.exe -m src.blocking_experiment
.pipeline-venv/Scripts/python.exe -m src.refinement_report
.pipeline-venv/Scripts/python.exe -m src.package_refinement
```

On other machines substitute your environment's `python`. These commands require
the original six normalized datasets, training ground truth, `work/train.duckdb`,
and `artifacts/model.joblib` from the baseline pipeline. The reverse search and
wider-blocking experiments need substantial disk space and runtime. Existing
trusted local caches allow resuming this exact experiment. If input data, model,
split, or settings change, use fresh experiment caches/databases; cached files
are not automatically invalidated against changed inputs.

The commands run diagnosis, isotonic calibration, rules, mutual-neighbor
features, segment cutoffs, and training-only hard-negative mining in order.
The separate wider-blocking branch is gated on measured blocking recall below
97%. It preserves cohort IDs and final K=20.

The calibration cohort is split by entity into isotonic-fitting and
threshold-tuning subsets. The latter selects configurations. Audit labels never
fit models, calibrators, thresholds, or mined negatives. Repeated audit inspection
still means these are diagnostic comparisons, not a new untouched final holdout.
France is absent from labelled training/validation, so these scores do not
establish performance there.

`src.refinement.DecisionPolicy` exposes calibrated probabilities and rule/threshold
decisions. Pass all candidates for each S1 together when applying segment cutoffs.
`src.refined_model.ContextualModel` requires the mutual-neighbor feature explicitly;
it rejects scoring without that context. Its artifacts cannot replace a
`MatchingModel` inside the existing pair-only inference command. Reverse nearest
neighbors are approximate within retrieval candidates, with ties retained and no
one-to-one assignment constraint.

`artifacts/refinement/wide_model/model.joblib` uses the ordinary `MatchingModel`
API but requires the experiment's retrieval settings (max_df=1024, pool_size=64,
top_k=20) for batch parity. Consult the report before choosing any experimental
model. Load joblib artifacts only from trusted sources.

The original `artifacts/model.joblib`, test outputs, and submission ZIP remain the
baseline deliverable. These validation commands do not regenerate test predictions
or update that ZIP. A selected experimental model needs fresh full test inference,
output validation, and packaging before submission.

Share `dist/entity_resolution_refinement.zip` for team review. It contains source,
tests, documentation, reports, model artifacts, and a SHA-256 manifest, with
archive integrity checked during packaging. It excludes datasets, DuckDB work
databases, and regenerated feature/reverse-neighbor caches. It is a validation
handoff, not a replacement for the baseline competition submission ZIP.
