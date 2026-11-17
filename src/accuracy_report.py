"""Report progress against the requested 0.99 target using measured outcomes."""
import json
import shutil
import hashlib
from pathlib import Path
from .refinement import score_of


def write():
    out=Path('artifacts/accuracy')
    reports=json.loads((out/'results.json').read_text())
    previous=json.loads(Path('artifacts/refinement/ablation_results.json').read_text())
    candidates={**previous,**reports}
    chosen=max(candidates,key=lambda k:candidates[k]['selection_macro_f05'])
    selected=candidates[chosen]
    selection={'selected_stage':chosen,'selection_macro_f05':selected['selection_macro_f05'],
               'audit_macro_f05':score_of(selected),'target':0.99,
               'target_reached_on_audit':score_of(selected)>=.99,
               'test_predictions_regenerated':False,
               'criterion':'Highest threshold-tuning calibration macro F0.5; audit not used for selection'}
    fresh_note='The additional fresh audit has not been run yet.'
    output_note='The original test outputs and submission ZIP have not been replaced. The selected model needs fresh full test inference, validation and packaging before submission.'
    run_info=Path('output_improved/run_info.json')
    if chosen in reports and run_info.exists():
        run=json.loads(run_info.read_text())
        model_hash=hashlib.sha256((out/f'{chosen}.joblib').read_bytes()).hexdigest()
        if run.get('validation_passed') and run.get('model_sha256')==model_hash:
            selection['test_predictions_regenerated']=True
            selection['matching_results_path']=run['files']['matching_results.tsv']['path']
            output_note=('Full test predictions for this selected model are complete and strictly validated. '
                         'Submit `output_improved/matching_results.tsv`; its actual candidates are in '
                         '`output_improved/candidate_pairs.tsv`. `output_improved/run_info.json` records '
                         'model/input provenance and output SHA-256 hashes. The old `output/` files and baseline submission ZIP are preserved.')
    if (out/'fresh_audit.json').exists():
        fresh=json.loads((out/'fresh_audit.json').read_text())
        fresh_score=next(s['macro_f05'] for s in fresh['segments'] if s['dimension']=='all')
        fresh_note=(f'The one-time fresh audit of {fresh["selected_stage"]} scored **{fresh_score:.6f} macro F0.5** '
                    f'on 3,000 new entities, with **{fresh["blocking"]["pair_recall"]:.2%} blocking recall**. '
                    'This is a separate cohort, not a same-split model delta. No model or cutoff was changed based on this result.')
        if 'macro_f05_bootstrap_95_interval' in fresh:
            lo,hi=fresh['macro_f05_bootstrap_95_interval']
            fresh_note+=f' The entity-bootstrap 95% interval is **[{lo:.6f}, {hi:.6f}]** (2,000 resamples).'
        if fresh['selected_stage']==chosen:
            selection['fresh_audit_macro_f05']=fresh_score
            selection['target_reached_on_fresh_audit']=fresh_score>=.99
    (out/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    if chosen in reports:
        deployment=out/'selected'; deployment.mkdir(exist_ok=True)
        shutil.copyfile(out/f'{chosen}.joblib',deployment/'model.joblib')
        (deployment/'validation_metrics.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
        config={k:selected[k] for k in ('strategy','max_df','pool_size','top_k')}
        (deployment/'inference_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    table=[]
    references={
        'wide_k64_base':'g_wider_retrieval_retrained',
        'wide_k20_capacity':'g_wider_retrieval_retrained',
        'wide_k20_enhanced':'wide_k20_capacity',
        'wide_k64_enhanced':'wide_k20_enhanced',
        'composite_k64_base':'wide_k64_base',
        'composite_k20_capacity':'wide_k20_capacity',
        'composite_k20_enhanced':'composite_k20_capacity',
        'composite_k64_enhanced':'composite_k20_enhanced',
        'composite_k64_enhanced_more_training':'composite_k64_enhanced'}
    for name,r in reports.items():
        b=r['audit']['blocking']
        a=next(s for s in r['audit']['segments'] if s['dimension']=='all')
        reference=r.get('frozen_classifier_from',references.get(name))
        delta=f'{score_of(r)-score_of(candidates[reference]):+.6f} vs {reference}' if reference in candidates else 'N/A'
        table.append(f'| {name} | {r["selection_macro_f05"]:.6f} | {score_of(r):.6f} | {delta} | {b["pair_recall"]:.2%} | {b["oracle_macro_f05"]:.6f} | {a["fp"]} | {a["fn"]} |')
    text=f'''# Accuracy follow-up: target 0.99

The calibration-selected configuration is **{chosen}**. Its audit macro F0.5 is
**{score_of(selected):.6f}**, versus the original baseline **0.871150**.
Target reached on this reused audit cohort: **{selection['target_reached_on_audit']}**.
No test labels exist here; no leaderboard score is claimed.

{fresh_note}

| Experiment | Calibration selection F0.5 | Audit F0.5 | Audit delta and comparator | Blocking recall | Candidate oracle F0.5 | FP | FN |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

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

{output_note}

After all configuration choices, `python -m src.fresh_audit` evaluates the selected
model once on 3,000 additional audit-bucket entities. It freezes configuration and
refuses a repeat after its result is exposed. See `fresh_audit.json` when present;
its score is not used to revise thresholds or select another model.
'''
    (out/'ACCURACY_REPORT.md').write_text(text,encoding='utf-8')
    print(json.dumps(selection,indent=2))
    return selection


if __name__=='__main__': write()
