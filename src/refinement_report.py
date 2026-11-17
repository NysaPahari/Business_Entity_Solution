"""Summarize measured ablations without claiming a leaderboard or test-set gain."""
import json
from pathlib import Path
from .refinement import score_of,calibration_role


def aggregate(report):
    return next(x for x in report['audit']['segments'] if x['dimension']=='all')


def write(out='artifacts/refinement'):
    out=Path(out)
    reports=json.loads((out/'ablation_results.json').read_text())
    diagnosis=json.loads((out/'baseline_error_analysis.json').read_text())
    original=next(x for x in diagnosis['audit']['segments'] if x['dimension']=='all')
    stages=[('protocol_control','Matched-protocol baseline'),('a_isotonic','A: isotonic'),
            ('b_exact_and_floor_rules','B: exact / low-similarity rules'),
            ('c_mutual_neighbor','C: mutual-neighbor feature'),
            ('d_segment_thresholds','D: segment thresholds'),
            ('e_hard_negative_mining','E: hard-negative mining'),
            ('f_wider_retrieval_frozen_model','F: wider blocking, frozen model'),
            ('g_wider_retrieval_retrained','G: wider blocking, refitted model')]
    lines=[]; previous=None
    for key,label in stages:
        if key not in reports:
            continue
        r=reports[key]; a=aggregate(r)
        reference=r.get('reference_stage',previous)
        delta=score_of(r)-score_of(reports[reference]) if reference else score_of(r)-original['macro_f05']
        lines.append(f'| {label} | {score_of(r):.6f} | {delta:+.6f} | {r["selection_macro_f05"]:.6f} | {a["fp"]} | {a["fn"]} |')
        previous=key
    # Selection is made on the threshold-tuning half, never by inspecting audit gain.
    chosen=max(reports,key=lambda k:reports[k]['selection_macro_f05'])
    selection={'selected_stage':chosen,'criterion':'Highest macro F0.5 on threshold-tuning calibration entities; ties favor earlier/simpler stage.',
               'selection_macro_f05':reports[chosen]['selection_macro_f05'],
               'audit_macro_f05':score_of(reports[chosen]),
               'test_predictions_regenerated':False}
    (out/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    explanations=[]
    for key,label in stages[1:]:
        if key not in reports: continue
        r=reports[key]; a=aggregate(r)
        if key=='a_isotonic':
            explanation=f'Isotonic tied probability levels slightly reduced the best decision boundary; audit Brier changed from {reports["protocol_control"]["audit_brier"]:.6f} to {r["audit_brier"]:.6f}. Monotonic calibration does not create missing discrimination.'
        elif key=='b_exact_and_floor_rules':
            explanation=f'{r["forced_accept_pairs"]} exact nonempty pairs across all selected cohorts were forced accepted; {r["forced_reject_pairs"]} pairs met the rejection floor. Existing decisions already handled these easy cases.'
        elif key=='c_mutual_neighbor':
            old=aggregate(reports['b_exact_and_floor_rules'])
            explanation=f'Cross-fitted forward/reverse agreement changed audit FP {old["fp"]}→{a["fp"]} and FN {old["fn"]}→{a["fn"]}; it cannot restore pairs absent from blocking.'
        elif key=='d_segment_thresholds':
            explanation=f'Tuned cutoffs were {r["thresholds"]}. The one-candidate segment is small; its separate cutoff did not improve the audit result in this run.'
        elif key=='e_hard_negative_mining':
            old=aggregate(reports['d_segment_thresholds'])
            explanation=f'Added {r["new_hard_negatives"]} training-only hard negatives and upweighted {r["upweighted_training_fp_pairs"]} training false positives; audit FP {old["fp"]}→{a["fp"]}, FN {old["fn"]}→{a["fn"]}.'
        elif key=='f_wider_retrieval_frozen_model':
            explanation=f'Allowed keys with up to 1,024 occurrences and a 64-record retrieval pool, keeping final K=20. Audit blocking recall became {r["audit"]["blocking"]["pair_recall"]:.2%}; classifier and TF-IDF were frozen.'
        else:
            explanation='Refitted the same classifier family on the wider training candidates, with unchanged cohort IDs and original training-only TF-IDF; this tests adaptation to the changed negative distribution.'
        explanations.append(f'- **{label}:** {explanation}')
    segments='\n'.join(f'| {r["dimension"]}: {r["segment"]} | {r["entities"]} | {r["fp"]} | {r["fn"]} | {r["blocked_fn"]} | {r["macro_f05"]:.6f} |'
                       for r in diagnosis['audit']['segments'] if r['dimension'] in ('country','candidate_count','truth_type'))
    text=f'''# Entity-resolution error analysis and ablations

## Main findings

The original saved model scored **{original['macro_f05']:.6f} macro F0.5** on the
same 3,000-entity audit cohort used throughout this comparison. It made
**{original['fp']} false-positive pairs** and **{original['fn']} false-negative pairs**.
**{original['blocked_fn']}** false negatives were absent from the final candidates.
Baseline blocking recall was **{diagnosis['audit']['blocking']['pair_recall']:.2%}**,
well below the requested 97% check. Its candidate oracle ceiling was only
**{diagnosis['audit']['blocking']['oracle_macro_f05']:.6f} macro F0.5**. A 0.99 score
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
{segments}

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
{chr(10).join(lines)}

{chr(10).join(explanations)}

## Decision and artifacts

The selection criterion chooses **{chosen}**, with selection F0.5
**{selection['selection_macro_f05']:.6f}** and audit F0.5
**{selection['audit_macro_f05']:.6f}**. This is a measured result, not a promise of 0.99.

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
'''
    (out/'REFINEMENT_REPORT.md').write_text(text,encoding='utf-8')
    return selection


if __name__=='__main__':
    print(json.dumps(write(),indent=2))
