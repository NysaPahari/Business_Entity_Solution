"""Complete requested mutual-feature, threshold-segment, and hard-negative trials."""
import json
from pathlib import Path
import joblib
import numpy as np
from .data import log
from .model import MatchingModel
from .mutual import mutual_features
from .refined_model import ContextualModel
from .refinement import fit_policy,policy_report,score_of


def run(data,out='artifacts/refinement'):
    out=Path(out)
    reports=json.loads((out/'ablation_results.json').read_text())
    selected=data['selected']; x=data['x'][selected]
    rows=[data['rows'][i] for i in selected]
    train=np.array([data['queries'][r[1]]['cohort']=='train' for r in rows])
    labels=np.array([r[5] in data['truth'][r[1]] for r in rows],dtype=np.uint8)
    mutual=np.load(out/'mutual_feature.npy')
    base=MatchingModel.load('artifacts/model.joblib')
    model=ContextualModel(base.extractor).fit(x[train],mutual[train],labels[train])
    raw=model.raw_probabilities(x,mutual)
    for name,segments in [('c_mutual_neighbor',False),('d_segment_thresholds',True)]:
        model.policy=fit_policy(data,selected,raw,isotonic=True,rules=True,segments=segments)
        report=policy_report(data,selected,raw,model.policy)
        report['mutual_feature_gain']=float(model.classifier.booster_.feature_importance(importance_type='gain')[-1])
        reports[name]=report
        joblib.dump(model,out/f'{name}_model.joblib',compress=3)
        (out/'ablation_results.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
        log(f'{name}: audit F0.5={score_of(report):.8f}; selection F0.5={report["selection_macro_f05"]:.8f}; thresholds={model.policy.thresholds}')
    # Mining uses only training-query labels. Calibration/audit errors never enter fit.
    pred=model.policy.predict(rows,x,raw)
    training_fp=train&(labels==0)&pred
    fp_patterns={
        'same_name_partial_numeric_disagreement':int((training_fp&(x[:,5]==1)&(x[:,16]>0)&(x[:,16]<1)).sum()),
        'near_name_same_address':int((training_fp&(x[:,1]>=.8)&(x[:,13]==1)).sum()),
        'same_numbers_weaker_address_overlap':int((training_fp&(x[:,16]==1)&(x[:,10]<.8)).sum()),
        'total_training_false_positive_pairs':int(training_fp.sum())}
    (out/'training_false_positive_patterns.json').write_text(json.dumps(fp_patterns,indent=2),encoding='utf-8')
    fp_sources={r[1] for r,m,y,p in zip(rows,train,labels,pred) if m and not y and p}
    all_train=np.array([data['queries'][r[1]]['cohort']=='train' for r in data['rows']])
    all_labels=np.array([r[5] in data['truth'][r[1]] for r in data['rows']],dtype=np.uint8)
    already=np.zeros(len(data['rows']),bool); already[selected]=True
    all_x=data['x']
    # Similar-name/address confusion patterns seen in TRAINING false positives.
    similar=(all_x[:,1]>=.70)&((all_x[:,10]>=.65)|((all_x[:,16]>0)&(all_x[:,16]<1)))
    near_fp=np.array([r[1] in fp_sources for r in data['rows']])
    hard=all_train&(all_labels==0)&~already&(similar|((data['baseline_scores']>=.15)&near_fp))
    added=np.flatnonzero(hard)
    log(f'Hard-negative mining: {len(fp_sources):,} training FP entities; {len(added):,} new negative pairs outside the final training top-20')
    train_selected=selected[train]
    fitting=np.concatenate([train_selected,added])
    rank_models=joblib.load(out/'rank_models.joblib')
    reverse=joblib.load(out/'reverse_nearest.joblib')
    all_mutual=mutual_features(data,rank_models,reverse,indices=np.arange(len(data['rows'])))
    # Existing production groups retain the same feature values as stage C/D.
    all_mutual[selected]=mutual
    weights=np.ones(len(fitting),dtype=np.float32)
    train_fp_ids={int(i) for i,y,p in zip(selected,labels,pred) if data['queries'][data['rows'][i][1]]['cohort']=='train' and not y and p}
    for j,i in enumerate(fitting):
        if int(i) in train_fp_ids:
            weights[j]=4.0
        elif hard[i]:
            weights[j]=2.0
    mined=ContextualModel(base.extractor).fit(all_x[fitting],all_mutual[fitting],all_labels[fitting],weights)
    raw=mined.raw_probabilities(x,mutual)
    mined.policy=fit_policy(data,selected,raw,isotonic=True,rules=True,segments=True)
    report=policy_report(data,selected,raw,mined.policy)
    report.update(new_hard_negatives=len(added),training_fp_entities=len(fp_sources),
                  upweighted_training_fp_pairs=len(train_fp_ids),fitting_pairs=len(fitting),
                  training_false_positive_patterns=fp_patterns,
                  mining_protocol='Training-only: new rank-21-to-32 similar negatives weight 2; existing training false positives weight 4; no validation labels fitted.')
    reports['e_hard_negative_mining']=report
    joblib.dump(mined,out/'e_hard_negative_mining_model.joblib',compress=3)
    (out/'ablation_results.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    log(f'e_hard_negative_mining: audit F0.5={score_of(report):.8f}; selection F0.5={report["selection_macro_f05"]:.8f}; thresholds={mined.policy.thresholds}')
    with (out/'mined_training_negatives.tsv').open('w',encoding='utf-8') as f:
        f.write('source1_entity_id\tcandidate_entity_id\tbase_probability\n')
        for i in added:
            r=data['rows'][i]
            f.write(f'{r[1]}\t{r[5]}\t{data["baseline_scores"][i]:.10g}\n')
    return reports


if __name__=='__main__':
    from .error_analysis import collect
    run(collect())
