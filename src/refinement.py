"""Leakage-controlled probability calibration and decision-policy ablations.

The calibration cohort is partitioned by entity: one half fits isotonic mapping,
the other selects cutoffs/configurations. Audit labels never fit either component.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from .data import log
from .error_analysis import summarize
from .evaluate import entity_f05


def calibration_role(entity_id):
    return 'isotonic_fit' if hashlib.sha256(('calibration-2026:'+entity_id).encode()).digest()[0]<128 else 'threshold_tune'


def rule_masks(x, enabled):
    if not enabled:
        return np.zeros(len(x),bool),np.zeros(len(x),bool)
    # Exact flags already require nonempty strings in FeatureExtractor.
    accept=(x[:,5]==1)&(x[:,13]==1)
    reject=(np.max(x[:,:5],axis=1)<0.10)&(np.max(x[:,8:13],axis=1)<0.10)&~accept
    return accept,reject


def threshold_with_overrides(truth,rows,scores,accept,reject):
    """Exact macro-F0.5 sweep with unconditional accept/reject masks."""
    if not truth:
        raise ValueError('No threshold-tuning entities')
    counts=Counter(); positives=Counter()
    for row,forced in zip(rows,accept):
        if forced:
            counts[row[1]]+=1
            positives[row[1]]+=row[5] in truth[row[1]]
    def contribution(sid):
        return (1.25*positives[sid]/(0.25*len(truth[sid])+counts[sid])) if truth[sid] else float(counts[sid]==0)
    total=sum(contribution(s) for s in truth)
    best=total/len(truth); threshold=float(np.nextafter(1.,2.))
    free=np.flatnonzero(~accept&~reject)
    order=free[np.argsort(-scores[free],kind='stable')]
    i=0
    while i<len(order):
        cutoff=float(scores[order[i]])
        j=i
        while j<len(order) and scores[order[j]]==cutoff:
            row=rows[order[j]]; sid=row[1]
            previous=contribution(sid)
            counts[sid]+=1; positives[sid]+=row[5] in truth[sid]
            total+=contribution(sid)-previous
            j+=1
        if total/len(truth)>best+1e-12:
            threshold,best=cutoff,total/len(truth)
        i=j
    return threshold,best


@dataclass
class DecisionPolicy:
    calibrator: object = None
    rules: bool = False
    thresholds: dict = field(default_factory=dict)

    def probabilities(self, raw):
        return self.calibrator.predict(raw) if self.calibrator is not None else np.asarray(raw)

    def predict(self, rows, x, raw):
        scores=self.probabilities(raw)
        counts=Counter(r[1] for r in rows)
        accept,reject=rule_masks(x,self.rules)
        thresholds=np.array([self.thresholds.get('one' if counts[r[1]]==1 else 'many',
                                              self.thresholds['global']) for r in rows])
        return accept|(~reject&(scores>=thresholds))


def fit_policy(data,indices,raw,isotonic=False,rules=False,segments=False):
    rows=[data['rows'][i] for i in indices]
    x=data['x'][indices]
    fit=np.array([data['queries'][r[1]]['cohort']=='calibration' and calibration_role(r[1])=='isotonic_fit' for r in rows])
    tune=np.array([data['queries'][r[1]]['cohort']=='calibration' and calibration_role(r[1])=='threshold_tune' for r in rows])
    labels=np.array([r[5] in data['truth'][r[1]] for r in rows])
    policy=DecisionPolicy(rules=rules)
    if isotonic:
        policy.calibrator=IsotonicRegression(out_of_bounds='clip').fit(np.asarray(raw)[fit],labels[fit])
    scores=policy.probabilities(raw)
    all_accept,all_reject=rule_masks(x,rules)
    tune_truth={s:t for s,t in data['truth'].items() if data['queries'][s]['cohort']=='calibration' and calibration_role(s)=='threshold_tune'}
    threshold,_=threshold_with_overrides(tune_truth,[r for r,m in zip(rows,tune) if m],
                                         scores[tune],all_accept[tune],all_reject[tune])
    policy.thresholds['global']=threshold
    if segments:
        counts=Counter(r[1] for r in rows)
        for bucket in ('one','many'):
            group_truth={s:t for s,t in tune_truth.items() if ('one' if counts[s]==1 else 'many')==bucket}
            if len(group_truth)<5:
                continue
            mask=tune&np.array([r[1] in group_truth for r in rows])
            threshold,_=threshold_with_overrides(group_truth,[r for r,m in zip(rows,mask) if m],
                                                  scores[mask],all_accept[mask],all_reject[mask])
            policy.thresholds[bucket]=threshold
    return policy


def policy_report(data,indices,raw,policy):
    rows=[data['rows'][i] for i in indices]
    pred=policy.predict(rows,data['x'][indices],raw)
    probabilities=policy.probabilities(raw)
    report={'thresholds':policy.thresholds,'isotonic':policy.calibrator is not None,'rules':policy.rules}
    for cohort in ('calibration','audit'):
        result,_,_=summarize(data,indices,probabilities,pred,cohort=cohort)
        report[cohort]=result
    tune_ids={s for s,q in data['queries'].items() if q['cohort']=='calibration' and calibration_role(s)=='threshold_tune'}
    predictions=defaultdict(set)
    for r,p in zip(rows,pred):
        if p:
            predictions[r[1]].add(r[5])
    report['selection_macro_f05']=sum(entity_f05(data['truth'][s],predictions[s]) for s in tune_ids)/len(tune_ids)
    for group,ids in [('threshold_tune',tune_ids),('audit',{s for s,q in data['queries'].items() if q['cohort']=='audit'})]:
        mask=np.array([r[1] in ids for r in rows])
        labels=[r[5] in data['truth'][r[1]] for r,m in zip(rows,mask) if m]
        report[group+'_brier']=float(brier_score_loss(labels,probabilities[mask]))
    accept,reject=rule_masks(data['x'][indices],policy.rules)
    report['forced_accept_pairs']=int(accept.sum())
    report['forced_reject_pairs']=int(reject.sum())
    return report


def score_of(report,cohort='audit'):
    return next(r['macro_f05'] for r in report[cohort]['segments'] if r['dimension']=='all')


def run_ab(data,output_dir='artifacts/refinement'):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    indices=data['selected']; raw=data['baseline_scores'][indices]
    reports={}
    specs=[('protocol_control',False,False),('a_isotonic',True,False),('b_exact_and_floor_rules',True,True)]
    for name,iso,rules in specs:
        policy=fit_policy(data,indices,raw,isotonic=iso,rules=rules)
        report=policy_report(data,indices,raw,policy)
        reports[name]=report
        joblib.dump(policy,out/f'{name}_policy.joblib')
        (out/'ablation_results.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
        log(f'{name}: audit F0.5={score_of(report):.8f}; selection F0.5={report["selection_macro_f05"]:.8f}; thresholds={policy.thresholds}')
    return reports


if __name__=='__main__':
    from .error_analysis import collect
    run_ab(collect())
