"""Check candidate truncation using a frozen calibration-selected classifier."""
import json
from pathlib import Path
import numpy as np
from .error_analysis import collect
from .features import final_candidates
from .model import MatchingModel
from .refinement import fit_policy, policy_report, score_of
from .data import log


def run():
    out=Path('artifacts/accuracy')
    reports=json.loads((out/'results.json').read_text())
    eligible={k:r for k,r in reports.items() if r['strategy']=='composite' and '_rescore_' not in k}
    name=max(eligible,key=lambda k:eligible[k]['selection_macro_f05'])
    model=MatchingModel.load(out/f'{name}.joblib')
    data=collect(database='work/accuracy_composite.duckdb',output_dir=out/'composite')
    extras=np.load(out/'composite_extra.npy',mmap_mode='r')
    for k in (64,128):
        key=f'{name}_rescore_k{k}'
        if key in reports: continue
        chosen=final_candidates(data['rows'],data['x'],k)
        indices=np.array([i for i in chosen if data['queries'][data['rows'][i][1]]['cohort']!='train'])
        x=data['x'][indices]
        if eligible[name]['enhanced']:
            x=np.column_stack([x,extras[indices]])
        raw=model.probabilities(x)
        policy=fit_policy(data,indices,raw)
        model.threshold=policy.thresholds['global']
        report=policy_report(data,indices,raw,policy)
        report.update(strategy='composite',top_k=k,max_df=4096,pool_size=128,
                      enhanced=eligible[name]['enhanced'],frozen_classifier_from=name,
                      experiment='Frozen classifier; validation-only larger final candidate set and retuned calibration cutoff.')
        reports[key]=report
        model.save(out/f'{key}.joblib')
        (out/'results.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
        log(f'{key}: audit={score_of(report):.8f}; selection={report["selection_macro_f05"]:.8f}; recall={report["audit"]["blocking"]["pair_recall"]:.6f}; oracle={report["audit"]["blocking"]["oracle_macro_f05"]:.6f}')


if __name__=='__main__': run()
