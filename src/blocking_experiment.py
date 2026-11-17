"""A separately measured wider-retrieval branch, gated by diagnosed recall loss."""
import json
from pathlib import Path
from .blocking import build_index,generate_pool
from .data import connect,sql_string,log
from .error_analysis import collect
from .model import MatchingModel
from .refinement import fit_policy,policy_report,score_of
import numpy as np


def run(out='artifacts/refinement'):
    out=Path(out)
    reports=json.loads((out/'ablation_results.json').read_text())
    if 'e_hard_negative_mining' not in reports:
        raise ValueError('Complete requested model ablations A-E before the separate blocking experiment')
    diagnosis=json.loads((out/'baseline_error_analysis.json').read_text())
    if not diagnosis['audit']['blocking_bottleneck']:
        log('Blocking recall is at least 97%; leaving candidate generation unchanged')
        return reports
    database=Path('work/refinement_wide.duckdb')
    marker=Path(str(database)+'.ingested')
    if not marker.exists():
        con=connect(database,'2GB',2)
        try:
            con.execute(f'ATTACH {sql_string(Path("work/train.duckdb").resolve().as_posix())} AS base (READ_ONLY)')
            con.execute('CREATE OR REPLACE TABLE records AS SELECT * FROM base.records')
            con.execute('CREATE OR REPLACE TABLE queries AS SELECT * FROM base.queries')
            con.execute('DETACH base'); con.execute('CHECKPOINT')
        finally:
            con.close()
        marker.write_text('Original records and selected S1 cohort IDs copied unchanged.')
    # Same key definitions, reference universe, and final top-20; only the key
    # frequency allowance and pre-reranking retrieval pool are widened.
    build_index(database,1024,'2GB',2)
    generate_pool(database,64,memory='2GB',threads=2)
    data=collect(database=database,output_dir=out/'wide')
    original=collect(output_dir=out)
    assert {s:q['cohort'] for s,q in data['queries'].items()}=={s:q['cohort'] for s,q in original['queries'].items()}
    indices=data['selected']; rows=[data['rows'][i] for i in indices]
    raw=data['baseline_scores'][indices]
    policy=fit_policy(data,indices,raw)
    report=policy_report(data,indices,raw,policy)
    report.update(reference_stage='protocol_control',max_df=1024,pool_size=64,top_k=20,
                  experiment='Wider retrieval with frozen baseline classifier and TF-IDF; retuned global cutoff only.')
    reports['f_wider_retrieval_frozen_model']=report
    (out/'ablation_results.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    log(f'f_wider_retrieval_frozen_model: audit F0.5={score_of(report):.8f}; recall={report["audit"]["blocking"]["pair_recall"]:.6f}; selection F0.5={report["selection_macro_f05"]:.8f}')
    train=np.array([data['queries'][r[1]]['cohort']=='train' for r in rows])
    labels=np.array([r[5] in data['truth'][r[1]] for r in rows],dtype=np.uint8)
    base=MatchingModel.load('artifacts/model.joblib')
    model=MatchingModel(base.extractor,2026,2).fit(data['x'][indices][train],labels[train])
    raw=model.probabilities(data['x'][indices])
    policy=fit_policy(data,indices,raw)
    model.threshold=policy.thresholds['global']
    report=policy_report(data,indices,raw,policy)
    report.update(reference_stage='f_wider_retrieval_frozen_model',max_df=1024,pool_size=64,top_k=20,
                  experiment='Same wider retrieval; classifier refit on training candidates only; frozen original TF-IDF.')
    reports['g_wider_retrieval_retrained']=report
    (out/'ablation_results.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    deployment=out/'wide_model'; deployment.mkdir(exist_ok=True)
    model.save(deployment/'model.joblib')
    (deployment/'validation_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    log(f'g_wider_retrieval_retrained: audit F0.5={score_of(report):.8f}; selection F0.5={report["selection_macro_f05"]:.8f}')
    return reports


if __name__=='__main__':
    run()
