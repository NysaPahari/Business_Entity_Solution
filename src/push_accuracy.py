"""Measured retrieval/feature follow-ups; calibration selects, audit only reports."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from .accuracy_features import extra_features, EXTRA_NAMES
from .blocking import build_index, generate_pool
from .data import connect, sql_string, log
from .error_analysis import collect
from .features import final_candidates, FEATURE_NAMES
from .model import MatchingModel
from .refinement import fit_policy, policy_report, score_of


class EnhancedExtractor:
    def __init__(self, base): self.base=base
    def transform(self, rows):
        return np.column_stack([self.base.transform(rows),extra_features(rows)])


def prepare_composite():
    database=Path('work/accuracy_composite.duckdb')
    marker=Path(str(database)+'.ingested')
    if not marker.exists():
        con=connect(database,'2GB',2)
        try:
            con.execute(f'ATTACH {sql_string(Path("work/train.duckdb").resolve().as_posix())} AS base (READ_ONLY)')
            con.execute('CREATE OR REPLACE TABLE records AS SELECT * FROM base.records')
            con.execute('CREATE OR REPLACE TABLE queries AS SELECT * FROM base.queries')
            con.execute('DETACH base'); con.execute('CHECKPOINT')
        finally: con.close()
        marker.write_text('Same reference universe and cohort IDs as baseline')
    build_index(database,4096,'4GB',1,strategy='composite')
    generate_pool(database,128,batch_size=2000,memory='2GB',threads=1)
    return database


def run(strategy='wide'):
    out=Path('artifacts/accuracy'); out.mkdir(exist_ok=True)
    if strategy=='wide':
        data=collect(database='work/refinement_wide.duckdb',output_dir='artifacts/refinement/wide')
    else:
        data=collect(database=prepare_composite(),output_dir=out/'composite')
    extra_path=out/f'{strategy}_extra.npy'
    if extra_path.exists(): extras=np.load(extra_path)
    else:
        log(f'Computing generic extra features for {len(data["rows"]):,} pairs')
        extras=extra_features(data['rows']); np.save(extra_path,extras)
    base=MatchingModel.load('artifacts/model.joblib')
    result_path=out/'results.json'
    reports=json.loads(result_path.read_text()) if result_path.exists() else {}
    for k,enhanced,capacity in [(64,False,False),(20,False,True),(20,True,True),(64,True,True)]:
        name=f'{strategy}_k{k}_'+('enhanced' if enhanced else 'capacity' if capacity else 'base')
        if name in reports: continue
        indices=final_candidates(data['rows'],data['x'],k)
        rows=[data['rows'][i] for i in indices]
        train=np.array([data['queries'][r[1]]['cohort']=='train' for r in rows])
        labels=np.array([r[5] in data['truth'][r[1]] for r in rows],dtype=np.uint8)
        x=data['x'][indices]
        model=MatchingModel(EnhancedExtractor(base.extractor) if enhanced else base.extractor,2026,2)
        names=FEATURE_NAMES
        if enhanced:
            x=np.column_stack([x,extras[indices]])
            names=FEATURE_NAMES+EXTRA_NAMES
        if capacity:
            model.classifier.set_params(n_estimators=600,num_leaves=63,min_child_samples=60,reg_lambda=5.)
        log(f'Fitting {name}: {int(train.sum()):,} training pairs')
        model.classifier.fit(x[train],labels[train],feature_name=names)
        raw=model.probabilities(x)
        policy=fit_policy(data,indices,raw)
        model.threshold=policy.thresholds['global']
        report=policy_report(data,indices,raw,policy)
        report.update(strategy=strategy,top_k=k,enhanced=enhanced,increased_capacity=capacity,
                      max_df=1024 if strategy=='wide' else 4096,
                      pool_size=64 if strategy=='wide' else 128)
        reports[name]=report
        # Use an importable module name even when invoked with python -m.
        if enhanced:
            from src.push_accuracy import EnhancedExtractor as PersistedExtractor
            model.extractor=PersistedExtractor(base.extractor)
        model.save(out/f'{name}.joblib')
        result_path.write_text(json.dumps(reports,indent=2),encoding='utf-8')
        log(f'{name}: audit={score_of(report):.8f}, selection={report["selection_macro_f05"]:.8f}, recall={report["audit"]["blocking"]["pair_recall"]:.6f}, oracle={report["audit"]["blocking"]["oracle_macro_f05"]:.6f}')
    return reports


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--strategy',choices=['wide','composite'],default='wide')
    run(parser.parse_args().strategy)
