"""One-time evaluation on new audit entities after configuration selection."""
import csv
import json
from pathlib import Path
import numpy as np
from .blocking import key_sql
from .data import connect,log
from .features import final_candidates
from .model import MatchingModel
from .error_analysis import summarize


def run(size=3000):
    out=Path('artifacts/accuracy')
    destination=out/'fresh_audit.json'
    if destination.exists():
        raise ValueError('Fresh audit already exposed. Do not repeatedly select against this holdout.')
    selection=json.loads((out/'selection.json').read_text())
    name=selection['selected_stage']
    reports=json.loads((out/'results.json').read_text())
    config=reports[name]
    manifest=out/'fresh_audit_config.json'
    frozen=dict(selected_stage=name,configuration=config,size=size)
    if manifest.exists() and json.loads(manifest.read_text())!=frozen:
        raise ValueError('Fresh audit configuration changed after evaluation began')
    manifest.write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    strategy=config['strategy']
    database='work/accuracy_composite.duckdb' if strategy=='composite' else 'work/refinement_wide.duckdb'
    model=MatchingModel.load(out/f'{name}.joblib')
    con=connect(database,'2GB',1)
    con.execute('SET enable_progress_bar=false')
    tables={r[0] for r in con.execute('SHOW TABLES').fetchall()}
    if 'fresh_queries' not in tables:
        con.execute(f'''CREATE TABLE fresh_queries AS SELECT row_number() OVER(ORDER BY rid)::BIGINT qid,rid,entity_id
            FROM (SELECT rid,entity_id FROM records WHERE source=1
             AND hash(entity_id || '2026') % 10=1
             AND rid NOT IN (SELECT rid FROM queries)
             ORDER BY hash(entity_id || 'sample2026'),entity_id LIMIT {int(size)})''')
    if 'fresh_query_keys' not in tables:
        sql=key_sql('composite' if strategy=='composite' else 'baseline').replace('FROM records','FROM records WHERE rid IN (SELECT rid FROM fresh_queries)')
        con.execute('CREATE TABLE fresh_query_keys AS SELECT q.qid,k.key FROM ('+sql+') k JOIN fresh_queries q USING(rid)')
    con.execute('CREATE TABLE IF NOT EXISTS fresh_pool(qid BIGINT,rid BIGINT,retrieval_score FLOAT)')
    con.execute('CREATE TABLE IF NOT EXISTS fresh_progress(through BIGINT)')
    through=con.execute('SELECT coalesce(max(through),0) FROM fresh_progress').fetchone()[0]
    for lo in range(through+1,size+1,1000):
        hi=min(lo+999,size)
        log(f'Retrieving fresh audit entities {lo:,}-{hi:,}; model/cutoff frozen')
        con.execute('BEGIN')
        con.execute(f'''INSERT INTO fresh_pool SELECT q.qid,r.rid,sum(1.0/sqrt(r.df)) score
            FROM (SELECT * FROM fresh_query_keys WHERE qid BETWEEN {lo} AND {hi}) q JOIN ref_keys r USING(key)
            GROUP BY q.qid,r.rid QUALIFY row_number() OVER(PARTITION BY q.qid ORDER BY score DESC,r.rid)<={int(config['pool_size'])}''')
        con.execute('INSERT INTO fresh_progress VALUES (?)',[hi]); con.execute('COMMIT')
    queries={r[0]:dict(cohort='audit',country=r[1]) for r in con.execute(
        'SELECT q.entity_id,r.country FROM fresh_queries q JOIN records r USING(rid)').fetchall()}
    truth={}
    with Path('dataset/train/train_ground_truth.tsv').open(encoding='utf-8',newline='') as stream:
        for r in csv.DictReader(stream,delimiter='\t'):
            if r['source1_entity_id'] in queries:
                truth[r['source1_entity_id']]=set(filter(None,r['matched_entity_ids'].split(',')))
    if set(truth)!=set(queries): raise ValueError('Fresh audit truth missing')
    minimal=[]; probabilities=[]
    for lo in range(1,size+1,500):
        rows=con.execute('''SELECT q.qid,q.entity_id,s.name,s.address,s.country,r.entity_id,r.name,r.address,r.country,p.retrieval_score
            FROM fresh_pool p JOIN fresh_queries q USING(qid) JOIN records s ON s.rid=q.rid
            JOIN records r ON r.rid=p.rid WHERE q.qid BETWEEN ? AND ? ORDER BY q.qid,r.entity_id''',[lo,lo+499]).fetchall()
        x=model.extractor.transform(rows)
        chosen=final_candidates(rows,x,config['top_k'])
        probabilities.extend(model.probabilities(x[chosen]))
        minimal.extend((rows[i][0],rows[i][1],'','','',rows[i][5]) for i in chosen)
        log(f'Scored fresh audit through {min(lo+499,size):,}')
    con.close()
    data=dict(rows=minimal,queries=queries,truth=truth,threshold=model.threshold)
    report,errors,_=summarize(data,np.arange(len(minimal)),probabilities)
    error_scores={e['source1_entity_id']:e['entity_f05'] for e in errors}
    entity_scores=np.array([error_scores.get(s,1.0) for s in sorted(queries)])
    bootstrap=np.random.default_rng(2026).choice(entity_scores,size=(2000,len(entity_scores))).mean(axis=1)
    report['macro_f05_bootstrap_95_interval']=np.quantile(bootstrap,[.025,.975]).tolist()
    with (out/'fresh_audit_entities.tsv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream,delimiter='\t',lineterminator='\n')
        writer.writerow(['source1_entity_id','country','entity_f05'])
        writer.writerows((s,queries[s]['country'],score) for s,score in zip(sorted(queries),entity_scores))
    report.update(selected_stage=name,threshold=model.threshold,
                  protocol='New hash-bucket-1 entities, disjoint from training and all previous selected audit IDs. Model, features, retrieval and threshold frozen before scoring. No subsequent selection on this result.')
    destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'fresh_audit_errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
    score=next(r['macro_f05'] for r in report['segments'] if r['dimension']=='all')
    log(f'Fresh audit F0.5={score:.8f}; recall={report["blocking"]["pair_recall"]:.6f}')


if __name__=='__main__': run()
