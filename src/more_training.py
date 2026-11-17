"""Scale training queries without changing validation entities or retrieval rules."""
import csv
import gc
import json
from pathlib import Path
import numpy as np
from .blocking import key_sql
from .data import connect, log
from .features import final_candidates, FEATURE_NAMES
from .accuracy_features import EXTRA_NAMES
from .model import MatchingModel
from .error_analysis import collect
from .refinement import fit_policy, policy_report, score_of
from .push_accuracy import EnhancedExtractor


def prepare_extra(size=30000):
    con=connect('work/accuracy_composite.duckdb','2GB',1)
    con.execute('SET enable_progress_bar=false')
    tables={r[0] for r in con.execute('SHOW TABLES').fetchall()}
    if 'extra_queries' not in tables:
        log(f'Selecting {size:,} additional training S1 entities; validation unchanged')
        con.execute(f'''CREATE TABLE extra_queries AS
            SELECT row_number() OVER(ORDER BY rid)::BIGINT qid,rid,entity_id FROM
            (SELECT rid,entity_id FROM records WHERE source=1
             AND hash(entity_id || '2026') % 10 >= 2
             AND rid NOT IN (SELECT rid FROM queries)
             ORDER BY hash(entity_id || 'sample2026'),entity_id LIMIT {int(size)})''')
    if 'extra_query_keys' not in tables:
        sql=key_sql('composite').replace('FROM records','FROM records WHERE rid IN (SELECT rid FROM extra_queries)')
        con.execute('CREATE TABLE extra_query_keys AS SELECT q.qid,k.key FROM ('+sql+') k JOIN extra_queries q USING(rid)')
    count=con.execute('SELECT count(*) FROM extra_queries').fetchone()[0]
    if count != size:
        raise ValueError('Extra training size differs from existing cache')
    con.execute('CREATE TABLE IF NOT EXISTS extra_pool(qid BIGINT,rid BIGINT,retrieval_score FLOAT)')
    con.execute('CREATE TABLE IF NOT EXISTS extra_progress(through BIGINT)')
    through=con.execute('SELECT coalesce(max(through),0) FROM extra_progress').fetchone()[0]
    for lo in range(through+1,size+1,2000):
        hi=min(lo+1999,size)
        log(f'Retrieving extra training queries {lo:,}-{hi:,}')
        con.execute('BEGIN')
        con.execute(f'''INSERT INTO extra_pool SELECT q.qid,r.rid,sum(1.0/sqrt(r.df)) score
            FROM (SELECT * FROM extra_query_keys WHERE qid BETWEEN {lo} AND {hi}) q
            JOIN ref_keys r USING(key) GROUP BY q.qid,r.rid
            QUALIFY row_number() OVER(PARTITION BY q.qid ORDER BY score DESC,r.rid)<=128''')
        con.execute('INSERT INTO extra_progress VALUES (?)',[hi]); con.execute('COMMIT')
    con.execute('CHECKPOINT'); con.close()


def extra_arrays(model,k=64,size=30000):
    folder=Path('artifacts/accuracy/extra_training'); folder.mkdir(exist_ok=True)
    marker=folder/'complete.json'
    if marker.exists():
        info=json.loads(marker.read_text())
        if (info['k'],info['size'])!=(k,size): raise ValueError('Extra feature cache settings changed')
        return np.load(folder/'x.npy',mmap_mode='r')[:info['pairs']],np.load(folder/'y.npy',mmap_mode='r')[:info['pairs']]
    prepare_extra(size)
    con=connect('work/accuracy_composite.duckdb','2GB',1,read_only=True)
    ids={r[0] for r in con.execute('SELECT entity_id FROM extra_queries').fetchall()}
    truth={}
    with Path('dataset/train/train_ground_truth.tsv').open(encoding='utf-8',newline='') as stream:
        for r in csv.DictReader(stream,delimiter='\t'):
            if r['source1_entity_id'] in ids:
                truth[r['source1_entity_id']]=set(filter(None,r['matched_entity_ids'].split(',')))
    if set(truth)!=ids: raise ValueError('Extra training labels missing')
    x=np.lib.format.open_memmap(folder/'x.npy',mode='w+',dtype=np.float32,shape=(size*k,36))
    y=np.lib.format.open_memmap(folder/'y.npy',mode='w+',dtype=np.uint8,shape=(size*k,))
    offset=0
    for lo in range(1,size+1,500):
        rows=con.execute('''SELECT q.qid,q.entity_id,s.name,s.address,s.country,
            r.entity_id,r.name,r.address,r.country,p.retrieval_score
            FROM extra_pool p JOIN extra_queries q USING(qid)
            JOIN records s ON s.rid=q.rid JOIN records r ON r.rid=p.rid
            WHERE q.qid BETWEEN ? AND ? ORDER BY q.qid,r.entity_id''',[lo,lo+499]).fetchall()
        features=model.extractor.transform(rows)
        selected=final_candidates(rows,features,k)
        end=offset+len(selected)
        x[offset:end]=features[selected]
        y[offset:end]=[rows[i][5] in truth[rows[i][1]] for i in selected]
        offset=end
        log(f'Extra training features through query {min(lo+499,size):,}: {offset:,} pairs')
    con.close(); x.flush(); y.flush()
    marker.write_text(json.dumps(dict(k=k,size=size,pairs=offset)),encoding='utf-8')
    return x[:offset],y[:offset]


def run():
    out=Path('artifacts/accuracy')
    base=MatchingModel.load('artifacts/model.joblib')
    model=MatchingModel(EnhancedExtractor(base.extractor),2026,2)
    extra_x,extra_y=extra_arrays(model)
    data=collect(database='work/accuracy_composite.duckdb',output_dir=out/'composite')
    selected=final_candidates(data['rows'],data['x'],64)
    rows=[data['rows'][i] for i in selected]
    train=np.array([data['queries'][r[1]]['cohort']=='train' for r in rows])
    labels=np.array([r[5] in data['truth'][r[1]] for r in rows],dtype=np.uint8)
    extras=np.load(out/'composite_extra.npy',mmap_mode='r')
    features=np.column_stack([data['x'][selected],extras[selected]])
    train_x=np.concatenate([features[train],extra_x]); train_y=np.concatenate([labels[train],extra_y])
    val_x=features[~train]
    val_ids={r[1] for r,m in zip(rows,train) if not m}
    validation={'rows':[r for r,m in zip(rows,train) if not m],
                'x':data['x'][selected][~train],
                'queries':{s:q for s,q in data['queries'].items() if s in val_ids},
                'truth':{s:t for s,t in data['truth'].items() if data['queries'][s]['cohort']!='train'},
                'threshold':data['threshold']}
    # Include validation entities with zero candidates in the metric denominator.
    validation['queries']={s:q for s,q in data['queries'].items() if q['cohort']!='train'}
    del data,features,rows,extras,extra_x,extra_y
    gc.collect()
    model.classifier.set_params(n_estimators=600,num_leaves=63,min_child_samples=60,reg_lambda=5.)
    log(f'Fitting expanded training model: {len(train_y):,} pairs from 42,000 S1 entities')
    model.classifier.fit(train_x,train_y,feature_name=FEATURE_NAMES+EXTRA_NAMES)
    raw=model.probabilities(val_x); indices=np.arange(len(val_x))
    policy=fit_policy(validation,indices,raw); model.threshold=policy.thresholds['global']
    report=policy_report(validation,indices,raw,policy)
    report.update(strategy='composite',top_k=64,max_df=4096,pool_size=128,
                  enhanced=True,training_entities=42000,training_pairs=len(train_y))
    name='composite_k64_enhanced_more_training'
    model.save(out/f'{name}.joblib')
    reports=json.loads((out/'results.json').read_text()); reports[name]=report
    (out/'results.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    log(f'{name}: audit={score_of(report):.8f}; selection={report["selection_macro_f05"]:.8f}')


if __name__=='__main__': run()
