"""Train residual group-consistency models on previously unused training entities."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
from rapidfuzz import fuzz
from scipy.special import expit, logit
from .blocking import key_sql
from .data import connect, log, sql_string
from .evaluate import entity_f05
from .finalize_improved import digest
from .model import MatchingModel
from .refinement import calibration_role, threshold_with_overrides


def group_features(rows, probabilities):
    """Other candidates supply evidence; a record cannot support itself."""
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        if probabilities[i]>=.98: groups[r[0]].append(i)
    result=[]
    for i,r in enumerate(rows):
        anchors=sorted((j for j in groups[r[0]] if rows[j][5]!=r[5]),
                       key=lambda j:(-probabilities[j],rows[j][5]))[:3]
        evidence=[]
        for j in anchors:
            a=rows[j]
            n=fuzz.token_sort_ratio(r[6],a[6])/100 if r[6] and a[6] else 0.
            d=fuzz.token_sort_ratio(r[7],a[7])/100 if r[7] and a[7] else 0.
            evidence.append((n,d,min(n,d)))
        result.append([len(anchors),*(np.max(evidence,axis=0) if evidence else [0]*3),
                       *(np.mean(evidence,axis=0) if evidence else [0]*3)])
    return np.asarray(result,dtype=np.float32)


def run():
    out=Path('artifacts/accuracy/group_probe'); out.mkdir(exist_ok=True)
    model_path=Path('artifacts/accuracy/selected/model.joblib')
    config=dict(model_sha256=digest(model_path),training_entities=1000,calibration_entities=1000,version=1)
    marker=out/'config.json'
    if marker.exists() and json.loads(marker.read_text())!=config: raise ValueError('Cache configuration mismatch')
    marker.write_text(json.dumps(config,indent=2))
    model=MatchingModel.load(model_path)
    con=connect('work/group_probe.duckdb','512MB',1)
    try:
        con.execute(f"ATTACH {sql_string(Path('work/accuracy_composite.duckdb').resolve().as_posix())} AS original (READ_ONLY)")
        tables={r[0] for r in con.execute('SHOW TABLES').fetchall()}
        if 'queries' not in tables:
            con.execute('''CREATE TABLE queries AS SELECT row_number() OVER(ORDER BY rid)::BIGINT qid,
                rid,entity_id,'train' cohort FROM (SELECT rid,entity_id FROM original.records
                WHERE source=1 AND hash(entity_id || '2026')%10>=2
                AND rid NOT IN (SELECT rid FROM original.queries)
                AND rid NOT IN (SELECT rid FROM original.extra_queries)
                ORDER BY hash(entity_id || 'group-training-2026'),entity_id LIMIT 1000)''')
            con.execute('''INSERT INTO queries SELECT 1000+row_number() OVER(ORDER BY rid),rid,entity_id,'calibration'
                FROM (SELECT * FROM original.queries WHERE cohort='calibration'
                ORDER BY hash(entity_id || 'group-calibration-2026'),entity_id LIMIT 1000)''')
        queries=con.execute('SELECT qid,entity_id,cohort FROM queries ORDER BY qid').fetchall()
        if len(queries)!=2000: raise ValueError('Unexpected cohorts')
        if 'query_keys' not in tables:
            sql=key_sql('composite').replace('FROM records','FROM original.records WHERE rid IN (SELECT rid FROM queries)')
            con.execute('CREATE TABLE query_keys AS SELECT q.qid,k.key FROM ('+sql+') k JOIN queries q USING(rid)')
        con.execute('CREATE TABLE IF NOT EXISTS pool(qid BIGINT,rid BIGINT,score DOUBLE)')
        con.execute('CREATE TABLE IF NOT EXISTS progress(through BIGINT)')
        start=con.execute('SELECT coalesce(max(through),0)+1 FROM progress').fetchone()[0]
        for lo in range(start,2001,100):
            hi=min(lo+99,2000)
            con.execute('BEGIN')
            con.execute(f'''INSERT INTO pool SELECT q.qid,r.rid,sum(1.0/sqrt(r.df)) score
                FROM query_keys q JOIN original.ref_keys r USING(key) WHERE q.qid BETWEEN {lo} AND {hi}
                GROUP BY q.qid,r.rid QUALIFY row_number() OVER(PARTITION BY q.qid ORDER BY score DESC,r.rid)<=128''')
            con.execute('INSERT INTO progress VALUES (?)',[hi]); con.execute('COMMIT')
            log(f'Group trial retrieval: {hi}/2000')
        ids={s for _,s,_ in queries}; truth={}
        with Path('dataset/train/train_ground_truth.tsv').open(encoding='utf-8',newline='') as stream:
            for r in csv.DictReader(stream,delimiter='\t'):
                if r['source1_entity_id'] in ids:
                    truth[r['source1_entity_id']]=set(filter(None,r['matched_entity_ids'].split(',')))
        if set(truth)!=ids: raise ValueError('Missing labels')
        chunks=[]
        for lo in range(1,2001,100):
            cache=out/f'batch_{lo}.joblib'
            if cache.exists(): chunks.append(joblib.load(cache)); continue
            rows=con.execute(f'''SELECT q.qid,q.entity_id,s.name,s.address,s.country,
                r.entity_id,r.name,r.address,r.country,p.score FROM pool p JOIN queries q USING(qid)
                JOIN original.records s ON s.rid=q.rid JOIN original.records r ON r.rid=p.rid
                WHERE q.qid BETWEEN {lo} AND {lo+99} ORDER BY q.qid,r.entity_id''').fetchall()
            x=model.extractor.transform(rows); raw=model.probabilities(x)
            chunk=dict(x=np.column_stack([x,group_features(rows,raw)]),raw=raw,
                       rows=[(r[0],r[1],'','','',r[5]) for r in rows],
                       y=np.asarray([r[5] in truth[r[1]] for r in rows],dtype=np.uint8))
            joblib.dump(chunk,cache,compress=3); chunks.append(chunk)
            log(f'Group features: {lo+99}/2000')
    finally: con.close()
    x=np.concatenate([c['x'] for c in chunks]); raw=np.concatenate([c['raw'] for c in chunks])
    y=np.concatenate([c['y'] for c in chunks]); rows=[r for c in chunks for r in c['rows']]
    train=np.asarray([r[0]<=1000 for r in rows]); init=logit(np.clip(raw,1e-6,1-1e-6))
    fit_truth={s:truth[s] for _,s,c in queries if c=='calibration' and calibration_role(s)=='isotonic_fit'}
    tune_truth={s:truth[s] for _,s,c in queries if c=='calibration' and calibration_role(s)=='threshold_tune'}
    fit=np.asarray([r[1] in fit_truth for r in rows]); fit_rows=[r for r,m in zip(rows,fit) if m]
    def report(scores,cutoff):
        preds=defaultdict(set)
        for r,p in zip(rows,scores):
            if r[1] in tune_truth and p>=cutoff: preds[r[1]].add(r[5])
        return dict(tune_entities=len(tune_truth),macro_f05=sum(entity_f05(t,preds[s]) for s,t in tune_truth.items())/len(tune_truth),
                    fp=sum(len(preds[s]-t) for s,t in tune_truth.items()),fn=sum(len(t-preds[s]) for s,t in tune_truth.items()),cutoff=float(cutoff))
    reports={'selected_frozen':report(raw,model.threshold)}
    for name,width in (('residual_pair_only',36),('residual_group',43)):
        booster=lgb.train(dict(objective='binary',learning_rate=.03,num_leaves=15,min_data_in_leaf=80,
                               lambda_l2=10,num_threads=1,verbosity=-1,seed=2026,deterministic=True,force_col_wise=True),
                          lgb.Dataset(x[train,:width],label=y[train],init_score=init[train]),num_boost_round=150)
        scores=expit(init+booster.predict(x[:,:width],raw_score=True,num_threads=1))
        zeros=np.zeros(int(fit.sum()),dtype=bool)
        cutoff,_=threshold_with_overrides(fit_truth,fit_rows,scores[fit],zeros,zeros)
        reports[name]=report(scores,cutoff)
        booster.save_model(str(out/f'{name}.txt'))
        log(f'{name}: {reports[name]}')
    result=dict(protocol='1000 previously unused training-bucket entities excluded from base-model training; 1000 existing calibration entities split by calibration_role. Residual models trained only on train. Cutoffs fit on calibration fit half, comparisons on tune half. No fresh audit or deployment.',config=config,results=reports)
    (out/'results.json').write_text(json.dumps(result,indent=2)); log(json.dumps(result))


if __name__=='__main__': run()
