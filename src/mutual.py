"""Cross-fitted forward/reverse nearest-neighbor agreement.

Reverse retrieval searches all training S1 records, not just the validation cohort.
It is approximate over the same bounded retrieval mechanism. Ties count as nearest
neighbors; no unsupported one-to-one assignment constraint is imposed.
"""
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from .blocking import build_index,generate_pool,PAIR_SQL
from .data import connect,sql_string,log
from .infer import grouped_batches
from .model import MatchingModel


def fold_for(sid):
    return int.from_bytes(hashlib.sha256(('mutual-2026:'+sid).encode()).digest()[:4],'little')%3


def fit_rank_models(data,out='artifacts/refinement'):
    path=Path(out)/'rank_models.joblib'
    if path.exists():
        return joblib.load(path)
    original=MatchingModel.load('artifacts/model.joblib')
    selected=data['selected']
    rows=[data['rows'][i] for i in selected]
    train=np.array([data['queries'][r[1]]['cohort']=='train' for r in rows])
    folds=np.array([fold_for(r[1]) for r in rows])
    labels=np.array([r[5] in data['truth'][r[1]] for r in rows],dtype=np.uint8)
    models=[original]
    for fold in range(3):
        use=train&(folds!=fold)
        model=MatchingModel(original.extractor,2026,2).fit(data['x'][selected][use],labels[use])
        models.append(model)
        log(f'Mutual ranker fold {fold}: fitted {use.sum():,} pairs, excluded its own S1 fold')
    joblib.dump(models,path,compress=3)
    return models


def prepare_reverse(database='work/train.duckdb',reverse='work/refinement_reverse.duckdb'):
    marker=Path(str(reverse)+'.ingested.json')
    if marker.exists():
        return Path(reverse)
    con=connect(reverse,'2GB',2)
    try:
        con.execute(f'ATTACH {sql_string(Path(database).resolve().as_posix())} AS base (READ_ONLY)')
        con.execute('''CREATE OR REPLACE TABLE records AS
            SELECT rid,entity_id,name,address,country,CASE WHEN source=1 THEN 2 ELSE 1 END source
            FROM base.records WHERE source=1 OR rid IN (SELECT rid FROM base.pool)''')
        con.execute('''CREATE OR REPLACE TABLE queries AS SELECT row_number() OVER(ORDER BY entity_id)::BIGINT qid,
            rid,entity_id,'reverse' AS cohort FROM records WHERE source=1''')
        counts=con.execute('SELECT source,count(*) FROM records GROUP BY source').fetchall()
        con.execute('DETACH base'); con.execute('CHECKPOINT')
    finally:
        con.close()
    marker.write_text(json.dumps(counts))
    log(f'Reverse-search query/reference counts: {counts}')
    return Path(reverse)


_models=None


def _init_models(path):
    global _models
    _models=joblib.load(path)


def _reverse_batch(rows,needed):
    # Restore S1-left, S2/S3-right orientation before computing model features.
    oriented=[(r[0],r[5],r[6],r[7],r[8],r[1],r[2],r[3],r[4],r[9]) for r in rows]
    x=_models[0].extractor.transform(oriented)
    masks=np.asarray(needed,dtype=np.uint8)
    scores=[]
    for k,model in enumerate(_models):
        use=(masks&(1<<k))!=0
        p=np.full(len(rows),np.nan)
        if use.any():
            p[use]=model.probabilities(x[use])
        scores.append(p)
    result={}
    start=0
    while start<len(rows):
        end=start+1
        while end<len(rows) and rows[end][0]==rows[start][0]:
            end+=1
        winners=[]
        for k,p in enumerate(scores):
            if not needed[start]&(1<<k):
                winners.append(())
                continue
            maximum=p[start:end].max()
            winners.append(tuple(rows[j][5] for j in range(start,end) if p[j]>=maximum-1e-12))
        result[rows[start][1]]=winners
        start=end
    return rows[-1][0],result


def reverse_nearest(database,rank_model_path,output_path,needed_models,workers=4):
    output_path=Path(output_path)
    if output_path.exists():
        return joblib.load(output_path)
    checkpoint=Path(str(output_path)+'.progress')
    state=joblib.load(checkpoint) if checkpoint.exists() else {'through':0,'result':{}}
    result=state['result']; through=state['through']
    con=connect(database,'2GB',2,read_only=True)
    end=con.execute('SELECT max(qid) FROM queries').fetchone()[0]
    def batches():
        for lo in range(through+1,end+1,5000):
            cursor=con.execute(PAIR_SQL+' WHERE q.qid BETWEEN ? AND ? ORDER BY q.qid,r.entity_id',[lo,lo+4999])
            yield from grouped_batches(cursor,12000)
    last_log=len(result); last_checkpoint=len(result)
    def consume(future):
        nonlocal through,last_checkpoint
        through,new=future.result()
        result.update(new)
        if len(result)>=last_checkpoint+25000:
            temporary=Path(str(checkpoint)+'.tmp')
            joblib.dump({'through':through,'result':result},temporary,compress=1)
            temporary.replace(checkpoint)
            last_checkpoint=len(result)
    try:
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'),
                                 initializer=_init_models,initargs=(rank_model_path,)) as executor:
            pending=deque()
            for rows in batches():
                pending.append(executor.submit(_reverse_batch,rows,[needed_models[r[1]] for r in rows]))
                if len(pending)>=workers+1:
                    consume(pending.popleft())
                if len(result)>=last_log+10000:
                    log(f'Reverse neighbors scored for {len(result):,} / {end:,} reference candidates')
                    last_log=len(result)
            while pending:
                consume(pending.popleft())
    finally:
        con.close()
    joblib.dump(result,output_path,compress=3)
    checkpoint.unlink(missing_ok=True)
    return result


def mutual_features(data,models,reverse,indices=None):
    indices=data['selected'] if indices is None else np.asarray(indices)
    rows=[data['rows'][i] for i in indices]
    rank_model=np.array([1+fold_for(r[1]) if data['queries'][r[1]]['cohort']=='train' else 0 for r in rows])
    scores=np.zeros(len(rows))
    for k,model in enumerate(models):
        mask=rank_model==k
        scores[mask]=model.probabilities(data['x'][indices][mask])
    highest=defaultdict(lambda:float('-inf'))
    for r,p in zip(rows,scores):
        highest[r[1]]=max(highest[r[1]],p)
    feature=np.array([float(p>=highest[r[1]]-1e-12 and r[1] in reverse.get(r[5],[(),(),(),()])[k])
                      for r,p,k in zip(rows,scores,rank_model)],dtype=np.float32)
    return feature


def run(data,out='artifacts/refinement'):
    out=Path(out)
    models=fit_rank_models(data,out)
    database=prepare_reverse()
    build_index(database,128,'2GB',2)
    generate_pool(database,32,memory='2GB',threads=2)
    needed=defaultdict(int)
    for r in data['rows']:
        k=1+fold_for(r[1]) if data['queries'][r[1]]['cohort']=='train' else 0
        needed[r[5]]|=1<<k
    nearest=reverse_nearest(database,out/'rank_models.joblib',out/'reverse_nearest.joblib',needed)
    feature=mutual_features(data,models,nearest)
    np.save(out/'mutual_feature.npy',feature)
    log(f'Mutual-neighbor feature complete: {int(feature.sum()):,} of {len(feature):,} pairs flagged')
    return feature


if __name__=='__main__':
    from .error_analysis import collect
    run(collect())
