"""Screen exact-field retrieval outside frequency-pruned keys, on calibration."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from .data import connect, log, sql_string
from .evaluate import entity_f05
from .model import MatchingModel


def run():
    con=connect('work/recall_probe.duckdb','512MB',1)
    try:
        con.execute(f"ATTACH {sql_string(Path('work/accuracy_composite.duckdb').resolve().as_posix())} AS original (READ_ONLY)")
        con.execute('CREATE TEMP TABLE query_text AS SELECT q.qid,q.entity_id,r.name,r.address FROM probe_queries q JOIN original.records r USING(rid)')
        queries=con.execute('SELECT qid,entity_id FROM query_text ORDER BY qid').fetchall()
        con.execute('CREATE TABLE IF NOT EXISTS exact_pool(qid BIGINT,rid BIGINT,channel VARCHAR)')
        con.execute('CREATE TABLE IF NOT EXISTS exact_progress(channel VARCHAR)')
        done={r[0] for r in con.execute('SELECT channel FROM exact_progress').fetchall()}
        for field,other in (('name','address'),('address','name')):
            if field in done: continue
            log(f'Retrieving by exact {field}, reranking by {other}')
            con.execute('BEGIN')
            con.execute(f'''INSERT INTO exact_pool
                SELECT q.qid,r.rid,'{field}' FROM query_text q JOIN original.records r
                ON q.{field}=r.{field} AND q.{field}<>'' AND r.source<>1
                QUALIFY row_number() OVER(PARTITION BY q.qid ORDER BY
                    levenshtein(q.{other},r.{other})::DOUBLE/greatest(length(q.{other}),length(r.{other}),1),r.rid)<=128''')
            con.execute('INSERT INTO exact_progress VALUES (?)',[field])
            con.execute('COMMIT')
        log('Computing original retrieval features for fallback candidates')
        con.execute('''CREATE TEMP TABLE merged AS SELECT qid,rid,max(is_base) is_base FROM
            (SELECT qid,rid,true is_base FROM probe_pool WHERE rank<=128
             UNION ALL SELECT qid,rid,false FROM exact_pool) GROUP BY qid,rid''')
        con.execute('''CREATE TEMP TABLE merged_scores AS
            SELECT m.qid,m.rid,sum(1.0/sqrt(r.df)) score FROM merged m
            JOIN original.query_keys q USING(qid)
            JOIN original.ref_keys r ON r.key=q.key AND r.rid=m.rid GROUP BY m.qid,m.rid''')
        ids={s for _,s in queries}; truth={}
        with Path('dataset/train/train_ground_truth.tsv').open(encoding='utf-8',newline='') as stream:
            for r in csv.DictReader(stream,delimiter='\t'):
                if r['source1_entity_id'] in ids:
                    truth[r['source1_entity_id']]=set(filter(None,r['matched_entity_ids'].split(',')))
        if set(truth)!=ids: raise ValueError('Missing truth')
        model=MatchingModel.load('artifacts/accuracy/selected/model.joblib')
        candidates={k:defaultdict(set) for k in ('baseline128','with_exact_fallback')}
        predictions={k:defaultdict(set) for k in candidates}
        for lo in range(0,len(queries),50):
            selected=','.join(str(q) for q,_ in queries[lo:lo+50])
            rows=con.execute(f'''SELECT q.qid,q.entity_id,q.name,q.address,s.country,
                r.entity_id,r.name,r.address,r.country,coalesce(f.score,0),m.is_base
                FROM merged m JOIN query_text q USING(qid) JOIN probe_queries pq USING(qid)
                JOIN original.records s ON s.rid=pq.rid JOIN original.records r ON r.rid=m.rid
                LEFT JOIN merged_scores f ON f.qid=m.qid AND f.rid=m.rid
                WHERE q.qid IN ({selected}) ORDER BY q.qid,r.entity_id''').fetchall()
            scores=model.probabilities(model.extractor.transform([r[:10] for r in rows]))
            for r,p in zip(rows,scores):
                for k in candidates:
                    if k=='baseline128' and not r[10]: continue
                    candidates[k][r[1]].add(r[5])
                    if p>=model.threshold: predictions[k][r[1]].add(r[5])
            log(f'Exact fallback scoring: {lo+len(queries[lo:lo+50])}/300')
        reports={}; total=sum(map(len,truth.values()))
        for k in candidates:
            recovered=sum(len(t&candidates[k][s]) for s,t in truth.items())
            reports[k]=dict(recall=recovered/total,recovered_true_pairs=recovered,true_pairs=total,
                mean_candidates=sum(map(len,candidates[k].values()))/len(truth),
                oracle_macro_f05=sum(entity_f05(t,t&candidates[k][s]) for s,t in truth.items())/len(truth),
                frozen_model_macro_f05=sum(entity_f05(t,predictions[k][s]) for s,t in truth.items())/len(truth))
        control=json.loads(Path('artifacts/accuracy/recall_probe/results.json').read_text())['results']['128']
        if abs(control['frozen_model_macro_f05']-reports['baseline128']['frozen_model_macro_f05'])>1e-10:
            raise ValueError('Baseline control mismatch')
        result=dict(protocol='Same 300 calibration entities, frozen model/cutoff; exact nonempty fields retrieve candidates only, never force acceptance. No audit access or promotion.',results=reports)
        Path('artifacts/accuracy/recall_probe/exact_results.json').write_text(json.dumps(result,indent=2))
        log(json.dumps(result))
    finally: con.close()


if __name__=='__main__': run()
