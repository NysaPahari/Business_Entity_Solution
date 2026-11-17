"""Calibration-only screen of complementary, label-free posting rankings."""
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
        queries=con.execute('SELECT qid,entity_id FROM probe_queries ORDER BY qid').fetchall()
        if len(queries)!=300: raise ValueError('Run recall_probe first')
        con.execute('CREATE TABLE IF NOT EXISTS diverse_pool(qid BIGINT,rid BIGINT,score DOUBLE,base_rank BIGINT,rare_rank BIGINT,count_rank BIGINT)')
        con.execute('CREATE TABLE IF NOT EXISTS diverse_progress(qid BIGINT)')
        done={r[0] for r in con.execute('SELECT qid FROM diverse_progress').fetchall()}
        for lo in range(0,len(queries),50):
            ids=[q for q,_ in queries[lo:lo+50] if q not in done]
            if not ids: continue
            selected=','.join(map(str,ids))
            con.execute('BEGIN')
            con.execute(f'''INSERT INTO diverse_pool
                WITH aggregated AS (
                    SELECT q.qid,r.rid,sum(1.0/sqrt(r.df)) score,
                           sum(1.0/r.df) rare,count(*) hits
                    FROM original.query_keys q JOIN original.ref_keys r USING(key)
                    WHERE q.qid IN ({selected}) GROUP BY q.qid,r.rid
                ), ranked AS (
                    SELECT qid,rid,score,
                           row_number() OVER(PARTITION BY qid ORDER BY score DESC,rid) base_rank,
                           row_number() OVER(PARTITION BY qid ORDER BY rare DESC,score DESC,rid) rare_rank,
                           row_number() OVER(PARTITION BY qid ORDER BY hits DESC,score DESC,rid) count_rank
                    FROM aggregated
                ) SELECT * FROM ranked WHERE base_rank<=128 OR rare_rank<=128 OR count_rank<=128''')
            con.executemany('INSERT INTO diverse_progress VALUES (?)',[(q,) for q in ids])
            con.execute('COMMIT')
            log(f'Diverse retrieval: {lo+len(ids)}/300')
        ids={s for _,s in queries}; truth={}
        with Path('dataset/train/train_ground_truth.tsv').open(encoding='utf-8',newline='') as stream:
            for row in csv.DictReader(stream,delimiter='\t'):
                if row['source1_entity_id'] in ids:
                    truth[row['source1_entity_id']]=set(filter(None,row['matched_entity_ids'].split(',')))
        if set(truth)!=ids: raise ValueError('Missing labels')
        names=('baseline128','baseline_plus_rare','baseline_plus_count','three_rankings')
        candidates={n:defaultdict(set) for n in names}
        predictions={n:defaultdict(set) for n in names}
        model=MatchingModel.load('artifacts/accuracy/selected/model.joblib')
        # Retrieval features retain their original sum(1/sqrt(df)) definition.
        # Scores from alternative rankings only choose candidates, not features.
        for lo in range(0,len(queries),50):
            selected=','.join(str(q) for q,_ in queries[lo:lo+50])
            rows=con.execute(f'''SELECT q.qid,q.entity_id,s.name,s.address,s.country,
                r.entity_id,r.name,r.address,r.country,p.score,p.base_rank,p.rare_rank,p.count_rank
                FROM diverse_pool p JOIN probe_queries q USING(qid)
                JOIN original.records s ON s.rid=q.rid JOIN original.records r ON r.rid=p.rid
                WHERE q.qid IN ({selected}) ORDER BY q.qid,r.entity_id''').fetchall()
            scores=model.probabilities(model.extractor.transform([r[:10] for r in rows]))
            for r,p in zip(rows,scores):
                base,rare,count=r[10]<=128,r[11]<=128,r[12]<=128
                for name,keep in zip(names,(base,base or rare,base or count,True)):
                    if keep:
                        candidates[name][r[1]].add(r[5])
                        if p>=model.threshold: predictions[name][r[1]].add(r[5])
            log(f'Diverse scoring: {lo+len(queries[lo:lo+50])}/300')
        reports={}
        total=sum(map(len,truth.values()))
        for name in names:
            recovered=sum(len(t&candidates[name][s]) for s,t in truth.items())
            reports[name]=dict(true_pairs=total,recovered_true_pairs=recovered,recall=recovered/total,
                mean_candidates=sum(map(len,candidates[name].values()))/len(truth),
                oracle_macro_f05=sum(entity_f05(t,t&candidates[name][s]) for s,t in truth.items())/len(truth),
                frozen_model_macro_f05=sum(entity_f05(t,predictions[name][s]) for s,t in truth.items())/len(truth))
        control=json.loads(Path('artifacts/accuracy/recall_probe/results.json').read_text())['results']['128']
        if abs(control['frozen_model_macro_f05']-reports['baseline128']['frozen_model_macro_f05'])>1e-10:
            raise ValueError('Control does not reproduce previous baseline')
        result=dict(protocol='Same 300 calibration entities; frozen selected classifier and cutoff. Label-free ranking alternatives. No audit access or automatic promotion.',results=reports)
        Path('artifacts/accuracy/recall_probe/diverse_results.json').write_text(json.dumps(result,indent=2))
        log(json.dumps(result))
    finally: con.close()


if __name__=='__main__': run()
