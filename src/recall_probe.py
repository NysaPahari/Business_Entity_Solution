"""Screen wider candidate pools on calibration only; never promote automatically."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from .data import connect, log, sql_string
from .evaluate import entity_f05
from .model import MatchingModel


def run():
    out = Path('artifacts/accuracy/recall_probe')
    out.mkdir(exist_ok=True)
    con = connect('work/recall_probe.duckdb','768MB',1)
    try:
        con.execute(f"ATTACH {sql_string(Path('work/accuracy_composite.duckdb').resolve().as_posix())} AS original (READ_ONLY)")
        con.execute("CREATE TABLE IF NOT EXISTS probe_queries AS SELECT * FROM original.queries WHERE cohort='calibration' ORDER BY hash(entity_id || 'recall-probe-2026') LIMIT 300")
        queries = con.execute('SELECT qid,entity_id FROM probe_queries ORDER BY qid').fetchall()
        con.execute('CREATE TABLE IF NOT EXISTS probe_pool(qid BIGINT,rid BIGINT,score DOUBLE,rank BIGINT)')
        con.execute('CREATE TABLE IF NOT EXISTS progress(qid BIGINT)')
        completed = {r[0] for r in con.execute('SELECT qid FROM progress').fetchall()}
        for lo in range(0,len(queries),50):
            ids = [qid for qid,_ in queries[lo:lo+50] if qid not in completed]
            if not ids: continue
            selected = ','.join(map(str,ids))
            con.execute('BEGIN')
            con.execute(f'''INSERT INTO probe_pool
                SELECT q.qid,r.rid,sum(1.0/sqrt(r.df)) score,
                       row_number() OVER(PARTITION BY q.qid ORDER BY score DESC,r.rid) rank
                FROM original.query_keys q JOIN original.ref_keys r USING(key)
                WHERE q.qid IN ({selected}) GROUP BY q.qid,r.rid QUALIFY rank<=512''')
            con.executemany('INSERT INTO progress VALUES (?)',[(qid,) for qid in ids])
            con.execute('COMMIT')
            log(f'Wider retrieval complete for {lo+len(ids)}/{len(queries)} calibration queries')
        ids = {s for _,s in queries}
        truth = {}
        with Path('dataset/train/train_ground_truth.tsv').open(encoding='utf-8',newline='') as stream:
            for r in csv.DictReader(stream,delimiter='\t'):
                if r['source1_entity_id'] in ids:
                    truth[r['source1_entity_id']] = set(filter(None,r['matched_entity_ids'].split(',')))
        if set(truth)!=ids: raise ValueError('Missing calibration labels')
        model = MatchingModel.load('artifacts/accuracy/selected/model.joblib')
        candidates = {k:defaultdict(set) for k in (128,256,512)}
        predictions = {k:defaultdict(set) for k in candidates}
        for lo in range(0,len(queries),50):
            selected = ','.join(str(q) for q,_ in queries[lo:lo+50])
            batch = con.execute(f'''SELECT q.qid,q.entity_id,s.name,s.address,s.country,
                r.entity_id,r.name,r.address,r.country,p.score,p.rank
                FROM probe_pool p JOIN probe_queries q USING(qid)
                JOIN original.records s ON s.rid=q.rid JOIN original.records r ON r.rid=p.rid
                WHERE q.qid IN ({selected}) ORDER BY q.qid,p.rank''').fetchall()
            scores = model.probabilities(model.extractor.transform([r[:10] for r in batch]))
            for r,p in zip(batch,scores):
                for k in candidates:
                    if r[10]<=k:
                        candidates[k][r[1]].add(r[5])
                        if p>=model.threshold: predictions[k][r[1]].add(r[5])
        reports = {}
        for k in candidates:
            recovered = sum(len(t & candidates[k][s]) for s,t in truth.items())
            total = sum(map(len,truth.values()))
            reports[k] = dict(recovered_true_pairs=recovered,true_pairs=total,recall=recovered/total,
                oracle_macro_f05=sum(entity_f05(t,t & candidates[k][s]) for s,t in truth.items())/len(truth),
                frozen_model_macro_f05=sum(entity_f05(t,predictions[k][s]) for s,t in truth.items())/len(truth))
        result = dict(entities=len(ids),protocol='Deterministic 300-entity calibration screen. Frozen model/cutoff. Same composite keys and max_df4096. No audit access; not a generalization estimate.',results=reports)
        (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        log(json.dumps(result))
    finally:
        con.close()


if __name__=='__main__': run()
