"""One-hop expansion from predicted anchors, with no automatic graph merges."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from .data import connect, log, sql_string
from .evaluate import entity_f05
from .finalize_improved import digest
from .model import MatchingModel


def choose_anchors(scored, minimum=.98, limit=3):
    """Choose deterministic high-confidence anchors without labels."""
    groups=defaultdict(list)
    for qid,rid,probability in scored:
        if probability>=minimum: groups[qid].append((rid,probability))
    return [(qid,rid,p) for qid,items in sorted(groups.items())
            for rid,p in sorted(items,key=lambda x:(-x[1],x[0]))[:limit]]


def run():
    out=Path('artifacts/accuracy/graph_probe'); out.mkdir(exist_ok=True)
    model_path=Path('artifacts/accuracy/selected/model.joblib')
    config=dict(model_sha256=digest(model_path),anchor_threshold=.98,max_anchors=3,expansion_k=128)
    marker=out/'config.json'
    if marker.exists() and json.loads(marker.read_text())!=config:
        raise ValueError('Graph cache configuration changed')
    marker.write_text(json.dumps(config,indent=2))
    model=MatchingModel.load(model_path)
    con=connect('work/recall_probe.duckdb','512MB',1)
    try:
        con.execute(f"ATTACH {sql_string(Path('work/accuracy_composite.duckdb').resolve().as_posix())} AS original (READ_ONLY)")
        queries=con.execute('SELECT qid,entity_id FROM probe_queries ORDER BY qid').fetchall()
        if len(queries)!=300: raise ValueError('Expected existing 300 calibration queries')
        con.execute('CREATE TABLE IF NOT EXISTS graph_baseline(qid BIGINT,rid BIGINT,probability DOUBLE)')
        con.execute('CREATE TABLE IF NOT EXISTS graph_baseline_progress(qid BIGINT)')
        done={r[0] for r in con.execute('SELECT qid FROM graph_baseline_progress').fetchall()}
        for lo in range(0,len(queries),50):
            ids=[q for q,_ in queries[lo:lo+50] if q not in done]
            if not ids: continue
            selected=','.join(map(str,ids))
            rows=con.execute(f'''SELECT q.qid,q.entity_id,s.name,s.address,s.country,
                r.entity_id,r.name,r.address,r.country,p.score,p.rid
                FROM probe_pool p JOIN probe_queries q USING(qid)
                JOIN original.records s ON s.rid=q.rid JOIN original.records r ON r.rid=p.rid
                WHERE p.rank<=128 AND q.qid IN ({selected}) ORDER BY q.qid,r.entity_id''').fetchall()
            scores=model.probabilities(model.extractor.transform([r[:10] for r in rows]))
            con.execute('BEGIN')
            con.executemany('INSERT INTO graph_baseline VALUES (?,?,?)',[(r[0],r[10],float(p)) for r,p in zip(rows,scores)])
            con.executemany('INSERT INTO graph_baseline_progress VALUES (?)',[(q,) for q in ids])
            con.execute('COMMIT')
        anchors=choose_anchors(con.execute('SELECT * FROM graph_baseline').fetchall())
        con.execute('CREATE OR REPLACE TEMP TABLE anchors(qid BIGINT,rid BIGINT,probability DOUBLE)')
        if anchors: con.executemany('INSERT INTO anchors VALUES (?,?,?)',anchors)
        log(f'Selected {len(anchors)} predicted anchors for {len({q for q,_,_ in anchors})}/300 queries')
        con.execute('CREATE TEMP TABLE anchor_keys AS SELECT a.qid,a.rid anchor,k.key FROM anchors a JOIN original.ref_keys k USING(rid)')
        con.execute('CREATE TABLE IF NOT EXISTS graph_pool(qid BIGINT,rid BIGINT,bridge_score DOUBLE)')
        con.execute('CREATE TABLE IF NOT EXISTS graph_progress(qid BIGINT)')
        done={r[0] for r in con.execute('SELECT qid FROM graph_progress').fetchall()}
        for lo in range(0,len(queries),50):
            ids=[q for q,_ in queries[lo:lo+50] if q not in done]
            if not ids: continue
            selected=','.join(map(str,ids))
            con.execute('BEGIN')
            con.execute(f'''INSERT INTO graph_pool WITH links AS (
                SELECT a.qid,a.anchor,r.rid,sum(1.0/sqrt(r.df)) score
                FROM anchor_keys a JOIN original.ref_keys r USING(key)
                WHERE a.qid IN ({selected}) AND a.anchor<>r.rid
                GROUP BY a.qid,a.anchor,r.rid
            ) SELECT qid,rid,max(score) bridge_score FROM links GROUP BY qid,rid
              QUALIFY row_number() OVER(PARTITION BY qid ORDER BY bridge_score DESC,rid)<=128''')
            con.executemany('INSERT INTO graph_progress VALUES (?)',[(q,) for q in ids])
            con.execute('COMMIT')
            log(f'Graph retrieval: {lo+len(ids)}/300')
        con.execute('''CREATE TEMP TABLE graph_union AS SELECT qid,rid FROM graph_baseline
                       UNION SELECT qid,rid FROM graph_pool''')
        con.execute('''CREATE TEMP TABLE graph_scores AS SELECT m.qid,m.rid,sum(1.0/sqrt(r.df)) score
            FROM graph_union m JOIN original.query_keys q USING(qid)
            JOIN original.ref_keys r ON r.key=q.key AND r.rid=m.rid GROUP BY m.qid,m.rid''')
        ids={s for _,s in queries}; truth={}
        with Path('dataset/train/train_ground_truth.tsv').open(encoding='utf-8',newline='') as stream:
            for r in csv.DictReader(stream,delimiter='\t'):
                if r['source1_entity_id'] in ids:
                    truth[r['source1_entity_id']]=set(filter(None,r['matched_entity_ids'].split(',')))
        if set(truth)!=ids: raise ValueError('Missing labels')
        candidates={k:defaultdict(set) for k in ('baseline128','graph_expanded')}
        predictions={k:defaultdict(set) for k in candidates}
        for lo in range(0,len(queries),50):
            selected=','.join(str(q) for q,_ in queries[lo:lo+50])
            rows=con.execute(f'''SELECT q.qid,q.entity_id,s.name,s.address,s.country,
                r.entity_id,r.name,r.address,r.country,coalesce(f.score,0),b.probability
                FROM graph_union m JOIN probe_queries q USING(qid)
                JOIN original.records s ON s.rid=q.rid JOIN original.records r ON r.rid=m.rid
                LEFT JOIN graph_scores f ON f.qid=m.qid AND f.rid=m.rid
                LEFT JOIN graph_baseline b ON b.qid=m.qid AND b.rid=m.rid
                WHERE q.qid IN ({selected}) ORDER BY q.qid,r.entity_id''').fetchall()
            scores=model.probabilities(model.extractor.transform([r[:10] for r in rows]))
            for r,p in zip(rows,scores):
                for k in candidates:
                    if k=='baseline128' and r[10] is None: continue
                    candidates[k][r[1]].add(r[5])
                    if p>=model.threshold: predictions[k][r[1]].add(r[5])
            log(f'Graph scoring: {lo+len(queries[lo:lo+50])}/300')
        reports={}; total=sum(map(len,truth.values()))
        for k in candidates:
            recovered=sum(len(t&candidates[k][s]) for s,t in truth.items())
            reports[k]=dict(recall=recovered/total,recovered_true_pairs=recovered,true_pairs=total,
                mean_candidates=sum(map(len,candidates[k].values()))/len(truth),
                oracle_macro_f05=sum(entity_f05(t,t&candidates[k][s]) for s,t in truth.items())/len(truth),
                frozen_model_macro_f05=sum(entity_f05(t,predictions[k][s]) for s,t in truth.items())/len(truth),
                false_positives=sum(len(predictions[k][s]-t) for s,t in truth.items()),
                false_negatives=sum(len(t-predictions[k][s]) for s,t in truth.items()))
        control=json.loads(Path('artifacts/accuracy/recall_probe/results.json').read_text())['results']['128']
        if abs(control['frozen_model_macro_f05']-reports['baseline128']['frozen_model_macro_f05'])>1e-10:
            raise ValueError('Baseline control mismatch')
        result=dict(protocol='Same 300 calibration queries. Up to three predicted anchors at >=0.98; one expansion hop; no transitive acceptance, no retraining or audit access.',config=config,anchors=len(anchors),results=reports)
        (out/'results.json').write_text(json.dumps(result,indent=2))
        log(json.dumps(result))
    finally: con.close()


if __name__=='__main__': run()
