"""Diagnose calibration misses without injecting labelled pairs into retrieval."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from .blocking import key_sql
from .data import connect, sql_string


def run():
    con=connect('work/recall_probe.duckdb','768MB',1)
    try:
        con.execute(f"ATTACH {sql_string(Path('work/accuracy_composite.duckdb').resolve().as_posix())} AS original (READ_ONLY)")
        ids={r[0] for r in con.execute('SELECT entity_id FROM probe_queries').fetchall()}
        candidates=defaultdict(set)
        for sid,cid in con.execute('SELECT q.entity_id,r.entity_id FROM probe_pool p JOIN probe_queries q USING(qid) JOIN original.records r ON r.rid=p.rid').fetchall():
            candidates[sid].add(cid)
        missed=[]
        with Path('dataset/train/train_ground_truth.tsv').open(encoding='utf-8',newline='') as stream:
            for r in csv.DictReader(stream,delimiter='\t'):
                sid=r['source1_entity_id']
                if sid in ids:
                    missed.extend((sid,cid) for cid in set(filter(None,r['matched_entity_ids'].split(',')))-candidates[sid])
        con.execute('CREATE TEMP TABLE missed_ids(sid VARCHAR,cid VARCHAR)')
        if missed: con.executemany('INSERT INTO missed_ids VALUES (?,?)',missed)
        con.execute('CREATE TEMP TABLE missed_pairs AS SELECT m.*,q.qid,q.rid qr,r.rid rr FROM missed_ids m JOIN probe_queries q ON q.entity_id=m.sid JOIN original.records r ON r.entity_id=m.cid AND r.source<>1')
        sql=key_sql('composite').replace('FROM records','FROM original.records WHERE rid IN (SELECT qr FROM missed_pairs UNION SELECT rr FROM missed_pairs)')
        raw=defaultdict(set)
        for rid,_,key in con.execute(sql).fetchall(): raw[rid].add(key)
        retained=defaultdict(set)
        for qid,key in con.execute('SELECT qid,key FROM original.query_keys WHERE qid IN (SELECT qid FROM missed_pairs)').fetchall(): retained[qid].add(key)
        counts=Counter()
        examples=[]
        for sid,cid,qid,qr,rr in con.execute('SELECT * FROM missed_pairs').fetchall():
            shared=raw[qr]&raw[rr]
            category='no_shared_key' if not shared else 'shared_keys_filtered' if not shared&retained[qid] else 'rank_beyond_512'
            counts[category]+=1
            examples.append(dict(source1_entity_id=sid,candidate_entity_id=cid,reason=category))
        result=dict(protocol='Calibration-only diagnostic of missing labelled pairs; not retrieval candidates or training additions.',missing_pairs=len(missed),classified_pairs=len(examples),reasons=dict(counts),examples=examples)
        Path('artifacts/accuracy/recall_probe/miss_diagnosis.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k!='examples'}),flush=True)
    finally: con.close()


if __name__=='__main__': run()
