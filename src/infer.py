"""Stream the actual final candidate sets and their independently scored matches."""
from pathlib import Path
import json
import hashlib
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from .blocking import PAIR_SQL
from .data import connect,log
from .features import final_candidates
from .model import MatchingModel


_worker_model = None


def _initialize_worker(model_path):
    global _worker_model
    _worker_model = MatchingModel.load(model_path)


def _score_batch(batch, top_k):
    model = _worker_model
    x = model.extractor.transform(batch)
    selected = final_candidates(batch,x,top_k)
    scores = model.probabilities(x[selected])
    candidates,matches = {},{}
    for i,score in zip(selected,scores):
        row = batch[i]
        candidates.setdefault(row[0],[]).append(row[5])
        if score >= model.threshold:
            matches.setdefault(row[0],[]).append(row[5])
    return batch[-1][0],candidates,matches


def scored_batches(batches, model_path, top_k, workers):
    if workers==1:
        _initialize_worker(model_path)
        for batch in batches:
            yield _score_batch(batch,top_k)
        return
    # A bounded number of outstanding batches prevents executor.map from
    # eagerly consuming the complete multimillion-row database cursor.
    with ProcessPoolExecutor(max_workers=workers,initializer=_initialize_worker,
                             initargs=(model_path,),mp_context=get_context('spawn')) as executor:
        pending = deque()
        for batch in batches:
            pending.append(executor.submit(_score_batch,batch,top_k))
            if len(pending)>=workers+1:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def grouped_batches(cursor, target_pairs=16000):
    """Never split one query's candidates across reranking batches."""
    pending = []
    while True:
        chunk = cursor.fetchmany(target_pairs)
        if not chunk:
            if pending:
                yield pending
            break
        pending.extend(chunk)
        last_qid = pending[-1][0]
        boundary = len(pending)-1
        while boundary>0 and pending[boundary-1][0]==last_qid:
            boundary -= 1
        if boundary:
            yield pending[:boundary]
            pending = pending[boundary:]


def infer(database, model_path, output_dir, top_k=20, memory='2GB', threads=2, workers=3):
    output = Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    model = MatchingModel.load(model_path)
    con = connect(database,memory,threads,read_only=True)
    query_con = connect(database,memory,threads,read_only=True)
    end_qid = con.execute('SELECT coalesce(max(qid),0) FROM queries').fetchone()[0]
    checkpoint_path=output/'inference_checkpoint.json'
    paths = [output/'candidate_pairs.tsv',output/'matching_results.tsv']
    temporary = [Path(str(p)+'.tmp') for p in paths]
    config=dict(database=str(Path(database).resolve()),
                model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
                top_k=top_k,end_qid=end_qid)
    for suffix in ('.index.json',):
        marker=Path(str(database)+suffix)
        config[suffix]=marker.read_text() if marker.exists() else None
    ingest=Path(database).parent/'test_ingest.json'
    config['ingest']=ingest.read_text() if ingest.exists() else None
    state=json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else None
    if state and state['config']!=config:
        con.close(); query_con.close()
        raise ValueError('Inference inputs changed; use a new output directory')
    resume_after=state['through'] if state else 0
    if state and resume_after>=end_qid:
        # Recover even if interruption happened between the two final renames.
        for tmp,path,size in zip(temporary,paths,state['offsets']):
            present=tmp if tmp.exists() else path
            if not present.exists() or present.stat().st_size!=size:
                raise ValueError('Completed inference file is missing or truncated')
            if tmp.exists(): tmp.replace(path)
        (output/'inference_metrics.json').write_text(json.dumps(state['counts'],indent=2))
        checkpoint_path.unlink()
        con.close(); query_con.close()
        return state['counts']
    query_cursor = query_con.execute('SELECT qid,entity_id FROM queries WHERE qid>? ORDER BY qid',[resume_after])
    def all_queries():
        while chunk := query_cursor.fetchmany(10000):
            yield from chunk
    queries = iter(all_queries())
    current = next(queries,None)
    counts = dict(entities=0,candidate_pairs=0,matched_pairs=0,entities_with_matches=0,
                  threshold=model.threshold,top_k=top_k,model_sha256=config['model_sha256'])
    if state:
        counts=state['counts']
        for path,size in zip(temporary,state['offsets']):
            if not path.exists() or path.stat().st_size<size:
                raise ValueError('Inference checkpoint data is missing or truncated')
        log(f'Resuming after {counts["entities"]:,} completed test entities')
    def batches():
        # Bound the joins as well as Python feature matrices. A single global
        # multi-join/sort can exceed memory before fetchmany starts returning rows.
        for lo in range(resume_after+1,end_qid+1,10000):
            cursor = con.execute(PAIR_SQL +
                ' WHERE q.qid BETWEEN ? AND ? ORDER BY q.qid,r.entity_id', [lo,lo+9999])
            yield from grouped_batches(cursor)
    log('Joining retrieval pool in bounded query batches for feature extraction')
    try:
        mode='r+b' if state else 'wb'
        with temporary[0].open(mode) as cand, temporary[1].open(mode) as match:
            if state:
                for stream,offset in zip((cand,match),state['offsets']):
                    stream.seek(offset); stream.truncate()
            else:
                cand.write(b'source1_entity_id\tcandidate_entity_ids\n')
                match.write(b'source1_entity_id\tmatched_entity_ids\n')
            last_written=resume_after
            def checkpoint():
                cand.flush(); match.flush()
                progress=dict(config=config,through=last_written,counts=counts,
                              offsets=[cand.tell(),match.tell()])
                pending=checkpoint_path.with_suffix('.json.tmp')
                pending.write_text(json.dumps(progress),encoding='utf-8')
                pending.replace(checkpoint_path)
            def write_row(query,candidates,matches):
                nonlocal last_written
                cand.write((query[1]+'\t'+','.join(candidates)+'\n').encode('utf-8'))
                match.write((query[1]+'\t'+','.join(matches)+'\n').encode('utf-8'))
                last_written=query[0]
                counts['entities'] += 1
                counts['candidate_pairs'] += len(candidates)
                counts['matched_pairs'] += len(matches)
                counts['entities_with_matches'] += bool(matches)
            next_progress = counts['entities']+10000
            next_checkpoint = counts['entities']+1000
            checkpoint()
            for last_qid,candidates,matches in scored_batches(batches(),model_path,top_k,workers):
                while current is not None and current[0] <= last_qid:
                    write_row(current,candidates.get(current[0],[]),matches.get(current[0],[]))
                    current = next(queries,None)
                if counts['entities']>=next_checkpoint:
                    checkpoint()
                    next_checkpoint=counts['entities']+1000
                if counts['entities']>=next_progress:
                    log(f"Scored {counts['entities']:,} test entities; {counts['matched_pairs']:,} matches")
                    next_progress = counts['entities']+10000
            while current is not None:
                write_row(current,[],[])
                current = next(queries,None)
            checkpoint()
        for tmp,path in zip(temporary,paths):
            tmp.replace(path)
    finally:
        con.close()
        query_con.close()
    (output/'inference_metrics.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    checkpoint_path.unlink(missing_ok=True)
    return counts
