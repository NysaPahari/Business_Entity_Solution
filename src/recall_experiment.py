"""Retrieval-recall experiments layered on the selected composite configuration.

Stages (each resumable/idempotent, all writes under work/recall and artifacts/recall):
  ann-index   Encode every S2/S3 reference record per field; build IVF-PQ FAISS index.
  ann-search  Search model/extra-training queries; exact-cosine rerank of ANN hits.
The existing composite experiment database is only read. The one-time fresh-audit
queries are never encoded or searched here.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import connect, log
from .embed_retrieval import Model2VecEncoder, name_text, address_text, normalize_rows

SOURCE_DB = 'work/accuracy_composite.duckdb'
WORK = Path('work/recall')
OUT = Path('artifacts/recall')
MODEL_DIR = OUT / 'potion-base-8M'
FIELDS = {'name': (name_text, 'name'), 'address': (address_text, 'address')}


def encoder():
    if not MODEL_DIR.exists():
        from model2vec import StaticModel
        StaticModel.from_pretrained('minishlab/potion-base-8M').save_pretrained(str(MODEL_DIR))
    return Model2VecEncoder(str(MODEL_DIR))


def ann_index(field, nlist=4096, m=32, chunk=500000, threads=6):
    import faiss
    faiss.omp_set_num_threads(threads)
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f'ann_{field}.faiss'
    if path.exists():
        return
    fn, column = FIELDS[field]
    enc = encoder()
    con = connect(SOURCE_DB, '1GB', 1, read_only=True)
    sample = con.execute(f"SELECT {column} FROM records WHERE source<>1 AND hash(entity_id||'ann-train')%100<3").fetchall()
    log(f'{field}: training IVF{nlist},PQ{m} on {len(sample):,} reference vectors')
    train = enc.encode([fn(r[0]) for r in sample])
    quantizer = faiss.IndexFlatIP(train.shape[1])
    index = faiss.IndexIVFPQ(quantizer, train.shape[1], nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
    index.train(train)
    del train, sample
    end = con.execute('SELECT max(rid) FROM records').fetchone()[0]
    for lo in range(1, end + 1, chunk):
        batch = con.execute(f'SELECT rid,{column} FROM records WHERE source<>1 AND rid BETWEEN ? AND ?',
                            [lo, lo + chunk - 1]).fetchall()
        if batch:
            vectors = enc.encode([fn(r[1]) for r in batch])
            index.add_with_ids(vectors, np.array([r[0] for r in batch], dtype=np.int64))
        log(f'{field}: indexed through rid {min(lo + chunk - 1, end):,} ({index.ntotal:,} vectors)')
    con.close()
    faiss.write_index(index, str(path) + '.tmp')
    Path(str(path) + '.tmp').replace(path)


def query_sets(con):
    # 'main' = train/calibration/audit cohorts; 'extra' = additional training entities.
    return {'main': con.execute('SELECT q.qid,r.name,r.address FROM queries q JOIN records r USING(rid) ORDER BY q.qid').fetchall(),
            'extra': con.execute('SELECT q.qid,r.name,r.address FROM extra_queries q JOIN records r USING(rid) ORDER BY q.qid').fetchall()}


def ann_search(k=32, probe_k=64, nprobe=32, threads=6):
    """Approximate top-probe_k per field, exact-cosine reranked, top-k persisted."""
    import faiss
    faiss.omp_set_num_threads(threads)
    database = WORK / 'recall.duckdb'
    out = connect(database, '1GB', 1)
    out.execute('CREATE TABLE IF NOT EXISTS ann(space VARCHAR,qid BIGINT,rid BIGINT,field VARCHAR,rank INTEGER,sim FLOAT)')
    done = {(r[0], r[1]) for r in out.execute('SELECT DISTINCT space,field FROM ann').fetchall()}
    src = connect(SOURCE_DB, '1GB', 1, read_only=True)
    sets = query_sets(src)
    enc = encoder()
    for field, (fn, column) in FIELDS.items():
        index = None
        for space, rows in sets.items():
            if (space, field) in done:
                continue
            if index is None:
                index = faiss.read_index(str(WORK / f'ann_{field}.faiss'))
                index.nprobe = nprobe
            col = 1 if column == 'name' else 2
            qv = enc.encode([fn(r[col]) for r in rows])
            _, ids = index.search(qv, probe_k)
            # Exact cosine rerank of approximate hits (PQ distances are coarse).
            flat = np.unique(ids[ids >= 0])
            src.execute('CREATE OR REPLACE TEMP TABLE hit(rid BIGINT)')
            src.execute('INSERT INTO hit SELECT * FROM (SELECT unnest(?::BIGINT[]))', [flat.tolist()])
            texts = dict(src.execute(f'SELECT r.rid,r.{column} FROM records r JOIN hit USING(rid)').fetchall())
            order = np.array(sorted(texts), dtype=np.int64)
            rv = enc.encode([fn(texts[r]) for r in order])
            position = {r: i for i, r in enumerate(order.tolist())}
            records = []
            for (qid, *_), qvec, hits in zip(rows, qv, ids):
                hits = [h for h in hits.tolist() if h >= 0]
                if not hits or not np.any(qvec):
                    continue  # empty field: no dense evidence
                sims = rv[[position[h] for h in hits]] @ qvec
                best = np.argsort(-sims, kind='stable')[:k]
                records.extend((space, qid, hits[j], field, rank, float(sims[j])) for rank, j in enumerate(best, 1))
            out.execute('BEGIN')
            out.executemany('INSERT INTO ann VALUES (?,?,?,?,?,?)', records)
            out.execute('COMMIT')
            log(f'ANN {field}/{space}: {len(rows):,} queries, {len(records):,} neighbors')
    src.close(); out.execute('CHECKPOINT'); out.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['ann-index', 'ann-search'])
    parser.add_argument('--field', choices=list(FIELDS))
    args = parser.parse_args()
    if args.stage == 'ann-index':
        for field in ([args.field] if args.field else FIELDS):
            ann_index(field)
    else:
        ann_search()
