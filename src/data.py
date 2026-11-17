"""Bounded-memory ingestion and deterministic entity-level splits."""
from pathlib import Path
import json
import time
import duckdb


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def connect(path, memory='4GB', threads=6, read_only=False):
    connection = duckdb.connect(str(path), read_only=read_only)
    connection.execute(f"SET memory_limit={sql_string(memory)}")
    connection.execute(f"SET threads={int(threads)}")
    connection.execute("SET preserve_insertion_order=false")
    return connection


def prepare(dataset_dir, split, work_dir, train_size=12000, tune_size=3000,
            audit_size=3000, seed=2026, memory='4GB', threads=6):
    """Read all reference records; sample only labeled S1 queries for model fitting.

    Hash buckets 0/1 are independent calibration/audit sets, never fitted. All
    remaining buckets form the training population. Test queries are never sampled.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    path = work / f'{split}.duckdb'
    manifest = work / f'{split}_ingest.json'
    paths = [Path(dataset_dir) / split / f'{split}_source{i}_normalized.tsv'
             for i in (1, 2, 3)]
    config = dict(split=split, train_size=train_size, tune_size=tune_size,
                  audit_size=audit_size, seed=seed,
                  sources=[dict(path=str(p.resolve()), size=p.stat().st_size,
                                mtime_ns=p.stat().st_mtime_ns) for p in paths])
    if path.exists() and manifest.exists():
        if json.loads(manifest.read_text()) != config:
            raise ValueError('Ingestion configuration changed; use a new --work-dir')
        return path
    con = connect(path, memory, threads)
    try:
        selections = []
        for i, source in enumerate(paths, 1):
            selections.append(f"""SELECT entity_id, business_name_normalized AS name,
                business_address_normalized AS address, country, {i} AS source
                FROM read_csv({sql_string(source.as_posix())}, delim='\t', header=true,
                     all_varchar=true, nullstr='__NO_NULL_SENTINEL__', parallel=true)""")
        log(f'Importing normalized {split} records')
        con.execute('CREATE OR REPLACE TABLE records AS SELECT row_number() OVER ()::BIGINT rid, * FROM ('
                    + ' UNION ALL '.join(selections) + ')')
        if split == 'train':
            portions = []
            for group, condition, size in [('train', '>=2', train_size),
                                           ('calibration', '=0', tune_size),
                                           ('audit', '=1', audit_size)]:
                portions.append(f"""(SELECT rid, entity_id, {sql_string(group)} AS cohort
                    FROM records WHERE source=1 AND hash(entity_id || '{seed}') % 10 {condition}
                    ORDER BY hash(entity_id || 'sample{seed}'), entity_id LIMIT {int(size)})""")
            query = ' UNION ALL '.join(portions)
        else:
            query = "SELECT rid, entity_id, 'test' AS cohort FROM records WHERE source=1"
        con.execute('CREATE OR REPLACE TABLE queries AS SELECT row_number() OVER (ORDER BY rid)::BIGINT qid, * FROM (' + query + ')')
        counts = con.execute('SELECT source, count(*) FROM records GROUP BY source ORDER BY source').fetchall()
        log(f'{split} source counts: {counts}; selected queries: ' + str(con.execute('SELECT cohort,count(*) FROM queries GROUP BY cohort').fetchall()))
        con.execute('CHECKPOINT')
    finally:
        con.close()
    manifest.write_text(json.dumps(config, indent=2), encoding='utf-8')
    return path
