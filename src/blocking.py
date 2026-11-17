"""Disk-backed token/character candidate retrieval with soft country handling.

Character retrieval is approximate: bounded min-hash-selected four-gram postings
are unioned with name/address postings, then true 2-4-gram TF-IDF reranks the pool.
No full all-pairs matrix is constructed. High-frequency keys are omitted globally,
not by country. The final reranked top K is exactly what the classifier scores.
"""
from pathlib import Path
import json
from .data import connect, log

NAME_STOP = ('and', 'the', 'corporation', 'private', 'limited', 'incorporated',
             'company', 'co', 'llc', 'llp', 'of')
ADDRESS_STOP = ('road', 'street', 'apartment', 'near', 'the', 'and', 'no', 'number')


def key_sql(strategy='baseline', rid_bounds=None):
    ns = ','.join(repr(t) for t in NAME_STOP)
    ads = ','.join(repr(t) for t in ADDRESS_STOP)
    scope='' if rid_bounds is None else f' WHERE rid BETWEEN {int(rid_bounds[0])} AND {int(rid_bounds[1])}'
    extra = ''
    if strategy == 'composite':
        # Country-independent intersections survive when each token alone is too common.
        extra = """,
        flatten(list_transform(list_slice(nt,1,4), t->list_transform(list_slice(addr_tokens,1,4),a->hash('ta:' || t || ':' || a)))),
        flatten(list_transform(list_slice(addr_tokens,1,4), a->list_transform(list_filter(list_slice(addr_tokens,1,4),b->b>a),b->hash('aa:' || a || ':' || b)))),
        list_transform(nt,t->hash('sk:' || substring(t,1,1) || regexp_replace(substring(t,2),'[aeiou]','','g'))),
        flatten(list_transform(list_slice(nt,1,3),t->list_transform(nums,n->hash('sn:' || substring(t,1,1) || regexp_replace(substring(t,2),'[aeiou]','','g') || ':' || ltrim(n,'0')))))"""
    elif strategy != 'baseline':
        raise ValueError(f'Unknown blocking strategy: {strategy}')
    # Deterministic smallest hashed ngrams are shared despite moderate local edits.
    return f"""
    WITH tokens AS (
      SELECT rid, source,
        list_sort(list_distinct(list_filter(string_split(name,' '),
          x -> length(x)>=2 AND x NOT IN ({ns})))) nt,
        list_sort(list_distinct(list_filter(string_split(address,' '),
          x -> length(x)>=3 AND x NOT IN ({ads}) AND NOT regexp_full_match(x,'[0-9]+')))) addr_tokens,
        list_slice(list_distinct(regexp_extract_all(address,'[0-9]+')),1,3) nums,
        name, address
      FROM records{scope}
    ), keys AS (
      SELECT rid, source, list_distinct(list_concat(
        CASE WHEN len(nt)>0 THEN [hash('n:' || array_to_string(nt,' '))] ELSE [] END,
        CASE WHEN address<>'' THEN [hash('e:' || address)] ELSE [] END,
        list_transform(nt,x->hash('t:' || x)),
        list_transform(addr_tokens,x->hash('a:' || x)),
        list_slice(list_sort(list_distinct(list_transform(range(1,greatest(length(name)-2,1)),
           i->hash('g:' || substring(name,i,4))))),1,12),
        flatten(list_transform(list_slice(nt,1,3), t->list_transform(nums,n->hash('h:' || t || ':' || n)))) {extra}
      )) kk FROM tokens
    ) SELECT rid, source, unnest(kk) AS key FROM keys
    """


def build_index(database, max_df=128, memory='4GB', threads=6, strategy='baseline', chunk_size=250000):
    marker = Path(str(database) + '.index.json')
    config = dict(max_df=max_df, schema=1)
    if strategy != 'baseline':
        config['strategy'] = strategy
    if marker.exists():
        if json.loads(marker.read_text()) != config:
            raise ValueError('Blocking parameters changed; use a new --work-dir')
        return
    con = connect(database, memory, threads)
    try:
        log('Building name, address, numeric/name, and character postings')
        raw_marker=Path(str(database)+'.raw_index.json')
        if raw_marker.exists() and json.loads(raw_marker.read_text())!=config:
            raise ValueError('Partial index parameters changed; use a new work directory')
        present={r[0] for r in con.execute('SHOW TABLES').fetchall()}
        if not raw_marker.exists() or not {'raw_keys','raw_key_progress'}<=present:
            con.execute('CREATE OR REPLACE TABLE raw_keys(rid BIGINT,source INTEGER,key UBIGINT)')
            con.execute('CREATE OR REPLACE TABLE raw_key_progress(through BIGINT)')
            raw_marker.write_text(json.dumps(config),encoding='utf-8')
        start=con.execute('SELECT coalesce(max(through),0)+1 FROM raw_key_progress').fetchone()[0]
        end=con.execute('SELECT coalesce(max(rid),0) FROM records').fetchone()[0]
        # Bound expensive list expressions and persist completed record ranges.
        # Keep S1 keys only for selected queries; all S2/S3 remain searchable.
        for lo in range(start,end+1,chunk_size):
            hi=min(lo+chunk_size-1,end)
            con.execute('BEGIN')
            con.execute('INSERT INTO raw_keys SELECT k.* FROM ('+key_sql(strategy,(lo,hi))
                        +') k LEFT JOIN queries q ON k.rid=q.rid WHERE k.source<>1 OR q.rid IS NOT NULL')
            con.execute('INSERT INTO raw_key_progress VALUES (?)',[hi])
            con.execute('COMMIT')
            log(f'Indexed record keys through {hi:,} / {end:,}')
        log('Counting posting frequencies and removing uninformative common keys')
        con.execute(f'''CREATE OR REPLACE TABLE key_stats AS
            SELECT key, count(*)::INTEGER df FROM raw_keys WHERE source<>1
            GROUP BY key HAVING count(*) BETWEEN 1 AND {int(max_df)}''')
        log('Posting frequencies complete; sorting reference postings')
        con.execute('''CREATE OR REPLACE TABLE ref_keys AS
            SELECT r.key,r.rid,s.df FROM raw_keys r JOIN key_stats s USING(key)
            WHERE r.source<>1 ORDER BY r.key''')
        log('Reference postings complete; building query postings')
        con.execute('''CREATE OR REPLACE TABLE query_keys AS
            SELECT q.qid,r.key FROM raw_keys r JOIN queries q ON r.rid=q.rid
            JOIN key_stats s USING(key) WHERE r.source=1 ORDER BY q.qid''')
        con.execute('DROP TABLE raw_keys')
        con.execute('DROP TABLE raw_key_progress')
        con.execute('DROP TABLE key_stats')
        con.execute('CHECKPOINT')
        log('Blocking index complete')
    finally:
        con.close()
    marker.write_text(json.dumps(config), encoding='utf-8')
    raw_marker.unlink(missing_ok=True)


def generate_pool(database, pool_size=32, batch_size=50000, memory='4GB', threads=6):
    """Persist a bounded retrieval pool, with resumable query-range batches."""
    con = connect(database, memory, threads)
    try:
        con.execute('CREATE TABLE IF NOT EXISTS pool(qid BIGINT, rid BIGINT, retrieval_score FLOAT)')
        con.execute('CREATE TABLE IF NOT EXISTS pool_progress(start_qid BIGINT, end_qid BIGINT, pool_size INTEGER)')
        previous = con.execute('SELECT DISTINCT pool_size FROM pool_progress').fetchall()
        if previous and previous != [(pool_size,)]:
            raise ValueError('Pool size changed; use a new --work-dir')
        end = con.execute('SELECT coalesce(max(qid),0) FROM queries').fetchone()[0]
        start = con.execute('SELECT coalesce(max(end_qid),0)+1 FROM pool_progress').fetchone()[0]
        for lo in range(start, end + 1, batch_size):
            hi = min(lo + batch_size - 1, end)
            log(f'Retrieving candidates for queries {lo:,}-{hi:,} / {end:,}')
            con.execute('BEGIN')
            con.execute(f'''INSERT INTO pool
                SELECT q.qid, r.rid, sum(1.0 / sqrt(r.df)) AS score
                FROM (SELECT * FROM query_keys WHERE qid BETWEEN {lo} AND {hi}) q
                JOIN ref_keys r USING(key) GROUP BY q.qid,r.rid
                QUALIFY row_number() OVER(PARTITION BY q.qid ORDER BY score DESC,r.rid) <= {int(pool_size)}''')
            con.execute('INSERT INTO pool_progress VALUES (?,?,?)', [lo, hi, pool_size])
            con.execute('COMMIT')
        con.execute('CHECKPOINT')
    finally:
        con.close()


PAIR_SQL = '''SELECT q.qid,q.entity_id,s.name,s.address,s.country,
                    r.entity_id,r.name,r.address,r.country,p.retrieval_score
             FROM pool p JOIN queries q USING(qid)
             JOIN records s ON s.rid=q.rid JOIN records r ON r.rid=p.rid'''


def candidate_rows(connection, lo, hi):
    return connection.execute(PAIR_SQL + ' WHERE q.qid BETWEEN ? AND ? ORDER BY q.qid,r.entity_id', [lo, hi]).fetchall()
