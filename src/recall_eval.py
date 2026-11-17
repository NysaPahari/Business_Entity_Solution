"""Union-of-channels candidate evaluation with a cached pair-feature store.

Candidate set per query = existing composite pool (unchanged) UNION each enabled
extra channel's top-k. Pairs are featurized once with the frozen selected
extractor; retrieval_score is the same composite key score used in production
(0.0 when a pair shares no retained key). Cutoffs are fitted on the calibration
tuning half only; audit is reported, never used for selection. Validation
entities with zero candidates stay in every denominator.
"""
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from .data import connect, log, sql_string
from .features import final_candidates, FEATURE_NAMES
from .accuracy_features import EXTRA_NAMES
from .model import MatchingModel
from .refinement import fit_policy, policy_report, score_of

SOURCE_DB = Path('work/accuracy_composite.duckdb')
DATABASE = Path('work/recall/recall.duckdb')
SELECTED = Path('artifacts/accuracy/selected/model.joblib')
RESULTS = Path('artifacts/recall/results.json')
SPACES = {'main': ('queries', 'query_keys', 'pool'), 'extra': ('extra_queries', 'extra_query_keys', 'extra_pool')}
NF = len(FEATURE_NAMES) + len(EXTRA_NAMES)
COLUMNS = ','.join(f'f{i} FLOAT' for i in range(NF))

_extractor = None


def _init_worker(path):
    global _extractor
    _extractor = MatchingModel.load(path).extractor


def _featurize(rows):
    return _extractor.transform(rows)


def open_db(memory='2GB'):
    con = connect(DATABASE, memory, 2)
    con.execute('SET enable_progress_bar=false')
    con.execute(f'ATTACH IF NOT EXISTS {sql_string(SOURCE_DB.resolve().as_posix())} AS src (READ_ONLY)')
    for space in SPACES:
        con.execute(f'CREATE TABLE IF NOT EXISTS feat_{space}(qid BIGINT, rid BIGINT, label UTINYINT, {COLUMNS})')
    return con


def load_truth(ids):
    truth = {}
    with Path('dataset/train/train_ground_truth.tsv').open(encoding='utf-8', newline='') as stream:
        for r in csv.DictReader(stream, delimiter='\t'):
            if r['source1_entity_id'] in ids:
                truth[r['source1_entity_id']] = set(filter(None, r['matched_entity_ids'].split(',')))
    return truth


def ensure_features(con, space, need_table, workers=4, batch=20000):
    """Featurize pairs in `need_table`(qid,rid) that are not yet in the store."""
    queries, keys, pool = SPACES[space]
    con.execute(f'''CREATE OR REPLACE TEMP TABLE missing AS SELECT DISTINCT n.qid,n.rid FROM {need_table} n
        ANTI JOIN feat_{space} f ON f.qid=n.qid AND f.rid=n.rid''')
    total = con.execute('SELECT count(*) FROM missing').fetchone()[0]
    if not total:
        return
    log(f'Featurizing {total:,} new {space} pairs')
    # Composite key score for pairs outside the stored pool, same formula as blocking.
    con.execute(f'''CREATE OR REPLACE TEMP TABLE missing_rows AS
        WITH ks AS (SELECT m.qid,m.rid,sum(1.0/sqrt(rk.df)) score FROM missing m
                    JOIN src.{keys} qk ON qk.qid=m.qid JOIN src.ref_keys rk ON rk.key=qk.key AND rk.rid=m.rid
                    GROUP BY m.qid,m.rid)
        SELECT m.qid,m.rid,q.entity_id sid,s.name sn,s.address sa,s.country sc,
               r.entity_id cid,r.name rn,r.address ra,r.country rc,
               coalesce(p.retrieval_score, ks.score::FLOAT, 0.0::FLOAT) score
        FROM missing m JOIN src.{queries} q ON q.qid=m.qid JOIN src.records s ON s.rid=q.rid
        JOIN src.records r ON r.rid=m.rid
        LEFT JOIN src.{pool} p ON p.qid=m.qid AND p.rid=m.rid
        LEFT JOIN ks ON ks.qid=m.qid AND ks.rid=m.rid
        ORDER BY m.qid,r.entity_id''')
    ids = {r[0] for r in con.execute('SELECT DISTINCT sid FROM missing_rows').fetchall()}
    truth = load_truth(ids)
    done = 0
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(str(SELECTED),)) as pool_exec:
        for lo in range(0, total, batch * workers):
            data = con.execute('SELECT qid,rid,sid,sn,sa,sc,cid,rn,ra,rc,score FROM missing_rows LIMIT ? OFFSET ?',
                               [batch * workers, lo]).fetchall()
            rows = [(r[0], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10]) for r in data]
            chunks = [rows[i:i + batch] for i in range(0, len(rows), batch)]
            x = np.concatenate(list(pool_exec.map(_featurize, chunks)))
            frame = pd.DataFrame(x, columns=[f'f{i}' for i in range(NF)])
            frame.insert(0, 'label', np.array([r[6] in truth[r[2]] for r in data], dtype=np.uint8))
            frame.insert(0, 'rid', np.array([r[1] for r in data], dtype=np.int64))
            frame.insert(0, 'qid', np.array([r[0] for r in data], dtype=np.int64))
            con.register('frame', frame)
            con.execute(f'INSERT INTO feat_{space} SELECT * FROM frame')
            con.unregister('frame')
            done += len(data)
            log(f'  {space}: featurized {done:,}/{total:,}')
    con.execute('DROP TABLE missing_rows')


def channel_sql(space, channels):
    """(qid,rid) candidate union: existing pool plus each enabled channel."""
    _, _, pool = SPACES[space]
    parts = [f'SELECT qid,rid FROM src.{pool}']
    for name, k in channels.items():
        if not k:
            continue
        if name.startswith('ann_'):
            parts.append(f"SELECT qid,rid FROM ann WHERE space='{space}' AND field='{name[4:]}' AND rank<={int(k)}")
        elif name == 'phonetic':
            parts.append(f"SELECT qid,rid FROM phon_pool WHERE space='{space}' AND rank<={int(k)}")
        else:
            raise ValueError(name)
    return ' UNION '.join(parts)


def load_pairs(con, space, channels, cohorts, workers=4):
    queries = SPACES[space][0]
    cohort_filter = '' if cohorts is None else 'WHERE q.cohort IN (' + ','.join(map(sql_string, cohorts)) + ')'
    con.execute(f'''CREATE OR REPLACE TEMP TABLE need AS SELECT c.qid,c.rid FROM ({channel_sql(space, channels)}) c
        JOIN (SELECT q.qid FROM src.{queries} q {cohort_filter}) q USING(qid)''')
    ensure_features(con, space, 'need', workers)
    fcols = ','.join(f'f.f{i}' for i in range(NF))
    data = con.execute(f'''SELECT n.qid,q.entity_id,r.entity_id,f.label,{fcols} FROM need n
        JOIN feat_{space} f ON f.qid=n.qid AND f.rid=n.rid JOIN src.{queries} q ON q.qid=n.qid
        JOIN src.records r ON r.rid=n.rid ORDER BY n.qid,r.entity_id''').fetchall()
    rows = [(d[0], d[1], None, None, None, d[2]) for d in data]
    x = np.array([d[4:] for d in data], dtype=np.float32).reshape(-1, NF)
    y = np.array([d[3] for d in data], dtype=np.uint8)
    return rows, x, y


def validation_data(con, channels, workers=4, cap=None):
    rows, x, _ = load_pairs(con, 'main', channels, ('calibration', 'audit'), workers)
    queries = {r[0]: dict(cohort=r[1], country=r[2]) for r in con.execute(
        '''SELECT q.entity_id,q.cohort,r.country FROM src.queries q JOIN src.records r USING(rid)
           WHERE q.cohort IN ('calibration','audit')''').fetchall()}
    if cap:
        keep = final_candidates(rows, x, cap)
        rows, x = [rows[i] for i in keep], x[keep]
    return dict(rows=rows, x=x, queries=queries, truth=load_truth(set(queries)), threshold=0.5)


def evaluate(model, data):
    indices = np.arange(len(data['rows']))
    raw = model.probabilities(data['x'])
    policy = fit_policy(data, indices, raw)
    report = policy_report(data, indices, raw, policy)
    audit = report['audit']
    errors = next(s for s in audit['segments'] if s['dimension'] == 'all')
    return dict(selection_macro_f05=report['selection_macro_f05'], audit_macro_f05=score_of(report),
                calibration_macro_f05=score_of(report, 'calibration'), threshold=policy.thresholds['global'],
                audit_blocking_recall=audit['blocking']['pair_recall'],
                calibration_blocking_recall=report['calibration']['blocking']['pair_recall'],
                audit_oracle_f05=audit['blocking']['oracle_macro_f05'],
                audit_mean_candidates=audit['blocking']['mean_candidates'],
                calibration_mean_candidates=report['calibration']['blocking']['mean_candidates'],
                audit_fp=errors['fp'], audit_fn=errors['fn'])


def retrain(con, channels, workers=4, seed=2026):
    """Selected recipe (600 trees, 63 leaves, 42k training entities), new candidates."""
    xs, ys = [], []
    for space, cohorts in (('main', ('train',)), ('extra', None)):
        _, x, y = load_pairs(con, space, channels, cohorts, workers)
        xs.append(x); ys.append(y)
    x, y = np.concatenate(xs), np.concatenate(ys)
    del xs, ys
    base = MatchingModel.load(SELECTED)
    model = MatchingModel(base.extractor, seed, 4)
    model.classifier.set_params(n_estimators=600, num_leaves=63, min_child_samples=60, reg_lambda=5.)
    log(f'Retraining on {len(y):,} pairs ({int(y.sum()):,} positive)')
    model.classifier.fit(x, y, feature_name=FEATURE_NAMES + EXTRA_NAMES)
    return model, len(y)


def record(name, result):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results[name] = result
    RESULTS.write_text(json.dumps(results, indent=2), encoding='utf-8')
    log(f'{name}: ' + json.dumps(result))


def run(name, channels, retrained=False, cap=None, workers=4):
    con = open_db()
    try:
        results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
        frozen_name = name + ('' if cap is None else f'_cap{cap}') + '_frozen'
        if frozen_name not in results:
            data = validation_data(con, channels, workers, cap)
            result = evaluate(MatchingModel.load(SELECTED), data)
            result.update(channels=channels, cap=cap, model='frozen selected')
            record(frozen_name, result)
        retrained_name = name + '_retrained'
        if retrained and cap is None and retrained_name not in results:
            model, pairs = retrain(con, channels, workers)
            data = validation_data(con, channels, workers)
            result = evaluate(model, data)
            result.update(channels=channels, cap=None, model='retrained selected recipe', training_pairs=pairs)
            model.threshold = result['threshold']
            model.save(Path('artifacts/recall') / f'{retrained_name}.joblib')
            record(retrained_name, result)
    finally:
        con.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    parser.add_argument('--channels', default='{}', help='JSON, e.g. {"ann_name":16,"ann_address":16}')
    parser.add_argument('--retrained', action='store_true')
    parser.add_argument('--cap', type=int)
    parser.add_argument('--workers', type=int, default=4)
    a = parser.parse_args()
    run(a.name, json.loads(a.channels), a.retrained, a.cap, a.workers)
