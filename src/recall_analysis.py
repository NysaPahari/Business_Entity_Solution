"""Categorize true matches that the selected composite retrieval never returned.

Read-only over the experiment database. Train/calibration breakdowns drive design
choices; the audit breakdown is reported as the requested diagnostic only.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from rapidfuzz import fuzz
from .data import connect, log

DATABASE = 'work/accuracy_composite.duckdb'
OUT = Path('artifacts/recall')
SIMILAR = 0.70  # prespecified; token-set ratio is generous, so below this is genuinely weak


def similarity(a, b):
    if not a.strip() or not b.strip():
        return None
    return max(fuzz.token_set_ratio(a, b), fuzz.ratio(a.replace(' ', ''), b.replace(' ', ''))) / 100


def category(name_sim, address_sim):
    name_ok = name_sim is not None and name_sim >= SIMILAR
    address_ok = address_sim is not None and address_sim >= SIMILAR
    if name_ok and address_ok:
        return 'both_similar_but_unretrieved'
    if address_ok:
        return 'name_similarity_too_low'
    if name_ok:
        return 'address_similarity_too_low'
    return 'low_signal_both_fields'


def load_truth(ids):
    truth = {}
    with Path('dataset/train/train_ground_truth.tsv').open(encoding='utf-8', newline='') as stream:
        for r in csv.DictReader(stream, delimiter='\t'):
            if r['source1_entity_id'] in ids:
                truth[r['source1_entity_id']] = set(filter(None, r['matched_entity_ids'].split(',')))
    return truth


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    con = connect(DATABASE, '2GB', 1, read_only=True)
    queries = con.execute('''SELECT q.qid,q.entity_id,q.cohort,r.name,r.address,r.country
        FROM queries q JOIN records r USING(rid)''').fetchall()
    info = {q[1]: q for q in queries}
    truth = load_truth(set(info))
    pool = defaultdict(set)
    for sid, cid in con.execute('SELECT q.entity_id,r.entity_id FROM pool p JOIN queries q USING(qid) JOIN records r ON r.rid=p.rid').fetchall():
        pool[sid].add(cid)
    missed = [(sid, cid) for sid, t in truth.items() for cid in t - pool[sid]]
    con.execute('CREATE TEMP TABLE missed(sid VARCHAR, cid VARCHAR)')
    con.executemany('INSERT INTO missed VALUES (?,?)', missed)
    # Shared retained keys and the retrieval score the missed pair would have had.
    details = {(r[0], r[1]): r[2:] for r in con.execute('''
        SELECT m.sid,m.cid,r.name,r.address,r.country,
          (SELECT count(*) FROM query_keys qk JOIN ref_keys rk USING(key) WHERE qk.qid=q.qid AND rk.rid=r.rid),
          (SELECT coalesce(sum(1.0/sqrt(rk.df)),0) FROM query_keys qk JOIN ref_keys rk USING(key) WHERE qk.qid=q.qid AND rk.rid=r.rid),
          (SELECT min(p.retrieval_score) FROM pool p WHERE p.qid=q.qid),
          (SELECT count(*) FROM pool p WHERE p.qid=q.qid)
        FROM missed m JOIN queries q ON q.entity_id=m.sid JOIN records r ON r.entity_id=m.cid''').fetchall()}
    con.close()
    rows = []
    for sid, cid in missed:
        q = info[sid]
        name, address, country, shared, score, floor, pool_count = details[sid, cid]
        ns, as_ = similarity(q[3], name), similarity(q[4], address)
        rows.append(dict(cohort=q[2], source1_entity_id=sid, candidate_entity_id=cid,
            category=category(ns, as_), name_similarity=ns, address_similarity=as_,
            shared_retained_keys=shared, missed_retrieval_score=round(score, 4),
            pool_min_score=None if floor is None else round(floor, 4), pool_count=pool_count,
            retrieval_cause='no_shared_key' if shared == 0 else 'ranked_below_pool_cutoff',
            same_country=q[5] == country, s1_name=q[3], s1_address=q[4], s1_country=q[5],
            match_name=name, match_address=address, match_country=country))
    report = {}
    for cohort in ('train', 'calibration', 'audit'):
        part = [r for r in rows if r['cohort'] == cohort]
        ids = [s for s in truth if info[s][2] == cohort]
        pairs = sum(len(truth[s]) for s in ids)
        entities = {r['source1_entity_id'] for r in part}
        report[cohort] = dict(
            entities=len(ids), true_pairs=pairs, missed_pairs=len(part),
            pair_recall=1 - len(part) / pairs, entities_with_a_missed_match=len(entities),
            entities_with_no_true_match_retrieved=sum(1 for s in ids if truth[s] and not truth[s] & pool[s]),
            by_category=dict(Counter(r['category'] for r in part).most_common()),
            by_retrieval_cause=dict(Counter(r['retrieval_cause'] for r in part).most_common()),
            category_x_cause=dict(Counter(f"{r['category']}|{r['retrieval_cause']}" for r in part).most_common()),
            by_country_pair=dict(Counter(f"{r['s1_country']}->{r['match_country']}" for r in part).most_common()),
            empty_field_pairs=sum(1 for r in part if r['name_similarity'] is None or r['address_similarity'] is None))
        log(f'{cohort}: ' + json.dumps(report[cohort]))
    (OUT / 'false_negative_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (OUT / 'false_negatives.tsv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    return report


if __name__ == '__main__':
    run()
