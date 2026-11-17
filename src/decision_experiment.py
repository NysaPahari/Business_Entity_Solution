"""Bounded-memory calibration-only trial of within-query relative confidence.

Does not modify the selected model or inspect either audit cohort.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .blocking import PAIR_SQL
from .data import connect, log
from .evaluate import entity_f05
from .finalize_improved import digest
from .model import MatchingModel
from .refinement import calibration_role, threshold_with_overrides


def relative_scores(rows, raw, weight):
    best = defaultdict(float)
    for row, score in zip(rows, raw):
        best[row[1]] = max(best[row[1]], float(score))
    return np.asarray([p - weight * best[r[1]] for r, p in zip(rows, raw)])


def run():
    out = Path('artifacts/accuracy/decision_experiment')
    out.mkdir(exist_ok=True)
    model_path = Path('artifacts/accuracy/selected/model.joblib')
    fingerprint = digest(model_path)
    model = MatchingModel.load(model_path)
    con = connect('work/accuracy_composite.duckdb', '512MB', 1, read_only=True)
    try:
        queries = con.execute("SELECT qid,entity_id FROM queries WHERE cohort='calibration' ORDER BY qid").fetchall()
        ids = {sid for _, sid in queries}
        truth = {}
        with Path('dataset/train/train_ground_truth.tsv').open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                if row['source1_entity_id'] in ids:
                    truth[row['source1_entity_id']] = set(filter(None, row['matched_entity_ids'].split(',')))
        if set(truth) != ids:
            raise ValueError('Incomplete calibration truth')
        rows, scores = [], []
        for start in range(0, len(queries), 100):
            group = queries[start:start + 100]
            batch = con.execute(PAIR_SQL + " WHERE q.cohort='calibration' AND q.qid BETWEEN ? AND ? ORDER BY q.qid,r.entity_id",
                                [group[0][0], group[-1][0]]).fetchall()
            scores.extend(model.probabilities(model.extractor.transform(batch)))
            rows.extend((r[0], r[1], '', '', '', r[5]) for r in batch)
            log(f'Calibration scored: {min(start + 100, len(queries))}/{len(queries)}')
    finally:
        con.close()
    raw = np.asarray(scores)
    reports = []
    # Three prespecified alternatives, not an exhaustive threshold/model search.
    for weight in (0., .1, .25, .5):
        adjusted = relative_scores(rows, raw, weight)
        fit_truth = {s:t for s,t in truth.items() if calibration_role(s) == 'isotonic_fit'}
        mask = np.array([r[1] in fit_truth for r in rows])
        fit_rows = [r for r,m in zip(rows,mask) if m]
        zeros = np.zeros(int(mask.sum()), dtype=bool)
        cutoff, fit_score = threshold_with_overrides(fit_truth, fit_rows, adjusted[mask], zeros, zeros)
        predictions = defaultdict(set)
        baseline = defaultdict(set)
        for row, p, original in zip(rows, adjusted, raw):
            if p >= cutoff: predictions[row[1]].add(row[5])
            if original >= model.threshold: baseline[row[1]].add(row[5])
        tune_ids = sorted(ids - set(fit_truth))
        delta = np.array([entity_f05(truth[s],predictions[s])-entity_f05(truth[s],baseline[s]) for s in tune_ids])
        score = np.mean([entity_f05(truth[s],predictions[s]) for s in tune_ids])
        reports.append(dict(weight=weight, cutoff=cutoff, fit_macro_f05=fit_score,
                            tune_macro_f05=float(score), delta_vs_selected=float(delta.mean()),
                            improved_entities=int((delta>0).sum()), worsened_entities=int((delta<0).sum())))
    result = dict(model_sha256=fingerprint, calibration_entities=len(ids), candidate_pairs=len(rows),
                  protocol='Cutoffs fit on calibration fit half; alternatives compared on calibration tune half. No audit access. No production promotion.',
                  experiments=reports)
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    log(json.dumps(result))


if __name__ == '__main__':
    run()
