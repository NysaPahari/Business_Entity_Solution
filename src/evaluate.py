"""Exact per-entity macro F0.5, candidate recall, and threshold selection."""
import numpy as np


def entity_f05(truth, prediction):
    truth, prediction = set(truth), set(prediction)
    if not truth:
        return float(not prediction)
    # Algebraically equivalent to 1.25 PR / (.25 P + R).
    return 1.25 * len(truth & prediction) / (0.25 * len(truth) + len(prediction))


def macro_f05(truth, predictions):
    if not truth:
        raise ValueError('Cannot evaluate an empty entity set')
    if set(predictions) - set(truth):
        raise ValueError('Predictions include unknown Source 1 entities')
    return sum(entity_f05(t, predictions.get(sid,())) for sid,t in truth.items())/len(truth)


def blocking_metrics(truth, candidates):
    actual = sum(len(t) for t in truth.values())
    recovered = sum(len(set(t) & set(candidates.get(sid,()))) for sid,t in truth.items())
    ceiling = macro_f05(truth, {sid:set(t) & set(candidates.get(sid,())) for sid,t in truth.items()})
    return dict(true_pairs=actual, recovered_pairs=recovered,
                pair_recall=recovered/actual if actual else 1.0,
                oracle_macro_f05=ceiling,
                mean_candidates=sum(len(candidates.get(sid,())) for sid in truth)/len(truth))


def predictions_at(rows, scores, threshold):
    result = {}
    for row, score in zip(rows,scores):
        if score >= threshold:
            result.setdefault(row[1],set()).add(row[5])
    return result


def select_threshold(truth, rows, scores):
    """Exact sweep over distinct probabilities; prefer higher thresholds on ties.

    Incrementally update only the affected entity's contribution, so cost is
    O(number of pairs log number of pairs), not entities x thresholds.
    """
    if not truth:
        raise ValueError('Calibration truth cannot be empty')
    if len(rows) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('Invalid score array')
    counts, true_positives = {}, {}
    total = sum(not t for t in truth.values())
    best = total/len(truth)
    threshold = float(np.nextafter(1.0,2.0))
    order = np.argsort(-np.asarray(scores),kind='stable')
    i = 0
    while i < len(order):
        cutoff = float(scores[order[i]])
        j = i
        while j < len(order) and float(scores[order[j]]) == cutoff:
            row = rows[order[j]]
            sid,cid = row[1],row[5]
            if sid not in truth:
                raise ValueError('Calibration contains an unknown entity')
            n = counts.get(sid,0)
            tp = true_positives.get(sid,0)
            size = len(truth[sid])
            old = (1.25*tp/(0.25*size+n)) if size else float(n==0)
            n += 1
            tp += cid in truth[sid]
            new = (1.25*tp/(0.25*size+n)) if size else 0.0
            total += new-old
            counts[sid],true_positives[sid] = n,tp
            j += 1
        score = total/len(truth)
        if score > best + 1e-12:
            best,threshold = score,cutoff
        i = j
    return threshold,best
