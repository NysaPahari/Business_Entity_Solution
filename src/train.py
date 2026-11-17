"""Train on candidate negatives; calibrate and independently audit by S1 entity."""
import csv
import json
from pathlib import Path
import numpy as np
from .blocking import candidate_rows
from .data import connect,log
from .evaluate import blocking_metrics,macro_f05,predictions_at,select_threshold
from .features import FeatureExtractor,final_candidates
from .model import MatchingModel


def train(database, dataset_dir, artifact_dir, top_k=20, seed=2026, threads=6, memory='4GB'):
    output = Path(artifact_dir)
    output.mkdir(parents=True,exist_ok=True)
    con = connect(database,memory,threads)
    queries = con.execute('SELECT qid,entity_id,cohort FROM queries ORDER BY qid').fetchall()
    groups = {sid:group for _,sid,group in queries}
    truth = {}
    with (Path(dataset_dir)/'train/train_ground_truth.tsv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            sid = row['source1_entity_id']
            if sid in groups:
                if sid in truth:
                    raise ValueError(f'Duplicate ground truth: {sid}')
                truth[sid] = set(filter(None,row['matched_entity_ids'].split(',')))
    if set(truth)!=set(groups):
        raise ValueError('Selected queries have missing ground truth')
    log('Fitting character TF-IDF on training-cohort text only')
    docs = con.execute('''SELECT DISTINCT r.name,r.address FROM records r
        WHERE r.rid IN (SELECT p.rid FROM pool p JOIN queries q USING(qid) WHERE cohort='train'
        UNION SELECT rid FROM queries WHERE cohort='train')
        ORDER BY hash(r.name || r.address) LIMIT 75000''').fetchall()
    extractor = FeatureExtractor().fit([x[0] for x in docs],[x[1] for x in docs])
    rows,matrices = [],[]
    end = max(q[0] for q in queries)
    for lo in range(1,end+1,1000):
        batch = candidate_rows(con,lo,lo+999)
        features = extractor.transform(batch)
        chosen = final_candidates(batch,features,top_k)
        rows.extend(batch[i] for i in chosen)
        matrices.append(features[chosen])
    con.close()
    x = np.concatenate(matrices)
    cohorts = np.array([groups[row[1]] for row in rows])
    labels = np.array([r[5] in truth[r[1]] for r in rows],dtype=np.uint8)
    model = MatchingModel(extractor,seed,threads)
    mask = cohorts=='train'
    log(f'Training on {mask.sum():,} candidate pairs ({labels[mask].sum():,} positives)')
    model.fit(x[mask],labels[mask])
    report = {'seed':seed,'top_k':top_k,'feature_importance_gain':model.feature_importance(),
              'train_positive_pairs':int(labels[mask].sum()),'train_pairs':int(mask.sum()),
              'split_protocol':'hash buckets: 80% train, 10% calibration, 10% untouched audit; deterministic capped samples'}
    scores = model.probabilities(x)
    for group in ('calibration','audit','train'):
        mask = cohorts==group
        subset_rows = [r for r,m in zip(rows,mask) if m]
        subset_truth = {sid:t for sid,t in truth.items() if groups[sid]==group}
        if not subset_truth:
            raise ValueError(f'Empty {group} cohort')
        subset_scores = scores[mask]
        if group=='calibration':
            model.threshold,_ = select_threshold(subset_truth,subset_rows,subset_scores)
        candidates = {}
        for r in subset_rows:
            candidates.setdefault(r[1],set()).add(r[5])
        pred = predictions_at(subset_rows,subset_scores,model.threshold)
        report[group] = {'entities':len(subset_truth),'singletons':sum(not t for t in subset_truth.values()),
                         'macro_f05':macro_f05(subset_truth,pred),
                         'blocking':blocking_metrics(subset_truth,candidates)}
        log(f'{group}: {report[group]}')
    report['threshold'] = model.threshold
    model.save(output/'model.joblib')
    (output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (output/'split_manifest.tsv').open('w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f,delimiter='\t',lineterminator='\n')
        writer.writerow(['source1_entity_id','cohort'])
        writer.writerows((sid,group) for _,sid,group in queries)
    return report
