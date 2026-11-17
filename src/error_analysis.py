"""Reproduce baseline validation errors without fitting or changing candidates."""
import argparse
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import joblib
import numpy as np

from .blocking import candidate_rows
from .data import connect, log, sql_string
from .evaluate import entity_f05, blocking_metrics
from .features import final_candidates
from .model import MatchingModel


def collect(database='work/train.duckdb', model_path='artifacts/model.joblib',
            dataset_dir='dataset', output_dir='artifacts/refinement'):
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    cache=out/'baseline_pairs.joblib'
    if cache.exists():
        return joblib.load(cache)
    model=MatchingModel.load(model_path)
    con=connect(database,'2GB',2,read_only=True)
    queries=con.execute('''SELECT q.qid,q.entity_id,q.cohort,r.name,r.address,r.country
                           FROM queries q JOIN records r USING(rid) ORDER BY q.qid''').fetchall()
    query_map={q[1]:dict(qid=q[0],cohort=q[2],name=q[3],address=q[4],country=q[5]) for q in queries}
    truth={}
    with (Path(dataset_dir)/'train/train_ground_truth.tsv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if r['source1_entity_id'] in query_map:
                truth[r['source1_entity_id']]=set(filter(None,r['matched_entity_ids'].split(',')))
    if set(truth)!=set(query_map):
        raise ValueError('Selected queries are missing ground truth')
    rows,features,selected=[],[],[]
    for lo in range(1,max(q[0] for q in queries)+1,1000):
        batch=candidate_rows(con,lo,lo+999)
        x=model.extractor.transform(batch)
        chosen=final_candidates(batch,x,20)
        selected.extend(len(rows)+int(i) for i in chosen)
        rows.extend(batch)
        features.append(x)
        log(f'Reconstructed query features through {lo+999:,}')
    con.close()
    x=np.concatenate(features)
    result=dict(rows=rows,x=x,selected=np.asarray(selected),queries=query_map,truth=truth,
                baseline_scores=model.probabilities(x),threshold=model.threshold)
    joblib.dump(result,cache,compress=3)
    return result


def summarize(data, indices, probabilities, predicted=None, cohort='audit'):
    rows=[data['rows'][i] for i in indices]
    candidates=defaultdict(set)
    predictions=defaultdict(set)
    scores={}
    for j,(r,p) in enumerate(zip(rows,probabilities)):
        if data['queries'][r[1]]['cohort']!=cohort:
            continue
        candidates[r[1]].add(r[5])
        scores[r[1],r[5]]=float(p)
        if (p>=data['threshold']) if predicted is None else bool(predicted[j]):
            predictions[r[1]].add(r[5])
    truth={s:t for s,t in data['truth'].items() if data['queries'][s]['cohort']==cohort}
    segments=defaultdict(lambda:dict(entities=0,fp=0,fn=0,tp=0,blocked_fn=0,score_sum=0.0))
    errors=[]
    for sid,t in truth.items():
        pred=predictions[sid]
        fp,fn=pred-t,t-pred
        blocked=t-candidates[sid]
        count=len(candidates[sid])
        bucket='0' if count==0 else '1' if count==1 else 'many'
        kind='singleton' if not t else 'real_match'
        country=data['queries'][sid]['country']
        keys=[('all','all'),('country',country),('candidate_count',bucket),('truth_type',kind),
              ('cross',f'{country}|{bucket}|{kind}')]
        for key in keys:
            stats=segments[key]
            stats['entities']+=1
            stats['fp']+=len(fp)
            stats['fn']+=len(fn)
            stats['tp']+=len(t & pred)
            stats['blocked_fn']+=len(blocked)
            stats['score_sum']+=entity_f05(t,pred)
        for label,ids in [('false_positive',fp),('false_negative',fn)]:
            for cid in sorted(ids):
                errors.append(dict(source1_entity_id=sid,candidate_entity_id=cid,error=label,
                    country=country,candidate_count=count,candidate_bucket=bucket,truth_type=kind,
                    true_label=int(cid in t),probability=scores.get((sid,cid)),
                    blocked=cid not in candidates[sid],entity_f05=entity_f05(t,pred)))
    table=[]
    for (dimension,segment),stats in sorted(segments.items()):
        stats=dict(stats)
        stats['macro_f05']=stats.pop('score_sum')/stats['entities']
        table.append(dict(dimension=dimension,segment=segment,**stats))
    blocking=blocking_metrics(truth,candidates)
    return dict(cohort=cohort,segments=table,blocking=blocking,
                blocking_bottleneck=blocking['pair_recall']<0.97),errors,candidates


def enrich_errors(errors,dataset_dir,database,model):
    if not errors:
        return errors
    wanted={r[k] for r in errors for k in ('source1_entity_id','candidate_entity_id')}
    con=connect(':memory:','1GB',2)
    con.execute('CREATE TABLE wanted(entity_id VARCHAR)')
    con.executemany('INSERT INTO wanted VALUES (?)',[(s,) for s in wanted])
    fields={}
    for i in (1,2,3):
        path=Path(dataset_dir)/'train'/f'train_source{i}_normalized.tsv'
        result=con.execute(f'''SELECT s.entity_id,s.business_name,s.business_address,
            s.business_name_normalized,s.business_address_normalized,s.country
            FROM read_csv({sql_string(path.as_posix())},delim='\t',header=true,all_varchar=true,
                          nullstr='__NO_NULL_SENTINEL__') s JOIN wanted USING(entity_id)''').fetchall()
        fields.update({r[0]:r[1:] for r in result})
    con.close()
    blocked=[]
    for error in errors:
        a,b=fields[error['source1_entity_id']],fields[error['candidate_entity_id']]
        for prefix,value in [('source1',a),('candidate',b)]:
            error.update({prefix+'_'+key:text for key,text in zip(
                ('raw_name','raw_address','normalized_name','normalized_address'),value[:4])})
        if error['blocked']:
            blocked.append((error,(0,error['source1_entity_id'],a[2],a[3],a[4],
                                   error['candidate_entity_id'],b[2],b[3],b[4],0.0)))
    # Counterfactual scores are explicitly separate: blocked pairs were NOT scored
    # by the submitted pipeline, and no diagnostic pair is injected into candidates.
    for start in range(0,len(blocked),5000):
        chunk=blocked[start:start+5000]
        p=model.probabilities(model.extractor.transform([r for _,r in chunk]))
        for (error,_),score in zip(chunk,p):
            error['diagnostic_only_probability']=float(score)
    return errors


def write_analysis(data, output_dir='artifacts/refinement',dataset_dir='dataset',
                   database='work/train.duckdb',model_path='artifacts/model.joblib'):
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    reports={}
    for cohort in ('calibration','audit'):
        indices=data['selected']
        summary,errors,candidates=summarize(data,indices,data['baseline_scores'][indices],cohort=cohort)
        reports[cohort]=summary
        errors=enrich_errors(errors,dataset_dir,database,MatchingModel.load(model_path))
        (out/f'{cohort}_errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
        with (out/f'{cohort}_candidate_pairs.tsv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.writer(stream,delimiter='\t',lineterminator='\n')
            writer.writerow(['source1_entity_id','candidate_entity_ids'])
            writer.writerows((sid,','.join(sorted(candidates.get(sid,())))) for sid in data['truth']
                             if data['queries'][sid]['cohort']==cohort)
        with (out/f'{cohort}_error_segments.tsv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summary['segments'][0]),delimiter='\t',lineterminator='\n')
            writer.writeheader(); writer.writerows(summary['segments'])
        fp=sorted((e for e in errors if e['error']=='false_positive'),key=lambda e:-e['probability'])[:15]
        fn=sorted((e for e in errors if e['error']=='false_negative'),
                  key=lambda e:(e['probability'] is not None,e['probability'] or 0,e['source1_entity_id']))[:15]
        fn_scored=sorted((e for e in errors if e['error']=='false_negative' and not e['blocked']),
                         key=lambda e:e['probability'])[:15]
        text=[f'# {cohort.title()} error examples',
              'Top false positives are ordered by confidence. Top false negatives prioritize blocking misses, then lowest scored probabilities.',
              'A blocked pair has no production probability. Its diagnostic-only score is counterfactual, with retrieval strength set to zero.']
        for heading,items in [('Top 15 false positives',fp),('Top 15 false negatives',fn),
                             ('Supplement: top 15 scored false negatives',fn_scored)]:
            text.append('\n## '+heading)
            for n,e in enumerate(items,1):
                text.append(f'\n### {n}. {e["source1_entity_id"]} / {e["candidate_entity_id"]}')
                text.append(f'True label: {e["true_label"]}; probability: {e["probability"]}; blocked: {e["blocked"]}; diagnostic-only probability: {e.get("diagnostic_only_probability","not applicable")}.')
                for prefix in ('source1','candidate'):
                    text.append(f'\n**{prefix} raw name:** '+e[prefix+'_raw_name'])
                    text.append(f'\n**{prefix} raw address:** '+e[prefix+'_raw_address'])
                    text.append(f'\n**{prefix} normalized name:** '+e[prefix+'_normalized_name'])
                    text.append(f'\n**{prefix} normalized address:** '+e[prefix+'_normalized_address'])
        (out/f'{cohort}_top_errors.md').write_text('\n\n'.join(text),encoding='utf-8')
        log(f'{cohort}: '+json.dumps(summary['segments'][0])+'; blocking '+json.dumps(summary['blocking']))
    (out/'baseline_error_analysis.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    return reports


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default='artifacts/refinement')
    args=parser.parse_args()
    write_analysis(collect(output_dir=args.output_dir),output_dir=args.output_dir)
