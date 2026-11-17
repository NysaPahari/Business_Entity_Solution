"""Small from-scratch byte cross-encoder residual pilot; no external weights."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.special import expit, logit
from .data import connect, log, sql_string
from .evaluate import entity_f05
from .refinement import calibration_role, threshold_with_overrides


def encode_pair(row):
    # Separate budgets ensure all four fields survive truncation.
    result=[257]
    for text,cap in zip((row[2],row[3],row[6],row[7]),(48,80,48,80)):
        result.extend(b+1 for b in text.encode('utf-8')[:cap]); result.append(258)
    return result+[0]*(261-len(result))


class TextResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens=nn.Embedding(259,48,padding_idx=0)
        self.positions=nn.Embedding(261,48)
        layer=nn.TransformerEncoderLayer(48,4,128,.1,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.head=nn.Linear(48,1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, tokens):
        h=self.tokens(tokens)+self.positions(torch.arange(tokens.shape[1],device=tokens.device))
        h=self.encoder(h,src_key_padding_mask=tokens.eq(0))
        return self.head(h[:,0]).squeeze(-1)


def run():
    out=Path('artifacts/accuracy/neural_probe'); out.mkdir(exist_ok=True)
    status=out/'status.json'
    def state(value,**extra): status.write_text(json.dumps(dict(state=value,**extra),indent=2))
    state('preparing')
    torch.set_num_threads(1); torch.manual_seed(2026)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for this pilot')
    # Reuse the group pilot's disjoint, previously-unused training cohort and scores.
    import joblib
    group=Path('artifacts/accuracy/group_probe')
    queries=[]; rows=[]; raw=[]; labels=[]
    token_path=out/'tokens.npy'
    con=connect('work/group_probe.duckdb','256MB',1,read_only=True)
    try:
        queries=con.execute('SELECT qid,entity_id,cohort FROM queries ORDER BY qid').fetchall()
        con.execute(f"ATTACH {sql_string(Path('work/accuracy_composite.duckdb').resolve().as_posix())} AS original (READ_ONLY)")
        tokens=np.lib.format.open_memmap(token_path,mode='w+',dtype=np.int16,shape=(256000,261))
        offset=0
        for lo in range(1,2001,100):
            chunk=joblib.load(group/f'batch_{lo}.joblib')
            batch=con.execute(f'''SELECT q.qid,q.entity_id,s.name,s.address,s.country,
                r.entity_id,r.name,r.address,r.country,p.score FROM pool p JOIN queries q USING(qid)
                JOIN original.records s ON s.rid=q.rid JOIN original.records r ON r.rid=p.rid
                WHERE q.qid BETWEEN {lo} AND {lo+99} ORDER BY q.qid,r.entity_id''').fetchall()
            if [(r[0],r[1],r[5]) for r in batch]!=[(r[0],r[1],r[5]) for r in chunk['rows']]:
                raise ValueError('Cached baseline row mismatch')
            tokens[offset:offset+len(batch)]=np.asarray([encode_pair(r) for r in batch],dtype=np.int16)
            offset+=len(batch); rows.extend(chunk['rows']); raw.extend(chunk['raw']); labels.extend(chunk['y'])
        tokens.flush(); tokens=tokens[:offset]
    finally: con.close()
    ids={s for _,s,_ in queries}; truth={}
    with Path('dataset/train/train_ground_truth.tsv').open(encoding='utf-8',newline='') as stream:
        for r in csv.DictReader(stream,delimiter='\t'):
            if r['source1_entity_id'] in ids: truth[r['source1_entity_id']]=set(filter(None,r['matched_entity_ids'].split(',')))
    if set(truth)!=ids: raise ValueError('Missing truth')
    raw=np.asarray(raw); y=np.asarray(labels,dtype=np.float32); base=logit(np.clip(raw,1e-6,1-1e-6))
    fit_truth={s:truth[s] for _,s,c in queries if c=='calibration' and calibration_role(s)=='isotonic_fit'}
    tune_truth={s:truth[s] for _,s,c in queries if c=='calibration' and calibration_role(s)=='threshold_tune'}
    train_groups=defaultdict(list)
    for i,r in enumerate(rows):
        if r[0]<=1000: train_groups[r[0]].append(i)
    rng=np.random.default_rng(2026); train=[]; weights={}
    for indices in train_groups.values():
        positives=[i for i in indices if y[i]]
        negatives=sorted((i for i in indices if not y[i]),key=lambda i:-raw[i])
        hard=negatives[:8]; remainder=negatives[8:]
        random=list(rng.choice(remainder,min(8,len(remainder)),replace=False)) if remainder else []
        for i in positives+hard: weights[i]=1.
        for i in random: weights[i]=len(remainder)/len(random)
        train.extend(positives+hard+random)
    fit_indices=np.asarray([i for i,r in enumerate(rows) if r[1] in fit_truth])
    tune_indices=np.asarray([i for i,r in enumerate(rows) if r[1] in tune_truth])
    model=TextResidual().cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4)
    def predict(indices):
        model.eval(); scores=[]
        with torch.no_grad():
            for start in range(0,len(indices),64):
                ix=indices[start:start+64]
                batch=torch.as_tensor(np.asarray(tokens[ix],dtype=np.int64),device='cuda')
                scores.extend(expit(base[ix]+model(batch).float().cpu().numpy()))
        return np.asarray(scores)
    best=-1.; best_cutoff=None; epoch_reports=[]
    for epoch in range(1,4):
        state('training',epoch=epoch,epochs=3,training_pairs=len(train))
        model.train(); rng.shuffle(train)
        for start in range(0,len(train),32):
            ix=np.asarray(train[start:start+32]); optimizer.zero_grad(set_to_none=True)
            batch=torch.as_tensor(np.asarray(tokens[ix],dtype=np.int64),device='cuda')
            prediction=model(batch)+torch.as_tensor(base[ix],dtype=torch.float32,device='cuda')
            loss=nn.functional.binary_cross_entropy_with_logits(prediction,torch.as_tensor(y[ix],device='cuda'),reduction='none')
            w=torch.tensor([weights[i] for i in ix],device='cuda')
            (loss*w).mean().backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        state('calibrating',epoch=epoch)
        fit_scores=predict(fit_indices); zeros=np.zeros(len(fit_indices),dtype=bool)
        cutoff,score=threshold_with_overrides(fit_truth,[rows[i] for i in fit_indices],fit_scores,zeros,zeros)
        epoch_reports.append(dict(epoch=epoch,fit_macro_f05=score,cutoff=cutoff))
        log(f'Neural epoch {epoch}: calibration fit F0.5={score:.6f}')
        if score>best:
            best=score; best_cutoff=cutoff; torch.save(model.state_dict(),out/'best.pt')
    model.load_state_dict(torch.load(out/'best.pt',weights_only=True))
    tuned=predict(tune_indices)
    from .model import MatchingModel
    threshold=MatchingModel.load('artifacts/accuracy/selected/model.joblib').threshold
    def report(scores,cutoff):
        predictions=defaultdict(set)
        for i,p in zip(tune_indices,scores):
            if p>=cutoff: predictions[rows[i][1]].add(rows[i][5])
        return dict(entities=len(tune_truth),macro_f05=sum(entity_f05(t,predictions[s]) for s,t in tune_truth.items())/len(tune_truth),
                    fp=sum(len(predictions[s]-t) for s,t in tune_truth.items()),fn=sum(len(t-predictions[s]) for s,t in tune_truth.items()))
    result=dict(protocol='From-scratch byte cross-encoder residual. 1000 unused training entities; hard plus weighted random negatives. Three epochs/cutoff selected on calibration fit half, final comparison on tune half. No external weights/data, audit access, or promotion.',epochs=epoch_reports,
                parameters=sum(p.numel() for p in model.parameters()),selected_cutoff=best_cutoff,
                baseline=report(raw[tune_indices],threshold),neural=report(tuned,best_cutoff))
    (out/'results.json').write_text(json.dumps(result,indent=2)); state('complete'); log(json.dumps(result))


if __name__=='__main__':
    try: run()
    except Exception:
        import traceback
        folder=Path('artifacts/accuracy/neural_probe'); folder.mkdir(exist_ok=True)
        (folder/'status.json').write_text(json.dumps(dict(state='failed',error=traceback.format_exc()),indent=2))
        raise
