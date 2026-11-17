"""Build a minimal, verified submission without altering working files."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import tokenize
import zipfile
from .team_identity import load_identity, methodology_identity


def clean_source(source):
    original_source=source
    tree=ast.parse(source)
    lines=source.splitlines(keepends=True)
    edits=[]
    for node in ast.walk(tree):
        body=getattr(node,'body',None)
        if isinstance(body,list) and body and isinstance(body[0],ast.Expr):
            value=body[0].value
            if isinstance(value,ast.Constant) and isinstance(value.value,str):
                first=next((s.strip() for s in value.value.splitlines() if s.strip()),'')
                doc=body[0]
                edits.append((doc.lineno,doc.end_lineno,' '*doc.col_offset+repr(first)+'\n'))
    for start,end,replacement in sorted(set(edits),reverse=True):
        lines[start-1:end]=[replacement]+['\n']*(end-start)
    source=''.join(lines)
    tokens=list(tokenize.generate_tokens(io.StringIO(source).readline))
    source=tokenize.untokenize([t._replace(string='') if t.type==tokenize.COMMENT else t for t in tokens])
    def semantic_tree(text):
        parsed=ast.parse(text)
        for node in ast.walk(parsed):
            body=getattr(node,'body',None)
            if isinstance(body,list) and body and isinstance(body[0],ast.Expr):
                value=body[0].value
                if isinstance(value,ast.Constant) and isinstance(value.value,str): body.pop(0)
        return ast.dump(parsed,include_attributes=False)
    if semantic_tree(source)!=semantic_tree(original_source):
        raise ValueError('Cleanup changed executable code')
    return source


def clean_archive(source_path,target_path,improved=False):
    identity=load_identity()
    prefix='code/business_entity_resolution/'
    with zipfile.ZipFile(source_path) as source:
        members=set(source.namelist())
        roots=['pipeline','normalize']+(['push_accuracy'] if improved else [])
        needed=set(); pending=roots[:]
        while pending:
            module=pending.pop()
            if module in needed: continue
            path=prefix+'src/'+module+'.py'
            if path not in members: raise ValueError(f'Missing source dependency: {module}')
            needed.add(module)
            tree=ast.parse(source.read(path).decode('utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom) and node.level==1 and node.module:
                    pending.append(node.module.split('.')[0])
        selected={prefix+'src/'+name+'.py' for name in needed}
        if prefix+'src/__init__.py' in members: selected.add(prefix+'src/__init__.py')
        selected|={'output/matching_results.tsv','output/candidate_pairs.tsv','Documentation_template.md',prefix+'requirements.txt'}
        renamed={}
        if improved:
            for filename in ('model.joblib','inference_config.json'):
                flat=prefix+'artifacts/'+filename
                original=flat if flat in members else prefix+'artifacts/accuracy/selected/'+filename
                selected.add(original)
                renamed[original]=flat
        else:
            selected|={prefix+'artifacts/model.joblib',prefix+'artifacts/metrics.json',prefix+'artifacts/split_manifest.tsv'}
        if not selected<=members: raise ValueError(f'Missing required members: {selected-members}')
        model_dir='artifacts'
        args='--blocking-strategy composite --max-df 4096 --pool-size 128 --top-k 128' if improved else '--max-df 128 --pool-size 32 --top-k 20'
        batch_arg=' --retrieval-batch-size 8000' if '--retrieval-batch-size' in source.read(prefix+'src/pipeline.py').decode('utf-8-sig') else ''
        score='0.94547987 fresh-audit' if improved else '0.87114969 baseline-audit'
        identity_note='Team identity is recorded in that document.' if identity else 'Fill in your team identity there before final upload.'
        readme=f'''# Business entity resolution submission

This archive contains the {'improved' if improved else 'completed baseline'} output. Local {score} macro F0.5 is not a leaderboard score.

## Reproduce the supplied outputs

Use 64-bit Python 3.12. Run from this directory. Place the supplied six original
files and training ground truth under dataset/train and dataset/test with their
original filenames. The trusted fitted model and TF-IDF are included; no external
business data or remote service is used.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m src.normalize --dataset-dir dataset
.venv/Scripts/python.exe -m src.pipeline --stage predict --dataset-dir dataset --artifact-dir {model_dir} --work-dir work --output-dir output {args} --memory 4GB --threads 1{batch_arg} --workers 1
```

On Linux/macOS use .venv/bin/python. Prediction performs normalization-input
ingestion, blocking, feature extraction, model scoring, and strict validation.
The resulting two TSVs are in output/. Check output/validation.json for success.
All S1 records must occur once; matches must be valid S2/S3 IDs and subsets of
the exported candidates. Keep both TSVs together. Use a fresh work directory if
changing inputs or retrieval parameters. Allow substantial runtime and disk space.

The supplied model is LightGBM (MIT license), with no external neural weights.
Its fitted extractor and threshold are embedded in model.joblib. Source includes
the scoring and candidate generation implementation for audit. Use only trusted
joblib artifacts. Full methodology and measured limitations are in the archive's
Documentation_template.md. {identity_note}

Upload output/matching_results.tsv to the leaderboard and this whole archive for
the final review. No leaderboard-upload history is available locally: ensure the
uploaded TSV is this exact file. The filename may be renamed to your team name.
'''
        target=Path(target_path); temporary=target.with_suffix('.zip.tmp')
        hashes={}
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as archive:
            for name in sorted(selected):
                if name.endswith('.py'):
                    original=source.read(name).decode('utf-8-sig')
                    cleaned=clean_source(original)
                    compile(cleaned,name,'exec')
                    archive.writestr(name,cleaned)
                elif name.startswith('output/'):
                    h=hashlib.sha256()
                    with source.open(name) as inp,archive.open(name,'w',force_zip64=True) as dst:
                        for block in iter(lambda:inp.read(8*1024*1024),b''):
                            dst.write(block); h.update(block)
                    hashes[name]=h.hexdigest()
                elif name=='Documentation_template.md':
                    archive.writestr(name,methodology_identity(source.read(name).decode('utf-8-sig'),identity))
                else: archive.writestr(renamed.get(name,name),source.read(name))
            archive.writestr(prefix+'README.md',readme)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None: raise ValueError('Archive CRC failure')
            for name,expected in hashes.items():
                h=hashlib.sha256()
                with archive.open(name) as stream:
                    for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
                if h.hexdigest()!=expected: raise ValueError('Output bytes changed')
            for name in archive.namelist():
                if name.endswith('.py'):
                    text=archive.read(name).decode('utf-8')
                    for node in ast.walk(ast.parse(text)):
                        body=getattr(node,'body',None)
                        if isinstance(body,list) and body and isinstance(body[0],ast.Expr):
                            value=body[0].value
                            if isinstance(value,ast.Constant) and isinstance(value.value,str) and len(value.value.splitlines())>2:
                                raise ValueError('Long docstring remains')
    temporary.replace(target)
    print(f'Clean submission: {target.resolve()} ({target.stat().st_size:,} bytes)',flush=True)
    return target


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--improved',action='store_true'); args=p.parse_args()
    source='dist/business_entity_resolution_improved_submission.zip' if args.improved else 'dist/business_entity_resolution_submission.zip'
    target='dist/business_entity_resolution_improved_submission.zip' if args.improved else 'dist/business_entity_resolution_clean_submission.zip'
    clean_archive(source,target,args.improved)
