"""Create the required submission layout from validated improved predictions."""
import json
from pathlib import Path
import zipfile
from .finalize_improved import digest


def package():
    root = Path.cwd()
    output = root/'output_improved'
    info = json.loads((output/'run_info.json').read_text())
    validation = json.loads((output/'validation.json').read_text())
    if not info.get('validation_passed') or not validation.get('passed') or validation.get('errors'):
        raise ValueError('Validated improved outputs required')
    if info['model_sha256'] != digest(root/'artifacts/accuracy/selected/model.joblib'):
        raise ValueError('Selected model changed after inference')
    if info['inference']['entities'] != 1732544:
        raise ValueError('Unexpected number of test entities')
    files = {}
    for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
        if digest(output/name) != info['files'][name]['sha256']:
            raise ValueError(f'Output changed since validation: {name}')
        files['output/'+name] = output/name
    prefix = 'code/business_entity_resolution/'
    for folder in ('src','tests','utils'):
        for path in (root/folder).rglob('*.py'):
            files[prefix+path.relative_to(root).as_posix()] = path
    for name in ('requirements.txt','NORMALIZATION.md','REFINEMENT.md','run_improved_predictions.ps1',
                 'SUBMISSION_README.md','Documentation_improved.md'):
        files[prefix+name] = root/name
    for path in (root/'artifacts/accuracy/selected').glob('*'):
        if path.is_file(): files[prefix+path.relative_to(root).as_posix()] = path
    selection = json.loads((root/'artifacts/accuracy/selection.json').read_text())
    for name in ('ACCURACY_REPORT.md','fresh_audit.json','selection.json','results.json',
                 selection['selected_stage']+'.joblib'):
        files[prefix+'artifacts/accuracy/'+name] = root/'artifacts/accuracy'/name
    files[prefix+'artifacts/refinement/ablation_results.json'] = root/'artifacts/refinement/ablation_results.json'
    files[prefix+'README.md'] = root/'SUBMISSION_README.md'
    files['Documentation_template.md'] = root/'Documentation_improved.md'
    for name in ('validation.json','run_info.json','inference_metrics.json'):
        files['output/'+name] = output/name
    target = root/'dist/business_entity_resolution_improved_submission.zip'
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as archive:
        for name,path in sorted(files.items()): archive.write(path,name)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None: raise ValueError('Archive CRC validation failed')
        if set(archive.namelist()) != set(files): raise ValueError('Archive layout mismatch')
    temporary.replace(target)
    from .clean_submission import clean_archive
    clean_archive(target,target,improved=True)
    from .team_identity import load_identity
    identity=load_identity()
    if identity:
        named=target.parent/(identity['team_name']+'_submission.zip')
        target.replace(named)
        target=named
    print(f'Submission bundle ready: {target}',flush=True)
    return target


if __name__ == '__main__': package()
