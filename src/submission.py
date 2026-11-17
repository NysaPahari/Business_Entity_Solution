"""Strict bounded-memory submission validation and reproducible zip assembly."""
import csv
import json
from pathlib import Path
import zipfile
import shutil
from .data import connect,sql_string,log


def validate_streaming(matching_path,candidate_path,test_dir,check_ids=True,memory='1GB'):
    """Validate format/subsets in lockstep and membership with disk-spilling SQL.

    Both generated files must have the same row order (the pipeline guarantees
    this). Original Source 1 order is irrelevant. Every invariant is a hard error.
    """
    errors = []
    count = 0
    def error(message):
        if len(errors)<20:
            errors.append(message)
    with open(matching_path,encoding='utf-8',newline='') as m, open(candidate_path,encoding='utf-8',newline='') as c:
        readers = [csv.reader(m,delimiter='\t'),csv.reader(c,delimiter='\t')]
        if next(readers[0],None)!=['source1_entity_id','matched_entity_ids']:
            error('Incorrect matching header')
        if next(readers[1],None)!=['source1_entity_id','candidate_entity_ids']:
            error('Incorrect candidate header')
        from itertools import zip_longest
        for mr,cr in zip_longest(*readers):
            count+=1
            if mr is None or cr is None or len(mr)!=2 or len(cr)!=2:
                error(f'Row {count}: mismatched row counts or malformed columns')
                continue
            if mr[0]!=cr[0]:
                error(f'Row {count}: candidate/matching Source 1 IDs differ')
            parsed=[]
            for row in (mr,cr):
                ids = row[1].split(',') if row[1] else []
                if len(ids)!=len(set(ids)) or any(not x.startswith(('S2-','S3-')) or x!=x.strip() for x in ids):
                    error(f'Row {count}: invalid or duplicate target IDs')
                parsed.append(set(ids))
            if not parsed[0] <= parsed[1]:
                error(f'Row {count}: a match is absent from the actual candidate set')
    if errors:
        return errors,[]
    con=connect(':memory:',memory,2)
    def read(path):
        return f"read_csv({sql_string(Path(path).as_posix())},delim='\t',header=true,all_varchar=true,nullstr='__NO_NULL_SENTINEL__')"
    try:
        source=Path(test_dir)
        con.execute(f'CREATE TABLE required AS SELECT entity_id FROM {read(source/"test_source1.tsv")}')
        con.execute(f'CREATE TABLE submitted AS SELECT source1_entity_id AS entity_id FROM {read(matching_path)}')
        for table in ('required','submitted'):
            duplicates=con.execute(f'SELECT count(*)-count(DISTINCT entity_id) FROM {table}').fetchone()[0]
            if duplicates:
                error(f'{table}: {duplicates} duplicate Source 1 IDs')
        difference=con.execute('''SELECT count(*) FROM
            ((SELECT entity_id FROM required EXCEPT SELECT entity_id FROM submitted)
             UNION ALL (SELECT entity_id FROM submitted EXCEPT SELECT entity_id FROM required))''').fetchone()[0]
        if difference:
            error(f'{difference} missing or unexpected Source 1 IDs')
        if check_ids:
            log('Validating candidate membership in real test Source 2/3 records')
            con.execute(f'''CREATE TABLE valid AS SELECT entity_id FROM {read(source/'test_source2.tsv')}
                UNION ALL SELECT entity_id FROM {read(source/'test_source3.tsv')}''')
            invalid=con.execute(f'''SELECT count(*) FROM
                (SELECT unnest(string_split(candidate_entity_ids,',')) AS entity_id
                 FROM {read(candidate_path)} WHERE candidate_entity_ids<>'') p
                ANTI JOIN valid v USING(entity_id)''').fetchone()[0]
            if invalid:
                error(f'{invalid} candidate IDs do not exist in test references')
    finally:
        con.close()
    warnings=[] if check_ids else ['Target ID existence check skipped']
    log(f'Checked {count:,} output rows')
    return errors,warnings


def package(root,output_dir,artifact_dir,destination):
    root,output_dir,artifact_dir,destination=map(Path,(root,output_dir,artifact_dir,destination))
    destination.parent.mkdir(parents=True,exist_ok=True)
    prefix='code/business_entity_resolution/'
    validation=json.loads((output_dir/'validation.json').read_text())
    if not validation.get('passed'):
        raise ValueError('A successful strict validation is required before packaging')
    staging=root/'code/business_entity_resolution'
    staging.mkdir(parents=True,exist_ok=True)
    for folder in ('src','tests','utils'):
        for path in (root/folder).rglob('*.py'):
            target=staging/path.relative_to(root)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
    for name in ('README.md','NORMALIZATION.md','PROBLEM_STATEMENT.md','requirements.txt','config.json'):
        if (root/name).exists():
            shutil.copy2(root/name,staging/name)
    (staging/'artifacts').mkdir(exist_ok=True)
    for name in ('model.joblib','metrics.json','split_manifest.tsv'):
        shutil.copy2(artifact_dir/name,staging/'artifacts'/name)
    if (artifact_dir/'data_profile.json').exists():
        shutil.copy2(artifact_dir/'data_profile.json',staging/'artifacts/data_profile.json')
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for folder in ('src','tests','utils'):
            for path in (root/folder).rglob('*.py'):
                archive.write(path,prefix+path.relative_to(root).as_posix())
        for name in ('README.md','NORMALIZATION.md','PROBLEM_STATEMENT.md','requirements.txt','config.json'):
            path=root/name
            if path.exists():
                archive.write(path,prefix+name)
        for name in ('matching_results.tsv','candidate_pairs.tsv','inference_metrics.json','validation.json'):
            archive.write(output_dir/name,'output/'+name)
        for name in ('model.joblib','metrics.json','split_manifest.tsv'):
            archive.write(artifact_dir/name,prefix+'artifacts/'+name)
        if (artifact_dir/'data_profile.json').exists():
            archive.write(artifact_dir/'data_profile.json',prefix+'artifacts/data_profile.json')
        archive.write(root/'Documentation_template.md','Documentation_template.md')
    log(f'Packaged {destination}')
