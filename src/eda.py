"""Summarize the supplied records without any country-specific assumptions."""
import argparse
import json
from pathlib import Path
from .data import connect


def profile(work_dir,artifact_dir):
    report={}
    for split in ('train','test'):
        con=connect(Path(work_dir)/f'{split}.duckdb','256MB',1,read_only=True)
        try:
            rows=con.execute('''SELECT source,country,count(*) AS records,
                sum(CASE WHEN name='' THEN 1 ELSE 0 END) AS missing_names,
                sum(CASE WHEN address='' THEN 1 ELSE 0 END) AS missing_addresses,
                avg(length(name)) AS mean_name_length,avg(length(address)) AS mean_address_length
                FROM records GROUP BY source,country ORDER BY source,country''').fetchall()
            names=['source','country','records','missing_names','missing_addresses',
                   'mean_name_length','mean_address_length']
            report[split]=[dict(zip(names,row)) for row in rows]
        finally:
            con.close()
    output=Path(artifact_dir)/'data_profile.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',default='work')
    parser.add_argument('--artifact-dir',default='artifacts')
    args=parser.parse_args()
    print(json.dumps(profile(args.work_dir,args.artifact_dir),indent=2))
