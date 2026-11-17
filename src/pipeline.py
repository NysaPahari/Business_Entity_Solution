"""CLI entry point: python -m src.pipeline --stage all."""
import argparse
import json
from pathlib import Path
from .blocking import build_index,generate_pool
from .data import prepare,log
from .infer import infer
from .train import train
from .submission import validate_streaming,package


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','train','predict','validate','package'],default='all')
    parser.add_argument('--dataset-dir',type=Path,default=Path('dataset'))
    parser.add_argument('--work-dir',type=Path,default=Path('work'))
    parser.add_argument('--artifact-dir',type=Path,default=Path('artifacts'))
    parser.add_argument('--output-dir',type=Path,default=Path('output'))
    parser.add_argument('--train-size',type=int,default=12000)
    parser.add_argument('--tune-size',type=int,default=3000)
    parser.add_argument('--audit-size',type=int,default=3000)
    parser.add_argument('--top-k',type=int,default=20)
    parser.add_argument('--pool-size',type=int,default=32)
    parser.add_argument('--max-df',type=int,default=128)
    parser.add_argument('--blocking-strategy',choices=['baseline','composite'],default='baseline')
    parser.add_argument('--retrieval-batch-size',type=int,default=50000)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--memory',default='2GB')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    if args.retrieval_batch_size<1:
        parser.error('--retrieval-batch-size must be positive')
    if not 1<=args.top_k<=args.pool_size or min(args.train_size,args.tune_size,args.audit_size)<1:
        parser.error('Require positive cohort sizes and 1 <= top-k <= pool-size')
    if args.stage in ('all','train','predict'):
        for split in (('train','test') if args.stage=='all' else ('train',) if args.stage=='train' else ('test',)):
            database=prepare(args.dataset_dir,split,args.work_dir,args.train_size,args.tune_size,
                             args.audit_size,args.seed,args.memory,args.threads)
            build_index(database,args.max_df,args.memory,args.threads,strategy=args.blocking_strategy)
            generate_pool(database,args.pool_size,batch_size=args.retrieval_batch_size,memory=args.memory,threads=args.threads)
            if split=='train':
                train(database,args.dataset_dir,args.artifact_dir,args.top_k,args.seed,args.threads,args.memory)
            else:
                infer(database,args.artifact_dir/'model.joblib',args.output_dir,args.top_k,args.memory,args.threads,args.workers)
    if args.stage in ('all','predict','validate'):
        errors,warnings=validate_streaming(args.output_dir/'matching_results.tsv',
            args.output_dir/'candidate_pairs.tsv',args.dataset_dir/'test',True,args.memory)
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/'validation.json').write_text(json.dumps({'passed':not errors,'errors':errors,'warnings':warnings},indent=2))
        if errors:
            raise RuntimeError('Submission validation failed: '+str(errors))
        log('PASS: strict submission validation, including ID existence and candidate subsets')
    if args.stage=='package':
        if all((args.work_dir/f'{split}.duckdb').exists() for split in ('train','test')):
            from .eda import profile
            profile(args.work_dir,args.artifact_dir)
        from .report import write_report
        write_report(Path.cwd(),args.artifact_dir,args.output_dir)
        package(Path.cwd(),args.output_dir,args.artifact_dir,Path('dist/business_entity_resolution_submission.zip'))


if __name__=='__main__':
    main()
