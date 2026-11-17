"""Run resumable improved prediction, validation, provenance, and packaging."""
import json
import argparse
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--memory',default='4GB')
    args=parser.parse_args()
    if args.workers<1: parser.error('workers must be positive')
    status = Path('work_accuracy_test/submission_job.json')
    def record(state, **extra):
        status.write_text(json.dumps(dict(state=state, updated_at_utc=datetime.now(timezone.utc).isoformat(), **extra),indent=2))
    try:
        record('running')
        with Path('work_accuracy_test/prediction.log').open('a',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u','-m','src.pipeline','--stage','predict',
                '--artifact-dir','artifacts/accuracy/selected','--work-dir','work_accuracy_test',
                '--output-dir','output_improved','--blocking-strategy','composite','--max-df','4096',
                '--pool-size','128','--top-k','128','--memory',args.memory,'--threads','1',
                '--retrieval-batch-size','8000','--workers',str(args.workers)],stdout=log,stderr=subprocess.STDOUT,check=True)
        record('finalizing')
        from .finalize_improved import finalize
        finalize()
        record('packaging')
        from .package_improved import package
        target = package()
        record('complete', bundle=str(target))
    except Exception:
        record('failed', error=traceback.format_exc())
        raise


if __name__ == '__main__': main()
