"""Wait for an existing prediction process and finalize validated output only."""
import argparse
import json
import time
from pathlib import Path
import psutil
from .finalize_improved import finalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int, required=True)
    args = parser.parse_args()
    process = psutil.Process(args.wait_pid)
    command = process.cmdline()
    if 'src.pipeline' not in command or 'work_accuracy_test' not in command:
        raise ValueError('Refusing to follow an unrelated process')
    print(f'Waiting for prediction process {args.wait_pid}', flush=True)
    while process.is_running():
        time.sleep(15)
    # finalize checks strict validation and the generated model fingerprint.
    info = finalize()
    status = Path('IMPROVED_OUTPUT_STATUS.md')
    status.write_text(
        '# Improved output ready\n\n'
        f'Completed (UTC): {info["completed_at_utc"]}\n\n'
        'Strict submission validation passed. Submit '
        '`output_improved/matching_results.tsv`. Keep '
        '`output_improved/candidate_pairs.tsv` with it.\n\n'
        'Selected-model fresh-audit macro F0.5: 0.9454798696591747. '
        'This is not a hidden-test accuracy score.\n\n'
        'See `output_improved/run_info.json` for output hashes and provenance.\n',
        encoding='utf-8')
    print(json.dumps(info['files'], indent=2), flush=True)


if __name__ == '__main__':
    main()
