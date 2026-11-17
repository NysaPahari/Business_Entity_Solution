"""Record provenance only after the improved full-test outputs pass validation."""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): value.update(block)
    return value.hexdigest()


def finalize():
    output=Path('output_improved')
    validation=json.loads((output/'validation.json').read_text())
    if not validation.get('passed') or validation.get('errors'):
        raise ValueError('Full output validation must pass first')
    metrics=json.loads((output/'inference_metrics.json').read_text())
    model=Path('artifacts/accuracy/selected/model.joblib')
    model_hash=digest(model)
    if metrics.get('model_sha256')!=model_hash:
        raise ValueError('Generated outputs do not identify the selected model fingerprint')
    info=dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),validation_passed=True,
              model_path=str(model.resolve()),model_sha256=model_hash,
              retrieval=json.loads(Path('artifacts/accuracy/selected/inference_config.json').read_text()),
              inference=metrics,
              source_manifest=json.loads(Path('work_accuracy_test/test_ingest.json').read_text()),files={})
    for name in ('matching_results.tsv','candidate_pairs.tsv'):
        path=output/name
        info['files'][name]=dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=digest(path))
    (output/'run_info.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    from .accuracy_report import write
    write()
    print(f'Validated improved submission: {(output/"matching_results.tsv").resolve()}')
    return info


if __name__=='__main__': finalize()
