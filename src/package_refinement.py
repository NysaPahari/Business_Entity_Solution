"""Package refinement code, reports and trusted local models for team review."""
import hashlib
import json
from pathlib import Path
import zipfile


def package(root='.'):
    root = Path(root).resolve()
    files = []
    for folder in ('src', 'tests'):
        files.extend((root / folder).glob('*.py'))
    files.extend(root / name for name in (
        'README.md', 'REFINEMENT.md', 'HANDOVER_CURRENT.md', 'NORMALIZATION.md', 'requirements.txt',
        'config.json', 'run_improved_predictions.ps1', 'artifacts/model.joblib', 'artifacts/metrics.json',
        'artifacts/split_manifest.tsv'))
    # Raw/feature caches and DuckDB databases are regenerated from the datasets.
    excluded = {'baseline_pairs.joblib', 'reverse_nearest.joblib', 'mutual_feature.npy'}
    for folder in ('refinement','accuracy'):
        for path in (root / 'artifacts' / folder).rglob('*'):
            if path.name.startswith('batch_') and path.suffix == '.joblib':
                continue
            if 'extra_training' in path.relative_to(root).parts:
                continue
            if path.is_file() and path.name not in excluded and path.suffix in {'.md', '.json', '.tsv', '.joblib', '.txt'}:
                files.append(path)
    files = sorted(set(files))
    manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    target = root / 'dist/entity_resolution_refinement.zip'
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr('SHA256_MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC verification failed')
        for name, digest in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f'Archive content mismatch: {name}')
    print(f'Verified {len(files)} files: {target} ({target.stat().st_size:,} bytes)')
    return target


if __name__ == '__main__':
    package()
