# Improved business entity resolution submission

Run commands from this directory with Python 3.12. Install dependencies with
`python -m pip install -r requirements.txt`. Place the supplied original data in
`dataset/train` and `dataset/test`, preserving its original filenames. No external
business data is needed. The trusted fitted model is included under
`artifacts/accuracy/selected`; source and fitted TF-IDF are included.

## Reproduce submitted predictions

```powershell
python -m src.normalize --help
python -m src.normalize
python -m src.pipeline --stage predict --artifact-dir artifacts/accuracy/selected --work-dir work_accuracy_test --output-dir output_improved --blocking-strategy composite --max-df 4096 --pool-size 128 --top-k 128 --memory 4GB --threads 1 --retrieval-batch-size 8000 --workers 6
python -m src.finalize_improved
```

See NORMALIZATION.md for normalization arguments. Prediction runs the complete
blocking, feature, matching and strict validation stages. Reduce batch size or
workers if RAM is constrained. Use the same work directory to resume an
interruption; do not run simultaneous writers. Final files are under
`output_improved`. The archive places these same files under `output` as required.

## Training provenance and verification

The included model was trained on 42,000 training S1 entities and 2,687,935 pairs.
It uses 600 LightGBM trees and 36 features. See Documentation_template.md at the
archive root and artifacts/accuracy/ACCURACY_REPORT.md for the experimental
sequence, calibration selection, and evaluation limitations. Training experiment
entry points are src.pipeline (baseline training), src.push_accuracy,
src.more_training, and src.full_pool; REFINEMENT.md documents prerequisites.
The included selected model permits prediction reproduction without repeating
the expensive training experiments or needing their caches.

Run unit tests with `python -m unittest discover -s tests -v`.
