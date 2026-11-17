"""Fill the methodology write-up from actual local run artifacts."""
import json
from pathlib import Path


def write_report(root=Path('.'),artifact_dir=Path('artifacts'),output_dir=Path('output')):
    root,artifact_dir,output_dir=map(Path,(root,artifact_dir,output_dir))
    metrics=json.loads((artifact_dir/'metrics.json').read_text())
    inference=json.loads((output_dir/'inference_metrics.json').read_text())
    validation=json.loads((output_dir/'validation.json').read_text())
    results='\n'.join(
        f"| {cohort} | {metrics[cohort]['entities']:,} | {metrics[cohort]['macro_f05']:.6f} | "
        f"{metrics[cohort]['blocking']['pair_recall']:.6f} | {metrics[cohort]['blocking']['oracle_macro_f05']:.6f} |"
        for cohort in ('train','calibration','audit'))
    important='\n'.join(f'| {name} | {gain:.3f} |'
                        for name,gain in list(metrics['feature_importance_gain'].items())[:10])
    text=f'''# ML Challenge 2026: Business Entity Resolution

**Team name and members:** To be filled in by the submitting team.

## 1. Executive Summary

This local baseline links each Source 1 record to zero or more Source 2/3 records.
Generic normalization and disk-backed blocking reduce the search space; a small
LightGBM classifier scores text and numeric similarities. A separate calibration
cohort selects the macro F0.5 threshold, with an independent audit cohort reporting
generalization. No external business lookup or entity augmentation is used.

## 2. Methodology

The six supplied source files contain 24,229,173 rows. Normalization preserves all
original fields and all 1,694,445 France-labelled rows. Unicode transliteration,
casefolding, punctuation handling and whole-token abbreviation expansions are
shared across countries. Numeric runs retain leading zeros. Missing components,
typos and word order remain for fuzzy comparison. Originals are unchanged.

Source 1 hash buckets deterministically divide the training population into 80%
training, 10% calibration and 10% audit. The default capped samples are 12,000,
3,000 and 3,000 entities respectively, seed {metrics['seed']}. The classifier is
not fitted on every training entity. All training references are searchable.
TF-IDF fits on training-cohort query/candidate text only; no calibration or audit
labels influence model fitting. The audit set never selects the cutoff. The saved
split manifest lists every sampled entity and its cohort.

## 3. Candidate Generation

DuckDB stores normalized records and inverted postings on disk. Keys include
significant name words, canonical sorted names, address words, exact addresses,
name-word/address-number combinations, and 12 deterministically smallest hashed
name four-grams. Globally common keys with more than 128 reference occurrences
are omitted. Keys are not country-specific. Weighted overlap retrieves at most
32 candidates. Character 2-4-gram TF-IDF cosine and token-sort similarity on names
and addresses, numeric overlap and a small country-equality bonus rerank these to
at most {metrics['top_k']} candidates. Character retrieval is approximate, not
global exact TF-IDF nearest-neighbor search. Country mismatch never excludes a
candidate. A strong address can compensate for name noise.

The exported candidate lists are exactly the sets scored by the classifier.
No ground-truth positives are injected into the retrieval result. Recall is
measured against every true match, including unretrieved ones; the oracle score
is the maximum macro F0.5 attainable within those lists.

## 4. Matching Model

LightGBM uses 250 trees, learning rate 0.05, 31 leaves, minimum 40 observations per
leaf, L2 regularization 2.0 and deterministic training. Positives and negatives
come from the same final training candidate sets. No neural embeddings or external
pretrained weights are used. LightGBM is MIT licensed.

Features: name/address token Jaccard, token-sort and token-set ratios, normalized
Levenshtein, character TF-IDF cosine, nonempty exact-match flags, length ratios,
field presence, numeric-token Jaccard/conflict, country equality/presence and
retrieval strength. Empty-empty text does not count as a perfect match.

The selected threshold is **{metrics['threshold']:.12g}**. An exact sweep over
distinct calibration probabilities maximizes per-entity macro F0.5, preferring a
higher threshold on ties. Scores are thresholded independently, allowing multiple
matches and singletons. The held-out threshold is retained for test inference.

## 5. Measured Local Results

| Cohort | S1 entities | Macro F0.5 | Blocking pair recall | Candidate oracle F0.5 |
| --- | ---: | ---: | ---: | ---: |
{results}

Training results are in-sample; calibration results were used for cutoff selection.
Only the audit result is the independent local estimate. There is no claimed
leaderboard score. France does not occur in training, so these metrics cannot
establish performance on France.

Largest feature gains:

| Feature | Gain |
| --- | ---: |
{important}

Likely remaining errors include false merges between businesses sharing generic
names or addresses, missed matches after aggressive key pruning, severe spelling
changes, missing components and more true matches than the candidate cap.
These are design limitations, not a claimed manual audit of particular examples.
Detailed measured counts and feature importances are in `artifacts/metrics.json`.

## 6. Submission and Reproduction

The test run covers **{inference['entities']:,}** Source 1 entities, with
**{inference['candidate_pairs']:,}** scored candidate pairs and
**{inference['matched_pairs']:,}** predicted matches. Strict validation passed:
**{validation['passed']}**. It checks complete unique S1 coverage, legal existing
S2/S3 target IDs, duplicate-free lists and the final-match subset requirement.

`README.md` contains installation and exact commands. `src/pipeline.py` coordinates
normalization inputs, blocking, training, inference and validation. Supporting
modules expose shared APIs for each stage. Tests include normalization, exact
macro scoring, threshold sweeping and a synthetic end-to-end integration test.
The final zip contains outputs, runnable code, dependency pins, trained artifacts,
the split manifest and this methodology write-up. Input datasets and rebuildable
disk indexes are excluded from the zip.

No portal upload is performed automatically. The team must fill in its identity
above and upload `output/matching_results.tsv` through the challenge portal.
'''
    (root/'Documentation_template.md').write_text(text,encoding='utf-8')
    return root/'Documentation_template.md'


if __name__=='__main__':
    write_report()
