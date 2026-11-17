# Business Entity Resolution: Precision, Coverage, and Reproducibility

**Team name:** EEE_2027_ML

**Team members:** Tanishi Sinha (Team Leader), Palak Agarwal, Nysa Pahari, Roshni Dhar

**Affiliation:** Birla Institute of Technology and Science Pilani, Hyderabad campus

## 1. Executive Summary

We built an entity-resolution pipeline that addresses two central challenges: finding the correct business despite noisy records, and avoiding false merges between businesses that share names or addresses.

Our selected system combines country-independent normalization, composite blocking, 36 complementary similarity features, and a LightGBM classifier. It achieved **0.94548 macro F0.5 on a one-time fresh audit of 3,000 entities**. On the original audit cohort, it improved macro F0.5 from 0.87115 to 0.93874: **a 6.76 percentage-point improvement on the same cohort**.

The strength of our approach lies in its integration. Retrieval preserves several forms of matching evidence; the classifier weighs agreement against contradictions; and a separate calibration process sets the decision threshold for the competition's precision-weighted objective. Every stage is inspectable, and the exported candidate set makes the blocking approach directly auditable.

## 2. Data Preparation and Normalization

The six supplied source files contain **24,229,173 records**. Our shared normalization module applies Unicode normalization, offline transliteration, casefolding, punctuation handling, whitespace normalization, and whole-token expansion of legal and address abbreviations. Original values remain available alongside normalized fields.

Numeric runs are extracted separately and retained in normalized text. We avoid assigning postal-code or building-number meanings from country-specific digit lengths or positions. Leading zeros survive normalization; additional model features can compare numeric values with leading zeros collapsed.

Unresolved typos, missing components, and word-order differences remain available for fuzzy comparison rather than being corrected by rules that might invent information. Missing text never produces a perfect-match feature.

Normalization and blocking have no country-specific branches. This matters for France, which occurs in test but has no labelled training examples. Country mismatch never excludes a candidate.

## 3. Candidate Generation and Blocking

A disk-backed DuckDB inverted index searches the reference universe without constructing an all-pairs comparison matrix. It combines:

- Significant name/address tokens, canonical sorted names, and exact addresses.
- Name-token/numeric combinations and bounded hashed name four-grams.
- Name/address token intersections and address-token pairs.
- Vowel-stripped name-token keys, including combinations with numeric evidence.

Composite keys retain discriminative evidence when individual words are too common. Keys appearing in more than **4,096 reference records** are omitted globally. Inverse-square-root frequency weighting ranks shared-key evidence, retaining up to **128 references per Source 1 business**. The selected inference configuration scores the full retained pool.

Candidate generation does not use truth labels, and labelled positives are never inserted into candidate lists artificially. The exported `candidate_pairs.tsv` contains exactly the candidates passed to the classifier. Retrieval recall includes all true matches, including those blocking fails to find. On the fresh audit, blocking recovered **94.62% of true pairs**.

## 4. Feature Engineering and Matching Model

The classifier uses **36 features** that combine complementary evidence:

| Evidence | Representative features | Purpose |
| --- | --- | --- |
| Names | Character TF-IDF, token-sort/set ratios, edit similarity, suffix-free and partial comparisons | Recognize spelling variation, word reordering, and abbreviated names |
| Addresses | Character/token similarity, exact agreement, partial comparison, text without numbers | Compare incomplete or differently ordered addresses |
| Numbers | Overlap, directional coverage, first/last numeric agreement, count differences | Distinguish otherwise similar records using numeric evidence |
| Context | Field presence, length ratios, country agreement/presence, retrieval strength | Interpret similarity against the available information |

Our LightGBM model uses **600 trees**, up to **63 leaves per tree**, minimum child size **60**, L2 regularization **5**, and seed **2026**. It was trained on **42,000 Source 1 entities and 2,687,935 candidate pairs**. The character TF-IDF representation remains frozen from training-only data. LightGBM is MIT licensed; the selected model contains no external neural weights.

Training used candidate sets of size up to 64. We then held the classifier fixed while comparing larger inference candidate sets on calibration data, selecting K=128 for production. The saved artifact embeds the exact extractor, classifier, and probability threshold.

A calibration threshold sweep maximizes per-entity macro F0.5. Decisions remain independent across candidates: multiple matches are allowed, no one-to-one constraint is imposed, and businesses may correctly receive no match.

## 5. Experimental Protocol and Results

Deterministic entity-level hash buckets separate training, calibration, and audit populations. Training labels fit the model; calibration selects configurations and cutoffs. Audit labels do not fit the classifier or choose its threshold.

The original audit was inspected repeatedly during development and is reported as a diagnostic comparison. After selecting the production configuration, we evaluated it once on a disjoint 3,000-entity fresh audit. That fresh result was not used to revise the selected model.

| Evaluation | Entities | Macro F0.5 | Blocking pair recall |
| --- | ---: | ---: | ---: |
| Baseline, original audit | 3,000 | 0.87115 | 80.93% |
| Selected model, same original audit | 3,000 | 0.93874 | 94.37% |
| Selected model, one-time fresh audit | 3,000 | **0.94548** | **94.62%** |

The fresh-audit entity-bootstrap 95% interval is **[0.94045, 0.95022]**, based on 2,000 resamples. Macro F0.5 averages over Source 1 entities, including singletons. It is not ordinary classification accuracy or a measured leaderboard score.

We also evaluated alternative calibration policies, harder negatives, wider and more diverse retrieval, graph expansion, group-consistency corrections, and a small neural text-pair pilot. These studies did not establish a sufficient gain to replace the selected configuration. We retained the stronger measured model rather than introduce unproven complexity into the submission.

The remaining constraints are explicit. The fresh-audit candidate oracle is **0.98032 macro F0.5**, so unretrieved matches remain a ceiling even for perfect decisions within those lists. Labelled validation cannot establish France performance. Our measurements support a strong local result, but do not establish 0.99 performance or a leaderboard ranking.

## 6. Reproduction, Validation, and Submission

```text
EEE_2027_ML_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/business_entity_resolution/
│   ├── src/
│   ├── artifacts/
│   ├── README.md
│   └── requirements.txt
└── Documentation_template.md
```

The code folder includes the trusted fitted model and source needed to regenerate predictions from the supplied data. The README gives exact setup, normalization, prediction, and validation instructions. Dependencies are pinned. Only the methodology and required README are included as Markdown; experimental caches, logs, and work databases are excluded.

The full test population contains **1,732,544 Source 1 businesses**. Packaging requires complete unique S1 coverage, valid existing S2/S3 target IDs, duplicate-free lists, correct formatting, and containment of every predicted match within its candidate list. The workflow checks model provenance and output hashes, verifies archive integrity, and confirms that packaging preserves the TSV bytes.

All business evidence comes from the supplied dataset. No external directories, registration lookups, geocoding services, or entity-resolution APIs are used. The leaderboard upload must use the exact `matching_results.tsv` contained in this archive. No portal score is assumed or claimed.
