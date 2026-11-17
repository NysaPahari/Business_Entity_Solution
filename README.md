# Business Entity Resolution

**Amazon ML Challenge 2026 · Team EEE_2027_ML · BITS Pilani, Hyderabad Campus**

We built a record-linkage pipeline that identifies the same business across three independent sources. The selected model scored **0.94548 macro F₀.₅** on a fresh local audit of 3,000 businesses. On the original audit cohort, the project improved from **0.87115 to 0.93874**, a gain of **6.76 percentage points**. The pipeline generated validated output for all **1,732,544** Source 1 test businesses.

## Problem statement

Business records from different sources have no shared identifier. Names can contain abbreviations, typos, reordered words, or different transliterations. Addresses can be incomplete or follow different formatting conventions. Distinct businesses can also share a name or address. For each business in the deduplicated Source 1 list, the challenge asks for every matching record in Sources 2 and 3, including an empty result where appropriate.

The scoring metric averages F₀.₅ equally across Source 1 businesses. It favors precision, and an entity without true matches receives full credit when its predicted list is empty. India and the US appear in labeled training data. France appears only in the test set.

## Dataset

| Source | Training records | Test records |
| --- | ---: | ---: |
| Source 1 | 2,206,821 | 1,732,544 |
| Source 2 | 5,034,616 | 4,887,273 |
| Source 3 | 5,285,603 | 5,082,316 |
| **Total** | **12,527,040** | **11,702,133** |

The six source files contain **24,229,173 records**. Exhaustive test matching would require about **17.27 trillion comparisons**. Candidate retrieval reduced that to **221.74 million scored pairs**, a **99.9987% reduction**. The test set contains 259,452 French Source 1 entities.

## Approach

1. **Normalize names and addresses.** Apply Unicode normalization, offline transliteration, case folding, punctuation cleanup, and whole-word abbreviation expansion. Retain original fields and extract numeric sequences, including leading zeros. Use the same rules across countries.
2. **Retrieve plausible candidates.** A disk-backed DuckDB inverted index combines name and address tokens, sorted names, exact addresses, character four-grams, spelling variants, and composite keys with numeric evidence. Overly common keys are pruned, rarer shared keys receive more weight, and up to 128 references are retained per query. Country mismatch never excludes a candidate. Ground-truth matches are never injected into candidate lists.
3. **Score candidates.** A 600-tree LightGBM model uses 36 features covering character TF-IDF, fuzzy name and address similarity, partial matches, numeric agreement, missing fields, country agreement, and retrieval strength. The final fit used 42,000 labeled Source 1 entities and 2,687,935 candidate pairs.
4. **Calibrate and validate.** Deterministic entity hashing separates model fitting, calibration, and audit. Calibration selected a probability threshold of 0.64249. Each pair is accepted independently, so a query can return zero, one, or many matches. The pipeline exports the exact candidate lists alongside the final matches.

## Evaluation

| Evaluation | Entities | Macro F₀.₅ | Blocking recall | Candidate oracle F₀.₅ |
| --- | ---: | ---: | ---: | ---: |
| Baseline, original audit | 3,000 | 0.87115 | 80.93% | 0.91430 |
| Selected model, same original audit | 3,000 | 0.93874 | 94.37% | 0.97992 |
| **Selected model, one-time fresh audit** | **3,000** | **0.94548** | **94.62%** | **0.98032** |

The team inspected the original audit during development and used it to diagnose errors. The fresh audit was disjoint and evaluated once after the configuration was frozen. Its 95% bootstrap interval was **0.94045–0.95022**. The model produced 9,189 true positives and 194 false positives. Of 1,151 false negatives, 556 never entered the candidate lists and 595 were rejected by the classifier.

Composite retrieval addressed the largest source of baseline errors. Extra similarity features and more labeled training entities improved matching after retrieval. A graph expansion pilot found more true candidates but yielded only a small gain in end-to-end score. A small neural text matcher scored below the frozen LightGBM model on the same calibration subset. Neither pilot replaced the selected configuration.

The **0.99 project target remains unmet**. The candidate oracle shows that perfect classification of the pairs retrieved on the fresh audit would reach 0.98032. The audit measured US and Indian performance. French matching quality and the hidden-test leaderboard score remain unverified.

## Test inference and submission

| Statistic | Count |
| --- | ---: |
| Source 1 entities processed | 1,732,544 |
| Candidate pairs scored | 221,741,951 |
| Predicted matching pairs | 5,593,887 |
| Entities with at least one predicted match | 1,624,665 |
| Entities with an empty predicted list | 107,879 |

The project exports `matching_results.tsv` and `candidate_pairs.tsv`. Both passed checks for complete Source 1 coverage, valid target IDs, duplicate-free lists, TSV formatting, and inclusion of each predicted match in its candidate list. The submission package includes the fitted model, Python source, pinned dependencies, and reproduction instructions. The pipeline uses the supplied records throughout, with no external business lookups.
