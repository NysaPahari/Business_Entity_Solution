# Normalization handoff (Owner A)

From the repository root, using Python 3.10+:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.normalize --dataset-dir dataset
```

The last command generates `train_source{1,2,3}_normalized.tsv` in
`dataset/train/` and `test_source{1,2,3}_normalized.tsv` in `dataset/test/`.
It processes exactly six sources, excluding ground truth and generated outputs.
Source columns, IDs, country strings, row order, and row counts are preserved.
Rerunning deterministically replaces generated files; source files are never modified.

## Shared API

```python
from src.normalize import normalize_name, normalize_address, normalize_record
from src.normalize import clean_text, extract_numeric_tokens

normalize_name("Café & Sons Pvt. Ltd.")  # 'cafe and sons private limited'
normalize_address("12B Main St., Apt 004")  # '12b main street apartment 004'
extract_numeric_tokens("12B Main St., Apt 004")  # ['12', '004']
features = normalize_record({"business_name": "Café Ltd", "business_address": "12 Rd"})
```

`normalize_record` returns a new dictionary containing original fields plus:

| Column | Python value | TSV encoding |
| --- | --- | --- |
| business_name_normalized | str | text |
| business_name_numeric_tokens | list[str] | JSON array |
| business_address_normalized | str | text |
| business_address_numeric_tokens | list[str] | JSON array |

Read TSVs with an explicit tab delimiter. Use `json.loads` on numeric feature
columns, not comma splitting. If using pandas, use `dtype=str, keep_default_na=False`
to preserve empty values and literal strings such as `NA`.

Normalization uses Unicode NFKC, casefolding, offline `text-unidecode`
transliteration, accent removal, and collapsed whitespace. Unsupported characters
are retained when possible. It is lossy and cannot reconcile all transliteration
spellings. Original text remains available for alternative features.
Whole-token substitutions expand corp/pvt/ltd/inc in names and rd/st/apt in
addresses. Ampersands become 'and'; internal apostrophes disappear; other
punctuation separates tokens. Legal suffixes are retained.

Numeric features contain every decimal digit run in order, including digits in
alphanumeric tokens, leading zeros, and duplicates. Numbers also remain in
normalized text. No country, digit-length, or position assumptions assign a
number to a postal-code or house-number role: those roles are ambiguous without
additional evidence. Downstream overlap features can compare these lists as sets
or multisets while retaining the raw address for richer parsing.

Typos, missing components, and word-order swaps are deliberately retained;
normalization does not invent spelling corrections or absent data. Owners B/C
can use token-set/sort and fuzzy comparisons. `st` is ambiguous (street/Saint);
the fixed expansion is a matching convention, not an address parser. There are
no country branches, fitted vocabulary, training labels, or identity lookups.
`None`, empty text, and float NaN map to empty text and empty numeric lists;
other non-string inputs raise TypeError.

## Verification of the supplied dataset

All 10 unit tests passed. Generated files contain:

| Output | Rows |
| --- | ---: |
| dataset/train/train_source1_normalized.tsv | 2,206,821 |
| dataset/train/train_source2_normalized.tsv | 5,034,616 |
| dataset/train/train_source3_normalized.tsv | 5,285,603 |
| dataset/test/test_source1_normalized.tsv | 1,732,544 |
| dataset/test/test_source2_normalized.tsv | 4,887,273 |
| dataset/test/test_source3_normalized.tsv | 5,082,316 |
| Total | 24,229,173 |

Streaming verification compared every original field and row position against
the corresponding output and decoded every numeric JSON array. SHA-256 checks
confirmed that all six originals were unchanged. All 1,694,445 rows labelled
France were preserved. Normalized values were also checked for reproducibility
and idempotence on every 1,000th row in each file.
