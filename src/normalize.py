"""Country-independent normalization; originals and numeric leading zeros survive.

Run ``python -m src.normalize`` from the repository root to process all six files.
No fitting, label access, external lookups, spelling correction, or country rules.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from text_unidecode import unidecode

__all__ = ["clean_text", "normalize_name", "normalize_address",
           "extract_numeric_tokens", "normalize_record", "normalize_file",
           "normalize_dataset"]

LEGAL_TERMS = {"corp": "corporation", "pvt": "private", "ltd": "limited",
               "inc": "incorporated"}
ADDRESS_TERMS = {"rd": "road", "st": "street", "apt": "apartment"}
TEXT_FIELDS = ("business_name", "business_address")
FEATURE_FIELDS = tuple(f"{field}_{suffix}" for field in TEXT_FIELDS
                       for suffix in ("normalized", "numeric_tokens"))


def _as_text(value: str | None | float) -> str:
    if value is None or isinstance(value, float) and math.isnan(value):
        return ""
    if not isinstance(value, str):
        raise TypeError("Expected text, None, or float NaN")
    return value


@lru_cache(maxsize=8192)
def _transliterate_char(char: str) -> str:
    # Preserve unsupported letters instead of silently dropping an identity signal.
    result = unidecode(char)
    if not result.strip() or "[?]" in result:
        return char
    return result


def clean_text(value: str | None | float) -> str:
    """Casefold, transliterate, turn punctuation into spaces, collapse whitespace.

    Ampersands become 'and'. Apostrophes within words are removed (O'Reilly ->
    oreilly). Other separators retain token boundaries (12/4 -> 12 4).
    Unknown scripts are retained if transliteration is unavailable.
    """
    text = unicodedata.normalize("NFKC", _as_text(value)).casefold()
    text = re.sub(r"(?<=\w)['\u2019\u2018\u02bc](?=\w)", "", text)
    text = text.replace("&", " and ")
    # Most source text is ASCII: avoid per-character transliteration on that path.
    if text.isascii():
        return re.sub(r"[^a-z0-9]+", " ", text).strip()
    text = "".join(str(unicodedata.decimal(c)) if c.isdecimal()
                   else _transliterate_char(c) for c in text)
    text = unicodedata.normalize("NFKD", text).casefold()
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join("".join(c if c.isalnum() else " " for c in text).split())


def normalize_name(value: str | None | float) -> str:
    """Expand whole-token legal abbreviations; keep suffixes and token order."""
    return " ".join(LEGAL_TERMS.get(t, t) for t in clean_text(value).split())


def normalize_address(value: str | None | float) -> str:
    """Expand whole-token address abbreviations without parsing address layouts.

    'st' is consistently interpreted as 'street'; it can also mean Saint. Keep
    the original address available to downstream features for this ambiguity.
    """
    return " ".join(ADDRESS_TERMS.get(t, t) for t in clean_text(value).split())


def extract_numeric_tokens(value: str | None | float) -> list[str]:
    """Return ordered decimal digit runs, including within alphanumeric tokens.

    Preserve duplicates and leading zeros: '12B / 004, 12' -> ['12','004','12'].
    Unicode decimal digits become ASCII. No guessed postal/house-number roles,
    lengths, or positions; missing text returns an empty list.
    """
    text = unicodedata.normalize("NFKC", _as_text(value))
    if not text.isascii():
        text = "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)
    return re.findall(r"[0-9]+", text)


def normalize_record(record: Mapping[str, str | None]) -> dict:
    """Copy a record and append normalized strings and list[str] numeric features.

    Missing text keys act as empty strings. IDs, countries, and other fields are
    copied unchanged. TSV writers serialize numeric lists as JSON, including [].
    """
    result = dict(record)
    for field, normalizer in zip(TEXT_FIELDS, (normalize_name, normalize_address)):
        value = record.get(field)
        result[f"{field}_normalized"] = normalizer(value)
        result[f"{field}_numeric_tokens"] = extract_numeric_tokens(value)
    return result


def normalize_file(source: str | Path, destination: str | Path | None = None) -> int:
    """Write <stem>_normalized.tsv beside a source; return the number of rows.

    Input must be a UTF-8 TSV with both text columns. Originals cannot be
    overwritten. Use a temporary sibling and atomic replace for completed output.
    """
    source = Path(source)
    destination = Path(destination) if destination else source.with_name(
        f"{source.stem}_normalized.tsv")
    if source.resolve() == destination.resolve():
        raise ValueError("Destination must differ from source")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    count = 0
    try:
        with source.open(encoding="utf-8-sig", newline="") as incoming:
            reader = csv.DictReader(incoming, delimiter="\t")
            fields = reader.fieldnames or []
            if not set(TEXT_FIELDS).issubset(fields):
                raise ValueError(f"Missing required text columns in {source}")
            if len(fields) != len(set(fields)) or set(FEATURE_FIELDS) & set(fields):
                raise ValueError(f"Duplicate or already-normalized columns in {source}")
            with temporary.open("w", encoding="utf-8", newline="") as outgoing:
                writer = csv.DictWriter(outgoing, fieldnames=fields + list(FEATURE_FIELDS),
                                        delimiter="\t", lineterminator="\n")
                writer.writeheader()
                for record in reader:
                    if None in record or any(v is None for v in record.values()):
                        raise ValueError(f"Malformed TSV row near line {reader.line_num}")
                    result = normalize_record(record)
                    for field in TEXT_FIELDS:
                        key = f"{field}_numeric_tokens"
                        result[key] = json.dumps(result[key], separators=(",", ":"))
                    writer.writerow(result)
                    count += 1
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def normalize_dataset(dataset_dir: str | Path = "dataset") -> dict[Path, int]:
    """Process exactly train/test x source1/2/3, never ground truth or outputs."""
    root = Path(dataset_dir)
    sources = [root / split / f"{split}_source{i}.tsv"
               for split in ("train", "test") for i in (1, 2, 3)]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    return {source: normalize_file(source) for source in sources}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    args = parser.parse_args()
    for source, count in normalize_dataset(args.dataset_dir).items():
        print(f"{source.stem}_normalized.tsv: {count:,} rows")


if __name__ == "__main__":
    main()
