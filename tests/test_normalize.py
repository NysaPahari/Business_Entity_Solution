import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.normalize import (clean_text, extract_numeric_tokens, normalize_address,
                           normalize_file, normalize_name, normalize_record)


class NormalizationTests(unittest.TestCase):
    def test_legal_abbreviations_and_ampersand(self):
        self.assertEqual(normalize_name(" ACME, Corp. & Sons Pvt. Ltd. "),
                         "acme corporation and sons private limited")
        self.assertEqual(normalize_name("ACME Corporation and Sons Private Limited"),
                         normalize_name("ACME Corp & Sons Pvt Ltd"))

    def test_unicode(self):
        self.assertEqual(normalize_name("Ｃafé Œuvre Straße"), "cafe oeuvre strasse")
        self.assertEqual(normalize_name("Café"), normalize_name("Cafe\u0301"))
        self.assertEqual(normalize_name("Москва"), "moskva")
        self.assertTrue(normalize_name("मॉडर्न फाइनेंस"))
        self.assertEqual(clean_text("O’Reilly's"), "oreillys")

    def test_address_terms_and_boundaries(self):
        self.assertEqual(normalize_address("12-B Main St., Apt. 004 / Side Rd."),
                         "12 b main street apartment 004 side road")
        self.assertEqual(normalize_name("Corporal Privet Ltdman"),
                         "corporal privet ltdman")
        self.assertEqual(normalize_name("St Clair"), "st clair")
        self.assertEqual(normalize_address("8 Rue de l’Église, 75001"),
                         "8 rue de leglise 75001")

    def test_typos_and_word_order_are_not_invented(self):
        self.assertEqual(normalize_name("Acme Corproation"), "acme corproation")
        first = normalize_name("Blue River Ltd.")
        swapped = normalize_name("River Blue Limited")
        self.assertNotEqual(first, swapped)
        self.assertEqual(sorted(first.split()), sorted(swapped.split()))

    def test_missing_components(self):
        for value in (None, "", " \t ", float("nan")):
            self.assertEqual(normalize_address(value), "")
            self.assertEqual(extract_numeric_tokens(value), [])
        self.assertEqual(normalize_address("Near Library"), "near library")
        self.assertEqual(normalize_record({})["business_name_normalized"], "")

    def test_numeric_features(self):
        self.assertEqual(extract_numeric_tokens("12B/004, 12, ０７５, ١٢٣"),
                         ["12", "004", "12", "075", "123"])
        self.assertEqual(extract_numeric_tokens("AB12 3CD, 75001-0002"),
                         ["12", "3", "75001", "0002"])

    def test_record_is_copied_and_country_independent(self):
        row = {"entity_id": "S1-001", "business_name": "Studio 007 Ltd.",
               "business_address": "12 Main Rd", "country": "Unseen"}
        original = row.copy()
        result = normalize_record(row)
        self.assertEqual(row, original)
        self.assertEqual(result["country"], "Unseen")
        self.assertEqual(result["business_name_numeric_tokens"], ["007"])
        other = normalize_record({**row, "country": "France"})
        self.assertEqual(result["business_address_normalized"],
                         other["business_address_normalized"])

    def test_idempotence(self):
        for value in ("Café & Co., Pvt Ltd", "12 Main St., Apt 2", "Москва", "École"):
            for function in (clean_text, normalize_name, normalize_address):
                self.assertEqual(function(function(value)), function(value))

    def test_tsv_roundtrip_preserves_originals(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.tsv"
            rows = [{"entity_id": "S1-001", "business_name": "Café\tLtd",
                     "business_address": "12 Rd\nApt 003", "country": "France"},
                    {"entity_id": "S1-002", "business_name": "", "business_address": "",
                     "country": ""}]
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            before = source.read_bytes()
            self.assertEqual(normalize_file(source), 2)
            output = source.with_name("source_normalized.tsv")
            with output.open(encoding="utf-8", newline="") as stream:
                actual = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual([{k: r[k] for k in rows[0]} for r in actual], rows)
            self.assertEqual(json.loads(actual[0]["business_address_numeric_tokens"]),
                             ["12", "003"])
            self.assertEqual(actual[1]["business_address_numeric_tokens"], "[]")
            first_output = output.read_bytes()
            normalize_file(source)
            self.assertEqual(output.read_bytes(), first_output)
            self.assertEqual(source.read_bytes(), before)
            with self.assertRaises(ValueError):
                normalize_file(source, source)

    def test_malformed_input_does_not_replace_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.tsv"
            output = Path(folder) / "source_normalized.tsv"
            output.write_text("existing output", encoding="utf-8")
            for bad in ("wrong\theader\n", "business_name\tbusiness_address\nOnlyName\n"):
                source.write_text(bad, encoding="utf-8")
                with self.assertRaises(ValueError):
                    normalize_file(source)
                self.assertEqual(output.read_text(encoding="utf-8"), "existing output")


if __name__ == "__main__":
    unittest.main()
