import unittest
import numpy as np
from src.accuracy_features import extra_features, EXTRA_NAMES
from src.blocking import key_sql
from src.data import connect


class AccuracyTests(unittest.TestCase):
    def test_missing_fields_not_perfect_numeric_or_name_matches(self):
        row=(1,'s','','','','c','','','',0)
        self.assertTrue(np.all(extra_features([row])==0))
        self.assertEqual(extra_features([]).shape,(0,len(EXTRA_NAMES)))

    def test_leading_zeros_and_legal_suffixes(self):
        row=(1,'s','acme limited','0012 road 009','France','c','acme corporation','12 road 9','France',0)
        x=extra_features([row])[0]
        self.assertEqual(x[0],1)
        self.assertEqual(x[5],1)
        self.assertEqual(x[8],1)
        self.assertEqual(x[9],1)

    def test_composite_keys_preserve_baseline_and_ignore_country(self):
        con=connect(':memory:','128MB',1)
        try:
            con.execute("CREATE TABLE records(rid INTEGER,source INTEGER,name VARCHAR,address VARCHAR)")
            con.execute("INSERT INTO records VALUES (1,1,'acme private limited','12 rue jardin 009')")
            baseline={r[2] for r in con.execute(key_sql()).fetchall()}
            expanded={r[2] for r in con.execute(key_sql('composite')).fetchall()}
            self.assertTrue(baseline < expanded)
            bounded={r[2] for r in con.execute(key_sql('composite',(1,1))).fetchall()}
            self.assertEqual(expanded,bounded)
            self.assertEqual(con.execute(key_sql('composite',(2,3))).fetchall(),[])
        finally: con.close()


if __name__=='__main__': unittest.main()
