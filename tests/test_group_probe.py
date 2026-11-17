import unittest
import numpy as np
from src.group_probe import group_features


class GroupFeatureTests(unittest.TestCase):
    def test_excludes_self_other_queries_and_missing_text_agreement(self):
        def row(qid,cid,name='',address=''):
            return (qid,str(qid),'','','',cid,name,address,'',0.)
        rows=[row(1,'a','alpha','road'),row(1,'b','alpha','road'),
              row(2,'c','alpha','road'),row(3,'d'),row(3,'e')]
        x=group_features(rows,np.array([.99,.5,.99,.99,.5]))
        self.assertEqual(x.shape,(5,7))
        np.testing.assert_array_equal(x[0],np.zeros(7))
        np.testing.assert_array_equal(x[1],np.ones(7))
        np.testing.assert_array_equal(x[2],np.zeros(7))
        np.testing.assert_array_equal(x[4],[1,0,0,0,0,0,0])
