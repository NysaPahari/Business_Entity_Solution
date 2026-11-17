import unittest
from unittest.mock import patch
import numpy as np
from src.evaluate import macro_f05
from src.refinement import DecisionPolicy,rule_masks,threshold_with_overrides,calibration_role
from src.error_analysis import summarize
from src.mutual import mutual_features
from src.mutual import _reverse_batch
from src.refined_model import ContextualModel


class RefinementTests(unittest.TestCase):
    def test_exact_rule_requires_present_fields(self):
        x=np.zeros((3,21)); x[1,5]=x[1,13]=1; x[2,5]=1
        accept,reject=rule_masks(x,True)
        self.assertEqual(accept.tolist(),[False,True,False])
        self.assertTrue(reject[0])

    def test_override_threshold_matches_brute_force(self):
        truth={'a':{'x','y'},'b':set(),'c':{'z'},'empty':set()}
        rows=[(0,'a','','','','x'),(0,'a','','','','y'),(0,'b','','','','p'),(0,'c','','','','z')]
        scores=np.array([.1,.8,.5,.9]); accept=np.array([True,False,False,False]); reject=np.array([False,False,False,True])
        cutoff,actual=threshold_with_overrides(truth,rows,scores,accept,reject)
        def score(t):
            predictions={s:set() for s in truth}
            for r,p,a,b in zip(rows,scores,accept,reject):
                if a or (not b and p>=t):
                    predictions[r[1]].add(r[5])
            return macro_f05(truth,predictions)
        self.assertAlmostEqual(actual,max(score(t) for t in [1.00001,.9,.8,.5,.1]))
        self.assertAlmostEqual(actual,score(cutoff))

    def test_entity_calibration_partition_is_stable(self):
        self.assertEqual(calibration_role('S1-123'),calibration_role('S1-123'))
        self.assertEqual({calibration_role(f'S1-{i}') for i in range(100)}, {'isotonic_fit','threshold_tune'})

    def test_missing_candidates_are_counted_as_false_negatives(self):
        d={'rows':[(1,'s','name','addr','X','c','name','addr','X',1)],
           'queries':{'s':{'cohort':'audit','country':'X'},'zero':{'cohort':'audit','country':'Y'}},
           'truth':{'s':{'c','lost'},'zero':{'missing'}},'threshold':.5}
        report,errors,_=summarize(d,[0],[.9])
        aggregate=next(s for s in report['segments'] if s['dimension']=='all')
        self.assertEqual((aggregate['fn'],aggregate['blocked_fn']),(2,2))
        self.assertEqual(report['blocking']['pair_recall'],1/3)
        self.assertTrue(all(e['probability'] is None for e in errors))

    def test_mutual_nearest_preserves_ties_and_requires_both_directions(self):
        class Ranker:
            def probabilities(self,x):
                return x[:,0]
        d={'rows':[(1,'s','','','','c1'),(1,'s','','','','c2'),(1,'s','','','','c3')],
           'queries':{'s':{'cohort':'audit'}},'selected':np.array([0,1,2]),
           'x':np.array([[.9],[.9],[.8]],dtype=np.float32)}
        reverse={'c1':[('s',)]*4,'c2':[('s','other')]*4,'c3':[('s',)]*4}
        self.assertEqual(mutual_features(d,[Ranker()]*4,reverse).tolist(),[1,1,0])
        self.assertEqual(mutual_features(d,[Ranker()]*4,{}).tolist(),[0,0,0])

    def test_reverse_scoring_uses_only_required_ranker_and_s1_orientation(self):
        class Extractor:
            def transform(self,rows):
                return np.array([[float(r[2])] for r in rows])
        class Ranker:
            extractor=Extractor()
            def __init__(self,active): self.active=active
            def probabilities(self,x):
                if not self.active: raise AssertionError('Unneeded ranker was evaluated')
                return 1-x[:,0]
        rows=[(1,'S2-c','ref','addr','X','S1-a','0.1','addr','X',1),
              (1,'S2-c','ref','addr','X','S1-b','0.9','addr','X',1)]
        with patch('src.mutual._models',[Ranker(False),Ranker(True),Ranker(False),Ranker(False)]):
            through,winners=_reverse_batch(rows,[2,2])
        self.assertEqual(through,1)
        self.assertEqual(winners['S2-c'],[(),('S1-a',),(),()])

    def test_contextual_model_refuses_missing_mutual_feature(self):
        with self.assertRaises(ValueError):
            ContextualModel._matrix(np.zeros((2,21)),None)


if __name__=='__main__':
    unittest.main()
