import unittest
from src.graph_probe import choose_anchors


class AnchorTests(unittest.TestCase):
    def test_low_confidence_abstention_ties_and_query_boundaries(self):
        rows=[(1,5,.99),(1,3,.99),(1,8,.97),(2,4,.5),(3,9,.98)]
        expected=[(1,3,.99),(3,9,.98)]
        self.assertEqual(choose_anchors(rows,limit=1),expected)
        self.assertEqual(choose_anchors(list(reversed(rows)),limit=1),expected)
