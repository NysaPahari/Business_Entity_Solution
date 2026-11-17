import ast
import unittest
from src.clean_submission import clean_source


class CleanupTests(unittest.TestCase):
    def test_preserves_code_and_literal_hash_and_sql(self):
        original='''"""Module description.
More documentation.
Third line.
"""
from __future__ import annotations
# first
# second
# third
def value():
    """Function description.
    More explanation.
    Third line.
    """
    sql = "SELECT '#literal'"  # comment
    return sql
'''
        result=clean_source(original)
        scope={}; exec(result,scope)
        self.assertEqual(scope['value'](),"SELECT '#literal'")
        self.assertEqual(ast.get_docstring(ast.parse(result)),'Module description.')
        self.assertNotIn('# third',result)
