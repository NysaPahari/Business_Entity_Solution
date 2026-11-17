import json
from pathlib import Path
import tempfile
import unittest
from src.team_identity import load_identity, methodology_identity


class IdentityTests(unittest.TestCase):
    def test_only_supplied_identity_and_safe_filename(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'team.json'
            self.assertIsNone(load_identity(path))
            path.write_text(json.dumps({'team_name':'Example Team','members':['A','B']}))
            identity=load_identity(path)
            text=methodology_identity('Team name and members: to be filled in by the submitting team.',identity)
            self.assertIn('Example Team',text)
            self.assertIn('A, B',text)
            self.assertNotIn('Affiliation',text)
            self.assertEqual(methodology_identity(text,identity),text)
            path.write_text(json.dumps({'team_name':'../escape'}))
            with self.assertRaises(ValueError): load_identity(path)
