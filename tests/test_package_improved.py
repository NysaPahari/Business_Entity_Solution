import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile
from src.finalize_improved import digest
from src.package_improved import package


class ImprovedPackageTests(unittest.TestCase):
    def test_verified_layout_and_changed_output_rejected(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                paths = ['src/pipeline.py','src/normalize.py','src/push_accuracy.py','requirements.txt','NORMALIZATION.md','REFINEMENT.md',
                         'run_improved_predictions.ps1','SUBMISSION_README.md','Documentation_improved.md',
                         'artifacts/accuracy/selected/model.joblib','artifacts/accuracy/chosen.joblib',
                         'artifacts/accuracy/selected/inference_config.json','artifacts/accuracy/selected/validation_metrics.json',
                         'artifacts/accuracy/ACCURACY_REPORT.md','artifacts/accuracy/fresh_audit.json',
                         'artifacts/accuracy/results.json','artifacts/refinement/ablation_results.json',
                         'output_improved/matching_results.tsv','output_improved/candidate_pairs.tsv',
                         'output_improved/inference_metrics.json']
                for name in paths:
                    path = Path(name); path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_text('fixture',encoding='utf-8')
                Path('artifacts/accuracy/selection.json').write_text(json.dumps({'selected_stage':'chosen'}))
                output = Path('output_improved')
                (output/'validation.json').write_text(json.dumps({'passed':True,'errors':[]}))
                info = dict(validation_passed=True,inference={'entities':1732544},
                            model_sha256=digest('artifacts/accuracy/selected/model.joblib'),files={})
                for name in ('matching_results.tsv','candidate_pairs.tsv'):
                    info['files'][name] = {'sha256':digest(output/name)}
                (output/'run_info.json').write_text(json.dumps(info))
                target = package()
                with zipfile.ZipFile(target) as archive:
                    self.assertIn('output/matching_results.tsv',archive.namelist())
                    self.assertIn('output/candidate_pairs.tsv',archive.namelist())
                    self.assertIn('Documentation_template.md',archive.namelist())
                    self.assertIn('code/business_entity_resolution/src/pipeline.py',archive.namelist())
                    self.assertIn('code/business_entity_resolution/artifacts/model.joblib',archive.namelist())
                    self.assertIn('code/business_entity_resolution/artifacts/inference_config.json',archive.namelist())
                    self.assertFalse(any('/accuracy/' in name for name in archive.namelist()))
                    self.assertNotIn(b'--artifact-dir artifacts/accuracy',archive.read('code/business_entity_resolution/README.md'))
                    self.assertIn(b'Reproduce the supplied outputs',archive.read('code/business_entity_resolution/README.md'))
                (output/'matching_results.tsv').write_text('changed')
                with self.assertRaisesRegex(ValueError,'Output changed'):
                    package()
                (output/'validation.json').write_text(json.dumps({'passed':False}))
                with self.assertRaisesRegex(ValueError,'Validated improved'):
                    package()
            finally:
                os.chdir(previous)
