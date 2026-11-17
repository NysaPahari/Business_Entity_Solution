import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from src.data import connect
from src.infer import infer


class InferenceResumeTests(unittest.TestCase):
    def test_resume_truncates_uncommitted_tail_and_recovers_partial_publish(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); db=root/'test.duckdb'; output=root/'out'
            model=root/'model.joblib'; model.write_bytes(b'trusted-test-model')
            con=connect(db,'128MB',1)
            con.execute("CREATE TABLE queries AS SELECT i::BIGINT qid,'S1-'||i entity_id FROM range(1,1006) t(i)")
            con.close()
            def interrupted(*args):
                yield 1000,{1:['S2-1']},{1:['S2-1']}
                raise RuntimeError('simulated interruption')
            kwargs=dict(top_k=128,memory='128MB',threads=1,workers=1)
            with patch('src.infer.MatchingModel.load',return_value=SimpleNamespace(threshold=.5)):
                with patch('src.infer.scored_batches',side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError,'simulated'):
                        infer(db,model,output,**kwargs)
                for name in ('candidate_pairs.tsv.tmp','matching_results.tsv.tmp'):
                    with (output/name).open('ab') as stream: stream.write(b'uncommitted-garbage\n')
                model.write_bytes(b'changed-model')
                with self.assertRaisesRegex(ValueError,'inputs changed'):
                    infer(db,model,output,**kwargs)
                model.write_bytes(b'trusted-test-model')
                original_replace=Path.replace
                def partial_publish(path,target):
                    if path.name=='matching_results.tsv.tmp': raise RuntimeError('publish interruption')
                    return original_replace(path,target)
                with patch('src.infer.scored_batches',return_value=iter([(1005,{1005:['S3-last']},{})])):
                    with patch.object(Path,'replace',partial_publish):
                        with self.assertRaisesRegex(RuntimeError,'publish interruption'):
                            infer(db,model,output,**kwargs)
                with patch('src.infer.scored_batches',side_effect=AssertionError('completed rows must not be rescored')):
                    counts=infer(db,model,output,**kwargs)
            self.assertEqual(counts['entities'],1005)
            self.assertEqual(counts['candidate_pairs'],2)
            self.assertEqual(counts['matched_pairs'],1)
            self.assertFalse((output/'inference_checkpoint.json').exists())
            with (output/'matching_results.tsv').open(newline='') as stream:
                rows=list(csv.DictReader(stream,delimiter='\t'))
            self.assertEqual(len(rows),1005)
            self.assertEqual(len({r['source1_entity_id'] for r in rows}),1005)
            self.assertEqual(rows[0]['matched_entity_ids'],'S2-1')
            self.assertEqual(rows[-1]['matched_entity_ids'],'')


if __name__=='__main__': unittest.main()
