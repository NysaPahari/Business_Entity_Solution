import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.evaluate import entity_f05,macro_f05,select_threshold,predictions_at
from src.features import FeatureExtractor,final_candidates
from src.normalize import normalize_dataset
from src.data import prepare,connect
from src.blocking import build_index,generate_pool
from src.train import train
from src.infer import infer
from src.submission import validate_streaming


class MetricsTests(unittest.TestCase):
    def test_exact_metric_and_singletons(self):
        self.assertEqual(entity_f05(set(),set()),1)
        self.assertEqual(entity_f05(set(),{'x'}),0)
        self.assertEqual(entity_f05({'x'},set()),0)
        self.assertAlmostEqual(entity_f05({'a','b','c'},{'a','b','d'}),2/3)
        self.assertAlmostEqual(macro_f05({'a':set(),'b':{'x'}},{'b':{'x','y'}}),(1+1.25/2.25)/2)

    def test_threshold_sweep_matches_brute_force(self):
        truth={'a':{'x','y'},'b':set(),'c':{'z'},'d':set()}
        rows=[(0,'a','','','','x'),(0,'a','','','','n'),(0,'b','','','','p'),(0,'c','','','','z')]
        scores=np.array([.8,.5,.5,.3])
        threshold,score=select_threshold(truth,rows,scores)
        brute=max(macro_f05(truth,predictions_at(rows,scores,t)) for t in [1.000001,.8,.5,.3])
        self.assertAlmostEqual(score,brute)
        self.assertAlmostEqual(score,macro_f05(truth,predictions_at(rows,scores,threshold)))

    def test_empty_fields_not_perfect_matches(self):
        extractor=FeatureExtractor().fit(['alpha'],['main street'])
        x=extractor.transform([(1,'s','','','','c','','','',0)])
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue((x==0).all())


class IntegrationTest(unittest.TestCase):
    def test_full_pipeline_and_strict_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data=root/'dataset'
            for split,count in [('train',120),('test',12)]:
                folder=data/split
                folder.mkdir(parents=True)
                files=[]
                for source in (1,2,3):
                    stream=(folder/f'{split}_source{source}.tsv').open('w',encoding='utf-8',newline='')
                    writer=csv.writer(stream,delimiter='\t',lineterminator='\n')
                    writer.writerow(['entity_id','business_name','business_address','country'])
                    for i in range(count):
                        name=f'Vendor {i} Blue Orchid Ltd'
                        address=f'{i+100} Maple Rd Apt {i%7}'
                        if source>1 and i%5==0:
                            name=f'Other Galaxy {i}'
                            address=f'{i+900} Cedar St'
                        if source==3:
                            name=name.replace('Ltd','Limited')
                        if split=='test' and source==1 and i==0:
                            name,address='',''
                        writer.writerow([f'S{source}-{i}',name,address,'France' if split=='test' else 'X'])
                    stream.close()
                if split=='train':
                    with (folder/'train_ground_truth.tsv').open('w',encoding='utf-8',newline='') as f:
                        writer=csv.writer(f,delimiter='\t',lineterminator='\n')
                        writer.writerow(['source1_entity_id','matched_entity_ids'])
                        writer.writerows((f'S1-{i}',f'S2-{i},S3-{i}' if i%5 else '') for i in range(count))
            normalize_dataset(data)
            for split in ('train','test'):
                db=prepare(data,split,root/'work',80,20,20,memory='256MB',threads=1)
                build_index(db,max_df=128,memory='256MB',threads=1)
                generate_pool(db,pool_size=12,memory='256MB',threads=1)
                if split=='train':
                    report=train(db,data,root/'artifacts',top_k=8,threads=1,memory='256MB')
                    self.assertIn('audit',report)
                    self.assertGreater(report['audit']['blocking']['pair_recall'],0.9)
                else:
                    result=infer(db,root/'artifacts/model.joblib',root/'output',top_k=8,memory='256MB',threads=1,workers=2)
                    self.assertEqual(result['entities'],12)
                    with (root/'output/candidate_pairs.tsv').open(encoding='utf-8',newline='') as f:
                        candidate_map={r['source1_entity_id']:r['candidate_entity_ids']
                                       for r in csv.DictReader(f,delimiter='\t')}
                    self.assertEqual(candidate_map['S1-0'],'')
            errors,_=validate_streaming(root/'output/matching_results.tsv',root/'output/candidate_pairs.tsv',data/'test',memory='256MB')
            self.assertEqual(errors,[])
            # A nonexistent candidate must fail even if its source prefix is legal.
            p=root/'output/candidate_pairs.tsv'
            lines=p.read_text().splitlines()
            lines[1]=lines[1]+',S2-doesnotexist' if not lines[1].endswith('\t') else lines[1]+'S2-doesnotexist'
            p.write_text('\n'.join(lines)+'\n')
            errors,_=validate_streaming(root/'output/matching_results.tsv',p,data/'test',memory='256MB')
            self.assertTrue(errors)


if __name__=='__main__':
    unittest.main()
