"""Small reproducible LightGBM model with a reusable pair-scoring interface."""
import joblib
import numpy as np
from lightgbm import LGBMClassifier
from .features import FEATURE_NAMES


class MatchingModel:
    def __init__(self, extractor, seed=2026, threads=6):
        self.extractor = extractor
        self.classifier = LGBMClassifier(n_estimators=250, learning_rate=0.05,
            num_leaves=31, max_depth=-1, min_child_samples=40, reg_lambda=2.0,
            colsample_bytree=0.9, random_state=seed, n_jobs=threads, verbosity=-1,
            deterministic=True, force_col_wise=True)
        self.threshold = 0.5

    def fit(self, features, labels):
        if len(set(labels)) != 2:
            raise ValueError('Training candidates must contain both positive and negative pairs')
        self.classifier.fit(features, np.asarray(labels,dtype=np.uint8), feature_name=FEATURE_NAMES)
        return self

    def probabilities(self, features):
        if len(features)==0:
            return np.empty(0,dtype=np.float64)
        return self.classifier.booster_.predict(features,num_threads=1)

    def score(self, source1_id, candidate_id, records):
        """Score one pair from an ID -> normalized record mapping.

        Retrieval_score is optional record metadata; production scoring uses the
        batch probabilities API with exactly the retained candidate features.
        """
        a,b = records[source1_id],records[candidate_id]
        row = (0,source1_id,a['business_name_normalized'],a['business_address_normalized'],a.get('country',''),
               candidate_id,b['business_name_normalized'],b['business_address_normalized'],b.get('country',''),
               b.get('retrieval_score',0.0))
        return float(self.probabilities(self.extractor.transform([row]))[0])

    def save(self, path):
        joblib.dump(self,path,compress=3)

    @classmethod
    def load(cls,path):
        # Only load locally generated/trusted joblib files.
        result = joblib.load(path)
        if not isinstance(result,cls):
            raise TypeError('Not a MatchingModel artifact')
        return result

    def feature_importance(self):
        values = self.classifier.booster_.feature_importance(importance_type='gain')
        names = self.classifier.booster_.feature_name()
        return dict(sorted(zip(names,map(float,values)),key=lambda item:-item[1]))
