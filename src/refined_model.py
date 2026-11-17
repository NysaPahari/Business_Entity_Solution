"""Context-aware experimental matcher; mutual evidence must be supplied explicitly."""
import numpy as np
from .features import FEATURE_NAMES
from .model import MatchingModel


class ContextualModel:
    feature_names=FEATURE_NAMES+['mutual_nearest_neighbor']

    def __init__(self,extractor):
        self.extractor=extractor
        self.classifier=MatchingModel(extractor,2026,2).classifier
        self.policy=None

    def fit(self,x,mutual,labels,sample_weight=None):
        self.classifier.fit(self._matrix(x,mutual),labels,sample_weight=sample_weight,
                            feature_name=self.feature_names)
        return self

    @staticmethod
    def _matrix(x,mutual):
        if mutual is None or len(mutual)!=len(x):
            raise ValueError('One label-free mutual-neighbor feature is required for each pair')
        return np.column_stack([x,np.asarray(mutual,dtype=np.float32)])

    def raw_probabilities(self,x,mutual):
        return self.classifier.booster_.predict(self._matrix(x,mutual),num_threads=1)

    def score_rows(self,rows,mutual):
        return self.policy.probabilities(self.raw_probabilities(self.extractor.transform(rows),mutual))

    def predict_rows(self,rows,mutual):
        """Rows must contain complete candidate groups for segment-specific cutoffs."""
        x=self.extractor.transform(rows)
        return self.policy.predict(rows,x,self.raw_probabilities(x,mutual))
