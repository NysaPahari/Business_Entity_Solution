"""Shared fitted TF-IDF and fuzzy pair features; fit on training text only."""
import re
import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

FEATURE_NAMES = [f'{field}_{metric}' for field in ('name','address')
                 for metric in ('jaccard','token_sort','token_set','levenshtein',
                                'tfidf','exact','length_similarity','both_present')]
FEATURE_NAMES += ['numeric_jaccard','numeric_conflict','country_equal',
                  'country_both_present','retrieval_score']


def jaccard(a,b):
    return len(a & b) / len(a | b) if a or b else 0.0


class FeatureExtractor:
    def fit(self, names, addresses):
        self.vectorizers = []
        for texts in (names, addresses):
            v = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4),
                                max_features=50000, dtype=np.float32, lowercase=False)
            # A fixed sentinel handles an entirely empty training field.
            v.fit(list(texts) + ['__empty_field__'])
            self.vectorizers.append(v)
        return self

    def transform(self, rows):
        if not rows:
            return np.empty((0,len(FEATURE_NAMES)),dtype=np.float32)
        features = []
        for index, vectorizer in zip((2,3), self.vectorizers):
            left = [r[index] for r in rows]
            right = [r[index+4] for r in rows]
            present = np.array([bool(a and b) for a,b in zip(left,right)])
            tokens = {t:set(t.split()) for t in set(left + right)}
            features.append(np.array([jaccard(tokens[a],tokens[b]) for a,b in zip(left,right)]))
            for scorer in (fuzz.token_sort_ratio, fuzz.token_set_ratio, Levenshtein.normalized_similarity):
                values = process.cpdist(left,right,scorer=scorer,workers=1,dtype=np.float32)
                if scorer is not Levenshtein.normalized_similarity:
                    values /= 100
                features.append(values * present)
            vocabulary = list(tokens)
            ids = {t:i for i,t in enumerate(vocabulary)}
            matrix = vectorizer.transform(vocabulary)
            a = matrix[[ids[t] for t in left]]
            b = matrix[[ids[t] for t in right]]
            features.append(np.asarray(a.multiply(b).sum(axis=1)).ravel())
            features.append(np.array([a==b for a,b in zip(left,right)]) * present)
            features.append(np.array([min(len(a),len(b))/max(len(a),len(b),1) for a,b in zip(left,right)]) * present)
            features.append(present)
        nums = {r[i]:set(re.findall('[0-9]+',r[i])) for r in rows for i in (3,7)}
        features.append(np.array([jaccard(nums[r[3]],nums[r[7]]) for r in rows]))
        features.append(np.array([bool(nums[r[3]] and nums[r[7]] and not nums[r[3]] & nums[r[7]]) for r in rows]))
        features.append(np.array([bool(r[4] and r[8] and r[4].casefold()==r[8].casefold()) for r in rows]))
        features.append(np.array([bool(r[4] and r[8]) for r in rows]))
        features.append(np.array([r[9] for r in rows]))
        return np.column_stack(features).astype(np.float32)


def final_candidates(rows, features, top_k=20):
    """Label-free reranking before scoring; returns row indices in query order."""
    if not rows:
        return np.empty(0,dtype=np.int64)
    # Union signal: a strong address can rescue a noisy name. Country is only a boost.
    score = (0.52*np.maximum(features[:,4],features[:,1])
             +0.35*np.maximum(features[:,12],features[:,9])
             +0.10*features[:,16]+0.03*features[:,18])
    selected = []
    start = 0
    while start < len(rows):
        end = start+1
        while end < len(rows) and rows[end][0] == rows[start][0]:
            end += 1
        # Stable ties follow candidate entity_id ordering from the database.
        indices = np.argsort(-score[start:end],kind='stable')[:top_k]+start
        selected.extend(sorted(indices))
        start = end
    return np.asarray(selected,dtype=np.int64)
