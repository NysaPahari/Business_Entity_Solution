"""Dense per-field encoders and FAISS ANN retrieval for candidate recall.

Fields are embedded independently (name, address), so a strong field retrieves a
match even when the other is empty or noisy. Encoders are fitted on training-split
text only; the same fitted encoder object is reused at inference time.
"""
from pathlib import Path
import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from .blocking import NAME_STOP


def name_text(name):
    # Same generic legal/stop-word list as blocking; keeps distinctive tokens.
    core = ' '.join(t for t in name.split() if t not in NAME_STOP)
    return core or name


def address_text(address):
    return address


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.divide(x, norms, out=x, where=norms > 0)
    return x


class CharEncoder:
    """Character 2-4-gram TF-IDF (hashed) projected to a dense space by SVD."""

    def __init__(self, dim=128, n_features=2**20, seed=2026):
        self.hasher = HashingVectorizer(analyzer='char_wb', ngram_range=(2, 4), n_features=n_features,
                                        alternate_sign=False, norm=None, lowercase=False)
        self.tfidf = TfidfTransformer(sublinear_tf=True)
        self.svd = TruncatedSVD(dim, algorithm='randomized', n_iter=4, random_state=seed)

    def fit(self, texts):
        counts = self.hasher.transform(texts)
        self.svd.fit(self.tfidf.fit_transform(counts))
        return self

    def encode(self, texts):
        return normalize_rows(self.svd.transform(self.tfidf.transform(self.hasher.transform(texts))))


class Model2VecEncoder:
    """Static distilled sentence-embedding model (model2vec)."""

    def __init__(self, model='minishlab/potion-base-8M'):
        self.model_name = model
        self.model = None

    def _load(self):
        if self.model is None:
            from model2vec import StaticModel
            self.model = StaticModel.from_pretrained(self.model_name)
        return self.model

    def fit(self, texts):
        self._load()
        return self

    def encode(self, texts):
        return normalize_rows(self._load().encode(list(texts), batch_size=4096))

    def __getstate__(self):
        return dict(model_name=self.model_name, model=None)


def build_ivfpq(vectors, path, nlist=4096, m=32, train_size=400000, seed=2026, threads=4):
    """Approximate IVF-PQ inner-product index; memory ~m bytes per vector."""
    import faiss
    faiss.omp_set_num_threads(threads)
    dim = vectors.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(seed)
    sample = np.ascontiguousarray(vectors[np.sort(rng.choice(len(vectors), min(train_size, len(vectors)), replace=False))], dtype=np.float32)
    index.train(sample)
    for lo in range(0, len(vectors), 500000):
        index.add(np.ascontiguousarray(vectors[lo:lo + 500000], dtype=np.float32))
    faiss.write_index(index, str(path))
    return index


def search(index, queries, k=32, nprobe=32, threads=4, batch=20000):
    import faiss
    faiss.omp_set_num_threads(threads)
    index.nprobe = nprobe
    sims, ids = [], []
    for lo in range(0, len(queries), batch):
        s, i = index.search(np.ascontiguousarray(queries[lo:lo + batch], dtype=np.float32), k)
        sims.append(s); ids.append(i)
    return np.concatenate(sims), np.concatenate(ids)


def save_encoder(encoder, path):
    joblib.dump(encoder, path)


def load_encoder(path):
    # Only load locally generated/trusted joblib files.
    return joblib.load(path)
