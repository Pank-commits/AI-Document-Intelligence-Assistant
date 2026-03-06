import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import re

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
TOKEN_PATTERN = re.compile(r"[A-Za-z]+|\d+\.\d+|\d+%?|[A-Z][a-z]?\d*")

def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())

class HybridIndex:

    def __init__(self, docs):
        self.docs = docs
        self.corpus = [tokenize(d.page_content) for d in docs]
        self.bm25 = BM25Okapi(self.corpus)

    def keyword_search(self, query, k=20):
        scores = self.bm25.get_scores(tokenize(query))
        idx = np.argsort(scores)[::-1][:k]
        return [(self.docs[i], scores[i]) for i in idx]

def reciprocal_rank_fusion(vector_results, keyword_results, k=60):
    scores = {}
    id_to_doc = {}

    def register(ranklist, weight):
        for rank, item in enumerate(ranklist):
            doc = item[0]
            chunk_id = doc.metadata["chunk_id"]

            id_to_doc[chunk_id] = doc
            scores.setdefault(chunk_id, 0.0)
            scores[chunk_id] += weight / (k + rank)

    register(vector_results, weight=1.0)
    register(keyword_results, weight=0.8)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    top_docs = []
    for chunk_id, score in ranked[:25]:
        d = id_to_doc[chunk_id]
        d.metadata["rrf_score"] = float(score)
        top_docs.append(d)

    return top_docs

def rerank(query, docs, top_k=6):
    if not docs:
        return []

    pairs = [(query, d.page_content[:800]) for d in docs]
    scores = reranker.predict(pairs)

    probs = 1 / (1 + np.exp(-scores))

    ranked = sorted(zip(docs, scores, probs), key=lambda x: x[1], reverse=True)

    out = []
    for d, raw_score, norm_score in ranked[:top_k]:
        d.metadata["rerank_score_raw"] = float(raw_score)
        d.metadata["rerank_score_norm"] = float(norm_score)
        out.append(d)

    return out
