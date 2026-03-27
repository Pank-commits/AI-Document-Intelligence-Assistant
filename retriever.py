import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import re

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
TOKEN_PATTERN = re.compile(r"[A-Za-z]+|\d+\.\d+|\d+%?|[A-Z][a-z]?\d*")
VECTOR_RRF_WEIGHT = float(os.getenv("VECTOR_RRF_WEIGHT", "0.65"))
BM25_RRF_WEIGHT = float(os.getenv("BM25_RRF_WEIGHT", "0.35"))

def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


def keyword_score(query, text):
    q_words = [w for w in tokenize(query) if len(w) > 1]
    if not q_words:
        return 0.0
    text_lower = (text or "").lower()
    matches = sum(1 for w in q_words if w in text_lower)
    return matches / len(q_words)

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

    register(vector_results, weight=VECTOR_RRF_WEIGHT)
    register(keyword_results, weight=BM25_RRF_WEIGHT)

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

    ranked_items = []
    for d, raw_score, norm_score in zip(docs, scores, probs):
        kw_score = keyword_score(query, d.page_content)
        final_rank_score = (0.7 * float(norm_score)) + (0.3 * float(kw_score))
        ranked_items.append((d, raw_score, norm_score, kw_score, final_rank_score))

    ranked = sorted(ranked_items, key=lambda x: x[4], reverse=True)

    out = []
    for d, raw_score, norm_score, kw_score, final_rank_score in ranked[:top_k]:
        d.metadata["rerank_score_raw"] = float(raw_score)
        d.metadata["rerank_score_norm"] = float(norm_score)
        d.metadata["keyword_score"] = float(kw_score)
        d.metadata["final_rank_score"] = float(final_rank_score)
        out.append(d)

    return out
