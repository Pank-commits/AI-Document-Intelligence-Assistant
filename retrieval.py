import logging
import time
from types import SimpleNamespace

from langchain_core.documents import Document

from qdrant_compat import vector_search
from retriever import reciprocal_rank_fusion, rerank


def _payload_to_doc(hit):
    return Document(
        page_content=hit.payload["text"],
        metadata=hit.payload,
    )

def _safe_float(value, default=0.0):
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _metadata_matches_filter(metadata, metadata_filter):
    if not metadata_filter:
        return True

    for key, value in metadata_filter.items():
        target = str(value or "").strip().lower()
        if not target:
            continue

        direct = str((metadata or {}).get(key) or "").strip().lower()
        if direct == target:
            continue

        plural_key = f"{key}s"
        multi_values = (metadata or {}).get(plural_key) or []
        if isinstance(multi_values, (list, tuple, set)):
            normalized = {str(item).strip().lower() for item in multi_values if item}
            if target in normalized:
                continue

        return False

    return True


def hybrid_retrieve(
    query,
    session_id,
    *,
    qdrant,
    collection_name,
    model,
    session_hybrid_index=None,
    use_expansion=True,
    expand_query_fn=None,
    vector_limit=15,
    final_top_k=8,
    collect_debug=False,
    encode_query_fn=None,
    metadata_filter=None,
):
    """Run Vector Search + BM25 retrieval, fuse candidates, then rerank."""
    t_start = time.perf_counter()
    debug_meta = {
        "stages_ms": {},
        "retrieval_queries_count": 1,
        "retrieval_queries": [query],
        "vector_hits": 0,
        "bm25_hits": 0,
        "fused_docs": 0,
    }

    t_expand = time.perf_counter()
    retrieval_queries = [query]
    if use_expansion and callable(expand_query_fn):
        retrieval_queries = expand_query_fn(query)
    debug_meta["stages_ms"]["expand"] = round((time.perf_counter() - t_expand) * 1000, 2)
    debug_meta["retrieval_queries_count"] = len(retrieval_queries)
    debug_meta["retrieval_queries"] = retrieval_queries

    t_vector = time.perf_counter()
    vector_hit_map = {}
    query_filter = {
        "must": [
            {"key": "session_id", "match": {"value": session_id}},
        ]
    }
    for key, value in (metadata_filter or {}).items():
        if value is not None:
            if key in {"country", "product"}:
                query_filter["must"].append(
                    {
                        "should": [
                            {"key": key, "match": {"value": value}},
                            {"key": f"{key}s", "match": {"value": value}},
                        ]
                    }
                )
            else:
                query_filter["must"].append({"key": key, "match": {"value": value}})

    for rq in retrieval_queries:
        if callable(encode_query_fn):
            encoded = encode_query_fn(rq)
            query_embedding = encoded.tolist() if hasattr(encoded, "tolist") else encoded
        else:
            query_embedding = model.encode(rq, normalize_embeddings=True).tolist()
        hits = vector_search(
            qdrant,
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=vector_limit,
            with_payload=True,
            query_filter=query_filter,
        )
        for hit in hits:
            chunk_id = hit.payload.get("chunk_id")
            if chunk_id not in vector_hit_map or hit.score > vector_hit_map[chunk_id].score:
                vector_hit_map[chunk_id] = hit
    debug_meta["stages_ms"]["vector_search"] = round((time.perf_counter() - t_vector) * 1000, 2)

    vector_hits = sorted(vector_hit_map.values(), key=lambda h: h.score, reverse=True)

    seen_pages = set()
    diverse_hits = []
    for h in vector_hits:
        page = h.payload.get("page")
        if page not in seen_pages or len(diverse_hits) < 5:
            diverse_hits.append(h)
            seen_pages.add(page)
        if len(diverse_hits) >= vector_limit:
            break
    vector_hits = diverse_hits    

    debug_meta["vector_hits"] = len(vector_hits)
    vector_docs = [(_payload_to_doc(hit), hit.score) for hit in vector_hits]

    t_keyword = time.perf_counter()
    keyword_docs = []
    if session_hybrid_index:
        keyword_doc_map = {}
        for rq in retrieval_queries:
            for doc, score in session_hybrid_index.keyword_search(rq, k=vector_limit):
                if not _metadata_matches_filter(doc.metadata, metadata_filter):
                    continue
                chunk_id = doc.metadata.get("chunk_id")
                if chunk_id not in keyword_doc_map or score > keyword_doc_map[chunk_id][1]:
                    keyword_doc_map[chunk_id] = (doc, score)
        keyword_docs = sorted(keyword_doc_map.values(), key=lambda x: x[1], reverse=True)[:vector_limit]
    debug_meta["stages_ms"]["bm25_search"] = round((time.perf_counter() - t_keyword) * 1000, 2)
    debug_meta["bm25_hits"] = len(keyword_docs)

    t_rerank = time.perf_counter()
    fused_docs = reciprocal_rank_fusion(vector_docs, keyword_docs)
    debug_meta["fused_docs"] = len(fused_docs)
    final_docs = rerank(query, fused_docs, top_k=final_top_k)
    debug_meta["stages_ms"]["rerank"] = round((time.perf_counter() - t_rerank) * 1000, 2)

    results = []
    for d in final_docs:
        rerank_score = _safe_float(d.metadata.get("rerank_score_norm"), 0.0)
        keyword_score = _safe_float(d.metadata.get("keyword_score"), 0.0)
        rrf_score = _safe_float(d.metadata.get("rrf_score"), 0.0)
        final_score = (rerank_score * 0.5) + (keyword_score * 0.2) + (rrf_score * 0.3)

        raw_score = d.metadata.get("rerank_score_norm")

        if raw_score is None:
            raw_score = d.metadata.get("rrf_score")

        if raw_score is None:
            raw_score = 0.0

        results.append(
            SimpleNamespace(
                payload={
                    **d.metadata, 
                    "text": d.page_content,
                    "rerank_score": rerank_score,
                    "keyword_score": keyword_score,
                    "rrf_score": rrf_score,
                    "final_score": final_score
                },
                score=final_score,
                id=d.metadata.get("chunk_id"),
            )
        )

    logging.info(f"Vector Search + BM25 retrieval returned {len(results)} results")
    debug_meta["stages_ms"]["total_retrieval"] = round((time.perf_counter() - t_start) * 1000, 2)
    if collect_debug:
        return results, retrieval_queries, debug_meta
    return results, retrieval_queries, None
