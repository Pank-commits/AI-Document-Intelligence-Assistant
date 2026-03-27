def vector_search(
    client,
    *,
    collection_name,
    query_vector,
    limit=10,
    with_payload=True,
    query_filter=None,
    **kwargs,
):
    """Compat wrapper for Qdrant vector search across client versions."""
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=with_payload,
            query_filter=query_filter,
            **kwargs,
        )
        return list(getattr(response, "points", response))

    return client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit,
        with_payload=with_payload,
        query_filter=query_filter,
        **kwargs,
    )
