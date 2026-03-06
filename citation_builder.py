ANSWER_SPAN_NOT_FOUND = "NOT_FOUND"

def build_citations(answer, filtered_results, answer_span=None, max_sources=3):
    """
    Picks chunks that actually support the answer
    (NOT just top retrieved ones)
    """

    citations = []
    seen = set()

    # prefer span based grounding
    evidence_text = (answer_span if answer_span and answer_span != ANSWER_SPAN_NOT_FOUND else answer).lower()

    for r in filtered_results:
        text = r.payload.get("text", "")
        text_low = text.lower()

        overlap = 0
        for word in evidence_text.split():
            if len(word) > 4 and word in text_low:
                overlap += 1

        if overlap >= 3:   # supporting evidence
            source = r.payload.get("source", "document")
            page = r.payload.get("page", "?")
            if page in (None, "", -1, "-1"):
                page = "?"

            key = (source, str(page))
            if key in seen:
                continue
            seen.add(key)
            citations.append(f"{source} p.{page}")

        if len(citations) >= max_sources:
            break

    return citations
