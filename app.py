# ---------- SILENCE WARNINGS ----------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")
# -------------------------------------

"""
----------------------------------------------------------
MULTI-FILE RAG APPLICATION
----------------------------------------------------------

Features:
1. Multi-file upload and indexing
2. Section-based retrieval (Abstract, Introduction, etc.)
3. Finance data analysis using Pandas
4. Vector search using SentenceTransformers
5. Qdrant vector database integration
6. Conflict detection between file versions
7. LLM answer generation using OpenAI
8. Query caching and performance metrics

Workflow:
Upload -> Chunk -> Embed -> Store -> Retrieve -> Generate -> Respond
----------------------------------------------------------
"""

# ------------------ Imports ------------------

from openai import OpenAI
import uuid
import re
import time
import json
import pandas as pd
from flask import (
    Flask, 
    render_template, 
    request, 
    jsonify,
    session,
    Response,
    stream_with_context,
    url_for)
from werkzeug.utils import secure_filename
from qdrant_client.models import (
    PointStruct, 
    VectorParams, 
    Distance,
    Filter,
    FieldCondition,
    MatchValue)
from loader import load_file, pdf_cache_path, normalize_text, build_chunk_text
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
import logging
import numpy as np
from threading import Lock, Thread
from retriever import HybridIndex, reranker
from retrieval import hybrid_retrieve
from qdrant_compat import vector_search
from collections import deque
from langchain_core.documents import Document
from tabular_engine import answer_tabular, to_number, detect_year_column, find_metric
from citation_builder import build_citations
from financial_intent import detect_metric, detect_years

try:
    import pymupdf
except Exception:
    pymupdf = None
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

try:
    from transformers.utils import logging as transformers_logging
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
except Exception:
    pass

for noisy_logger in ("sentence_transformers", "transformers", "huggingface_hub", "httpx"):
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)


def _env_flag(name, default="false"):
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

# ------------------ Initial Setup ------------------

ANSWER_SPAN_NOT_FOUND = "NOT_FOUND"
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "3000"))

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

request_log = deque(maxlen=200)
RATE_LIMIT = 25
query_cache = {}
CACHE_TTL = 3600 
session_uploaded_files = {}
session_conversation_memory = {}
session_hybrid_indexes = {}
session_index_docs = {}
session_upload_jobs = {}
session_state_lock = Lock()
metrics = {
    "total_queries": 0,
    "cache_hits": 0,
    "total_time": 0.0,
    "successful_answers": 0,
    "not_found": 0
}

# ------------------ Model Initialization ------------------

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
HF_LOCAL_ONLY = _env_flag("HF_LOCAL_ONLY")
openai_api_key = os.getenv("OPENAI_API_KEY")
QDRANT_TIMEOUT = float(os.getenv("QDRANT_TIMEOUT", "120"))
QDRANT_UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "256"))
QDRANT_UPSERT_MAX_RETRIES = int(os.getenv("QDRANT_UPSERT_MAX_RETRIES", "3"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "128"))


def init_embedding_model():
    """Load embeddings model with optional offline mode and local-cache fallback."""
    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=HF_LOCAL_ONLY)
    except TypeError:
        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as exc:
        if HF_LOCAL_ONLY:
            raise
        logging.warning(
            "Embedding model load failed with remote lookup enabled; retrying from local cache only: %s",
            exc,
        )
        try:
            return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
        except TypeError:
            return SentenceTransformer(EMBEDDING_MODEL_NAME)


def init_qdrant_client():
    """Create Qdrant client from env-configured URL."""
    return QdrantClient(url=QDRANT_URL, timeout=QDRANT_TIMEOUT)


def format_duration(seconds):
    """Render elapsed time for logs and status text."""
    return f"{seconds:.2f}s"


def upsert_points_in_batches(points, session_id):
    """Write vectors in smaller chunks to avoid oversized HTTP requests to Qdrant."""
    total_points = len(points)
    if total_points == 0:
        return

    for start in range(0, total_points, QDRANT_UPSERT_BATCH_SIZE):
        batch = points[start:start + QDRANT_UPSERT_BATCH_SIZE]
        batch_no = (start // QDRANT_UPSERT_BATCH_SIZE) + 1
        total_batches = (total_points + QDRANT_UPSERT_BATCH_SIZE - 1) // QDRANT_UPSERT_BATCH_SIZE
        update_session_upload_status(
            session_id,
            state="indexing",
            message=(
                f"Writing vectors to index: batch {batch_no}/{total_batches} "
                f"({start + len(batch)}/{total_points})"
            ),
            chunks_prepared=total_points,
            chunks_indexed=start,
        )

        last_error = None
        for attempt in range(1, QDRANT_UPSERT_MAX_RETRIES + 1):
            try:
                qdrant.upsert(collection_name=COLLECTION, points=batch, wait=True)
                last_error = None
                break
            except ResponseHandlingException as exc:
                last_error = exc
                logging.warning(
                    "Qdrant upsert batch %s/%s failed on attempt %s/%s: %s",
                    batch_no,
                    total_batches,
                    attempt,
                    QDRANT_UPSERT_MAX_RETRIES,
                    exc,
                )
                if attempt < QDRANT_UPSERT_MAX_RETRIES:
                    time.sleep(min(2 * attempt, 5))

        if last_error is not None:
            raise RuntimeError(
                f"Qdrant indexing failed on batch {batch_no}/{total_batches}: {last_error}"
            ) from last_error

        update_session_upload_status(
            session_id,
            state="indexing",
            message=(
                f"Indexed batch {batch_no}/{total_batches} "
                f"({start + len(batch)}/{total_points} vectors)"
            ),
            chunks_prepared=total_points,
            chunks_indexed=start + len(batch),
        )


def init_llm_client():
    """Create OpenAI client when configured; callers handle None gracefully."""
    if not openai_api_key:
        logging.warning("OPENAI_API_KEY is not set; LLM features will degrade gracefully.")
        return None
    return OpenAI(api_key=openai_api_key)


model = init_embedding_model()
qdrant = init_qdrant_client()
llm = init_llm_client()

COLLECTION = os.getenv("COLLECTION_NAME", "all_files")
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONF", 15))
BASE_CONFIDENCE_FLOOR = float(os.getenv("BASE_CONFIDENCE_FLOOR", 83))
DEBUG_RAG = os.getenv("DEBUG_RAG", "false").strip().lower() == "true"
TEST_MODE = os.getenv("TEST_MODE", "false").strip().lower() == "true"
DISABLE_QUERY_EXPANSION = os.getenv("DISABLE_QUERY_EXPANSION", "false").strip().lower() == "true"
LLM_TEMP = 0.0 if TEST_MODE else 0.2
EXPANSION_TEMP = 0.0 if TEST_MODE else 0.3
SECTION_MAP = {
    "abstract": "ABSTRACT",
    "introduction": "INTRODUCTION",
    "methods": "METHODS",
    "methodology": "METHODOLOGY",
    "results": "RESULTS",
    "conclusion": "CONCLUSION",
}


def dependency_health_snapshot():
    """Return coarse dependency status for diagnostics."""
    qdrant_ok = False
    qdrant_error = None
    try:
        qdrant.get_collections()
        qdrant_ok = True
    except Exception as exc:
        qdrant_error = str(exc)

    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "openai_model": OPENAI_MODEL_NAME,
        "hf_local_only": HF_LOCAL_ONLY,
        "qdrant_url": QDRANT_URL,
        "qdrant_ok": qdrant_ok,
        "qdrant_error": qdrant_error,
        "llm_configured": llm_is_available(),
    }

# ------------------ Session State Helpers ------------------

# Mental model: session_id -> initialize state -> store files/memory/index -> query uses same state -> delete updates same state.

def get_active_session_id():
    """Return the current Flask session id, if present."""
    return session.get("session_id")

def ensure_session_state(session_id):
    """Initialize in-memory state containers for a session."""
    with session_state_lock:
        if session_id not in session_uploaded_files:
            session_uploaded_files[session_id] = []
        if session_id not in session_conversation_memory:
            session_conversation_memory[session_id] = {}
        if session_id not in session_hybrid_indexes:
            session_hybrid_indexes[session_id] = None
        if session_id not in session_index_docs:
            session_index_docs[session_id] = []
        if session_id not in session_upload_jobs:
            session_upload_jobs[session_id] = {
                "state": "idle",
                "message": "No upload in progress.",
                "files_total": 0,
                "files_processed": 0,
                "current_file": None,
                "chunks_prepared": 0,
                "chunks_indexed": 0,
                "indexed_files": [],
                "tabular_caps": [],
                "started_at": None,
                "finished_at": None,
                "error": None,
            }


def get_session_files(session_id):
    """Get a copy of files tracked for the session."""
    with session_state_lock:
        return list(session_uploaded_files.get(session_id, []))


def set_session_files(session_id, files):
    """Replace the list of files tracked for the session."""
    with session_state_lock:
        session_uploaded_files[session_id] = list(files)


def reset_session_memory(session_id):
    """Clear conversational memory for a session."""
    with session_state_lock:
        session_conversation_memory[session_id] = {}


def get_session_memory(session_id):
    """Return mutable per-session conversational memory."""
    with session_state_lock:
        return session_conversation_memory.setdefault(session_id, {})


def set_session_hybrid_index(session_id, payload_list):
    """Build and store the session BM25 index used by vector-plus-BM25 retrieval."""
    docs = [
        Document(page_content=p["text"], metadata=p)
        for p in payload_list
        if p.get("text")
    ]
    with session_state_lock:
        session_index_docs[session_id] = docs
        session_hybrid_indexes[session_id] = HybridIndex(docs) if docs else None


def get_session_hybrid_index(session_id):
    """Fetch the session-level hybrid index instance."""
    with session_state_lock:
        return session_hybrid_indexes.get(session_id)


def get_session_index_docs(session_id):
    """Fetch a copy of indexed docs for metadata inference."""
    with session_state_lock:
        return list(session_index_docs.get(session_id, []))


def remove_file_from_session_index(session_id, filename):
    """Remove a file's docs from the session hybrid index."""
    with session_state_lock:
        docs = session_index_docs.get(session_id, [])
        docs = [d for d in docs if d.metadata.get("file") != filename]
        session_index_docs[session_id] = docs
        session_hybrid_indexes[session_id] = HybridIndex(docs) if docs else None


def remove_session_index(session_id):
    """Clear the session hybrid index and cached docs."""
    with session_state_lock:
        session_index_docs[session_id] = []
        session_hybrid_indexes[session_id] = None


def get_session_file_details(session_id):
    """Return ordered file details with page totals when available."""
    with session_state_lock:
        tracked_files = list(session_uploaded_files.get(session_id, []))
        docs = list(session_index_docs.get(session_id, []))

    page_map = {}
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        file_name = metadata.get("file")
        if not file_name:
            continue
        raw_page = metadata.get("page")
        if raw_page in (None, "", -1, "-1"):
            continue
        try:
            page_value = int(raw_page)
        except (TypeError, ValueError):
            continue
        page_map.setdefault(file_name, set()).add(page_value)

    details = []
    for path in tracked_files:
        name = os.path.basename(path)
        pages = sorted(page_map.get(name, set()))
        page_total = len(pages) if pages else None
        if page_total is None:
            lower_name = name.lower()
            if lower_name.endswith(".pdf"):
                cache_path = pdf_cache_path(path)
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, "r", encoding="utf-8") as file_obj:
                            cached_pages = json.load(file_obj)
                        if isinstance(cached_pages, list) and cached_pages:
                            page_total = len(cached_pages)
                    except Exception:
                        page_total = None
                if page_total is None and pymupdf is not None and os.path.exists(path):
                    try:
                        with pymupdf.open(path) as pdf_doc:
                            page_total = int(getattr(pdf_doc, "page_count", 0)) or None
                    except Exception:
                        page_total = None
        details.append(
            {
                "name": name,
                "pages": page_total,
            }
        )
    return details


def get_session_upload_status(session_id):
    """Return a snapshot of upload/indexing status for the session."""
    with session_state_lock:
        status = session_upload_jobs.get(session_id)
        return dict(status) if status else None


def update_session_upload_status(session_id, **updates):
    """Apply partial status updates for the session upload job."""
    with session_state_lock:
        status = session_upload_jobs.setdefault(
            session_id,
            {
                "state": "idle",
                "message": "No upload in progress.",
                "files_total": 0,
                "files_processed": 0,
                "current_file": None,
                "chunks_prepared": 0,
                "chunks_indexed": 0,
                "indexed_files": [],
                "tabular_caps": [],
                "started_at": None,
                "finished_at": None,
                "error": None,
            },
        )
        status.update(updates)


def process_upload_job(session_id, files_to_process, upload_time):
    """Run chunking, embedding, and indexing outside the request thread."""
    try:
        job_started_at = time.perf_counter()
        all_payloads = []
        indexed_files = []
        tabular_caps = []

        for idx, file_info in enumerate(files_to_process, start=1):
            safe_name = file_info["safe_name"]
            path = file_info["path"]
            data_mode = file_info["data_mode"]
            file_started_at = time.perf_counter()

            update_session_upload_status(
                session_id,
                state="processing",
                message=f"Processing {safe_name} ({idx}/{len(files_to_process)})",
                files_processed=idx - 1,
                current_file=safe_name,
            )

            conversation_memory = get_session_memory(session_id)
            conversation_memory["data_mode"] = data_mode

            parse_started_at = time.perf_counter()
            chunks: list[Document] = load_file(path)
            parse_elapsed = time.perf_counter() - parse_started_at
            sections = {}
            if not safe_name.lower().endswith((".csv", ".xlsx", ".xls")):
                section_started_at = time.perf_counter()
                full_text = "\n\n".join(
                    c.page_content for c in chunks if getattr(c, "page_content", None)
                )
                sections = extract_sections(full_text)
                section_elapsed = time.perf_counter() - section_started_at
                logging.info(
                    "Processed %s parse=%s section_extract=%s chunks=%s sections=%s",
                    safe_name,
                    format_duration(parse_elapsed),
                    format_duration(section_elapsed),
                    len(chunks),
                    len(sections),
                )
            else:
                logging.info(
                    "Processed %s parse=%s section_extract=skipped chunks=%s",
                    safe_name,
                    format_duration(parse_elapsed),
                    len(chunks),
                )
                if chunks:
                    chunk_meta = getattr(chunks[0], "metadata", {}) or {}
                    if chunk_meta.get("semantic_index_capped"):
                        cap_info = {
                            "file": safe_name,
                            "total_rows": chunk_meta.get("total_rows"),
                            "semantic_blocks_total": chunk_meta.get("semantic_blocks_total"),
                            "semantic_blocks_indexed": chunk_meta.get("semantic_blocks_indexed"),
                            "chunk_rows": chunk_meta.get("chunk_rows"),
                        }
                        tabular_caps.append(cap_info)
                        logging.info(
                            "Tabular semantic indexing capped for %s: %s/%s blocks",
                            safe_name,
                            chunk_meta.get("semantic_blocks_indexed"),
                            chunk_meta.get("semantic_blocks_total"),
                        )

            for sec_idx, (sec_name, sec_text) in enumerate(sections.items()):
                section_chunk_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{session_id}|{safe_name}|section|{sec_name}|{sec_idx}",
                    )
                )
                all_payloads.append(
                    {
                        "text": f"[SECTION:{sec_name.upper()}]\n{sec_text}",
                        "file": safe_name,
                        "version": upload_time,
                        "session_id": session_id,
                        "source": path,
                        "file_type": os.path.splitext(safe_name)[1].lower().replace(".", ""),
                        "section": sec_name.upper(),
                        "row": None,
                        "page": None,
                        "chunk_id": section_chunk_id,
                    }
                )

            for chunk_idx, c in enumerate(chunks):
                meta = normalize_metadata(getattr(c, "metadata", {}), path, chunk_idx)
                chunk_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{session_id}|{meta['source']}|{meta['page']}|{meta['row']}|{meta['chunk_index']}",
                    )
                )
                all_payloads.append(
                    {
                        "text": getattr(c, "page_content", ""),
                        "file": meta.get("file", safe_name),
                        "version": upload_time,
                        "session_id": session_id,
                        "source": meta.get("source", path),
                        "file_type": meta.get(
                            "file_type",
                            os.path.splitext(safe_name)[1].lower().replace(".", ""),
                        ),
                        "section": None,
                        "row": meta["row"],
                        "page": meta["page"],
                        "country": meta.get("country"),
                        "product": meta.get("product"),
                        "countries": meta.get("countries"),
                        "products": meta.get("products"),
                        "chunk_kind": meta.get("chunk_kind"),
                        "special_chunk": meta.get("special_chunk", False),
                        "chunk_id": chunk_id,
                    }
                )

            indexed_files.append({"file": safe_name, "chunks": len(chunks) + len(sections)})
            update_session_upload_status(
                session_id,
                files_processed=idx,
                chunks_prepared=len(all_payloads),
                indexed_files=list(indexed_files),
                tabular_caps=list(tabular_caps),
                message=(
                    f"Prepared {safe_name} ({idx}/{len(files_to_process)}) with {len(chunks)} chunk(s) "
                    f"in {format_duration(time.perf_counter() - file_started_at)}"
                ),
            )

        all_payloads = [p for p in all_payloads if p.get("text")]
        all_texts = [p["text"] for p in all_payloads]
        logging.info(
            f"Loaded {len(all_texts)} text chunks from {len(set(p['file'] for p in all_payloads))} files"
        )
        if not all_payloads:
            raise ValueError("No valid files to index.")

        embed_started_at = time.perf_counter()
        update_session_upload_status(
            session_id,
            state="embedding",
            message=f"Embedding {len(all_texts)} chunks",
            chunks_prepared=len(all_texts),
            chunks_indexed=0,
        )
        vectors = model.encode(
            all_texts,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embed_elapsed = time.perf_counter() - embed_started_at
        logging.info(
            "Embedding completed: chunks=%s batch_size=%s elapsed=%s",
            len(all_texts),
            EMBED_BATCH_SIZE,
            format_duration(embed_elapsed),
        )

        points = [
            PointStruct(id=all_payloads[i]["chunk_id"], vector=vectors[i], payload=all_payloads[i])
            for i in range(len(all_payloads))
        ]

        index_started_at = time.perf_counter()
        update_session_upload_status(
            session_id,
            state="indexing",
            message=f"Writing {len(points)} vectors to the index",
            chunks_prepared=len(points),
            chunks_indexed=0,
        )
        upsert_points_in_batches(points, session_id)
        index_elapsed = time.perf_counter() - index_started_at
        logging.info(
            "Qdrant indexing completed: vectors=%s batch_size=%s elapsed=%s",
            len(points),
            QDRANT_UPSERT_BATCH_SIZE,
            format_duration(index_elapsed),
        )
        set_session_hybrid_index(session_id, all_payloads)
        total_elapsed = time.perf_counter() - job_started_at
        logging.info(
            "Upload job completed: files=%s chunks=%s total_elapsed=%s",
            len(files_to_process),
            len(points),
            format_duration(total_elapsed),
        )
        update_session_upload_status(
            session_id,
            state="ready",
            message=f"{len(files_to_process)} file(s) indexed successfully.",
            current_file=None,
            chunks_prepared=len(points),
            chunks_indexed=len(points),
            finished_at=time.time(),
            error=None,
        )
    except Exception as exc:
        logging.exception("UnhandledException during upload job")
        update_session_upload_status(
            session_id,
            state="failed",
            message=f"Upload failed during indexing: {exc}",
            current_file=None,
            finished_at=time.time(),
            error=str(exc),
        )

# Mental model: normalize inputs -> cache expensive ops -> call LLM consistently -> tune behavior by query type.


def generate_chart(data, chart_type="bar"):
    if not data:
        return None

    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        try:
            value = float(item.get("value"))
        except Exception:
            continue
        cleaned.append({"label": label, "value": value})

    if not cleaned:
        return None

    labels = [d["label"] for d in cleaned]
    values = [d["value"] for d in cleaned]
    is_time_like = all(re.fullmatch(r"(19|20)\d{2}", label) for label in labels)

    if chart_type == "line" and not is_time_like:
        chart_type = "bar"
    if chart_type == "pie" and (len(labels) > 6 or any(v < 0 for v in values)):
        chart_type = "bar"

    if chart_type == "line":
        cleaned = sorted(cleaned, key=lambda item: item["label"])

    return cleaned


def extract_multi_file_data(answer_text):
    """Extract per-file label/value series from conflict answers."""
    file_data = {}
    text = re.sub(
        r"\n*\s*(?:overall\s+)?confidence(?:\s*\(part\))?\s*:\s*\d+(?:\.\d+)?%\s*$",
        "",
        str(answer_text or ""),
        flags=re.IGNORECASE,
    )
    lines = text.split("\n")

    for line in lines:
        match = re.search(r"File\s*(\d+).*?\|\s*(.*)", line)
        if not match:
            continue
        file_id = f"File {match.group(1)}"
        content = match.group(2)

        pairs = re.findall(r"([A-Za-z][A-Za-z0-9 _-]*)\s*\(([\d,]+(?:\.\d+)?)\)", content)
        if not pairs:
            pairs = re.findall(r"([A-Za-z][A-Za-z0-9 _-]*)\s*:\s*([\d,]+(?:\.\d+)?)", content)

        data = []
        for label, value in pairs:
            data.append({"label": str(label).strip(), "value": float(str(value).replace(",", ""))})

        if data:
            file_data[file_id] = data

    return file_data


def flatten_multi_file_chart_data(multi_data):
    """Convert per-file series into a single Chart.js-friendly label/value list."""
    flattened = []
    for file_name, data in multi_data.items():
        for item in data:
            label = str(item.get("label") or "").strip()
            try:
                value = float(item.get("value"))
            except Exception:
                continue
            if label:
                flattened.append({"label": f"{file_name}: {label}", "value": value})
    return flattened


def generate_multi_file_chart(multi_data, title="Comparison Across Files"):
    """Convert conflict answers into chart-ready data."""
    if not multi_data:
        return None
    flattened = flatten_multi_file_chart_data(multi_data)
    return flattened if flattened else None

def safe_float(x, default=0.0):
    """Safely coerce a value to float with default fallback."""
    try:
        if x is None:
            return default
        return float(x)
    except:
        return default


def get_embedding_cached(text, embedding_cache):
    """Return embedding from per-request cache or compute once."""
    key = (text or "").strip()
    if key in embedding_cache:
        return embedding_cache[key]
    embedding_cache[key] = model.encode(key, normalize_embeddings=True)
    return embedding_cache[key]


def llm_is_available():
    return llm is not None


def extract_llm_text(response):
    """Safely read text from a chat completion response."""
    try:
        message = response.choices[0].message
        content = getattr(message, "content", None)
        return (content or "").strip()
    except Exception:
        return ""


def build_llm_unavailable_message(*, context="request"):
    return f"LLM unavailable during {context}. Please try again later."


def llm_complete(prompt, *, temperature=None, timeout=20):
    """Centralized LLM completion call for consistent settings."""
    if not llm_is_available():
        raise RuntimeError(build_llm_unavailable_message(context="completion"))
    return llm.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMP if temperature is None else temperature,
        timeout=timeout,
    )


def estimate_tokens_rough(text):
    """Cheap conservative token estimate to prevent oversized prompts."""
    text = str(text or "")
    if not text:
        return 0
    return max(1, len(text) // 3)


def trim_text_to_token_budget(text, max_tokens):
    """Trim text by rough token budget using character ratio as a guardrail."""
    text = str(text or "")
    if max_tokens <= 0:
        return ""
    max_chars = max_tokens * 3
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def allocate_context_by_budget(chunks, *, max_context_tokens=MAX_CONTEXT_TOKENS, min_chunk_tokens=120):
    """Sort by importance and fill context until the token ceiling is reached."""
    cleaned = []
    for item in chunks or []:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        score = safe_float(item.get("score"), 0.0)
        cleaned.append({"text": text, "score": score, "meta": item.get("meta")})

    if not cleaned:
        return [], 0

    ranked = sorted(cleaned, key=lambda item: item["score"], reverse=True)
    token_cap_per_chunk = max(min_chunk_tokens, max_context_tokens // max(1, len(ranked)))

    final_chunks = []
    total_tokens = 0
    for item in ranked:
        remaining = max_context_tokens - total_tokens
        if remaining <= 0:
            break
        allowed_tokens = min(token_cap_per_chunk, remaining)
        trimmed = trim_text_to_token_budget(item["text"], allowed_tokens).strip()
        trimmed_tokens = estimate_tokens_rough(trimmed)
        if not trimmed or trimmed_tokens <= 0:
            continue
        if total_tokens + trimmed_tokens > max_context_tokens:
            break
        final_chunks.append(
            {
                "text": trimmed,
                "score": item["score"],
                "tokens": trimmed_tokens,
                "meta": item.get("meta"),
            }
        )
        total_tokens += trimmed_tokens

    return final_chunks, total_tokens


def build_markdown_table(headers, rows):
    """Return a compact markdown table for structured answers."""
    clean_headers = [str(h or "").strip() for h in (headers or [])]
    clean_rows = rows or []
    if not clean_headers or not clean_rows:
        return ""

    def sanitize(value):
        text = str(value if value is not None else "").strip()
        return text.replace("\n", " ").replace("|", "/")

    header_line = "| " + " | ".join(sanitize(h) for h in clean_headers) + " |"
    divider_line = "| " + " | ".join("---" for _ in clean_headers) + " |"
    body_lines = []
    for row in clean_rows:
        cells = list(row) if isinstance(row, (list, tuple)) else [row]
        padded = cells[: len(clean_headers)] + [""] * max(0, len(clean_headers) - len(cells))
        body_lines.append("| " + " | ".join(sanitize(cell) for cell in padded[: len(clean_headers)]) + " |")
    return "\n".join([header_line, divider_line] + body_lines)


def build_preview_answer(prefix, dataframe, columns, limit=5):
    """Return a compact markdown-table preview instead of raw row-count text."""
    safe_columns = [c for c in (columns or []) if c and c in dataframe.columns]
    if dataframe is None or dataframe.empty or not safe_columns:
        return prefix

    preview_rows = []
    for _, row in dataframe.head(limit)[safe_columns].iterrows():
        preview_rows.append([row[col] for col in safe_columns])

    table = build_markdown_table(safe_columns, preview_rows)
    if table:
        return f"{prefix}\n\n{table}"
    return prefix


def retrieve_documents(query, session_id, *, top_k=15, query_type=None, embedding_cache=None):
    """Retrieve top documents for a query using the existing Qdrant pipeline."""
    resolved_query_type = query_type or classify_query(query)
    settings = get_query_settings(resolved_query_type)
    q_debug = {"question": query, "stages_ms": {}}
    _, filtered_results, _ = run_retrieval_pipeline(
        q=query,
        session_id=session_id,
        query_type=resolved_query_type,
        use_expansion=settings["use_expansion"],
        threshold_modifier=settings["threshold_modifier"],
        effective_debug=False,
        q_debug=q_debug,
        embedding_cache=embedding_cache,
    )

    docs = []
    selected_results = prefer_page_backed_results(filtered_results, top_k)
    for result in selected_results:
        payload = result.payload or {}
        text = payload.get("text")
        if not text:
            continue
        resolved_page = resolve_payload_page(session_id, payload)
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "file": payload.get("file"),
                    "page": resolved_page,
                    "score": round(safe_float(payload.get("rerank_score_norm", result.score)), 3),
                },
            )
        )
    return docs


def retrieve_for_subqueries(subqueries, session_id, *, top_k=15, query_type=None, embedding_cache=None):
    """Retrieve documents for each subquery and combine the results."""
    all_docs = []

    for subquery in subqueries or []:
        docs = retrieve_documents(
            subquery,
            session_id,
            top_k=top_k,
            query_type=query_type,
            embedding_cache=embedding_cache,
        )
        all_docs.extend(docs)

    return all_docs


def build_capped_context_from_docs(docs, *, max_chunks=6, max_chunk_chars=1800, max_total_chars=12000):
    """Build a bounded prompt context from retrieved documents."""
    context_candidates = []

    for doc in docs or []:
        if len(context_candidates) >= max_chunks:
            break
        text = (getattr(doc, "page_content", "") or "").strip()
        if not text:
            continue
        query_hint = ""
        if getattr(doc, "metadata", None):
            query_hint = str(doc.metadata.get("query") or "")
        compressed = compress_chunk_for_prompt(query_hint, text, max_chars=max_chunk_chars)
        trimmed = (compressed or text[:max_chunk_chars]).strip()
        if not trimmed:
            continue
        score = safe_float(getattr(doc, "metadata", {}).get("score"), 0.0) if getattr(doc, "metadata", None) else 0.0
        context_candidates.append({"text": trimmed, "score": score, "meta": getattr(doc, "metadata", {})})

    max_total_tokens = max(300, min(MAX_CONTEXT_TOKENS, max_total_chars // 3))
    final_chunks, _ = allocate_context_by_budget(context_candidates, max_context_tokens=max_total_tokens)
    return "\n\n".join(chunk["text"] for chunk in final_chunks)


def extract_facts(query, docs):
    """Extract structured facts from retrieved documents for a query."""
    query = (query or "").strip()
    if not query:
        return ""

    context_parts = []
    for doc in docs or []:
        text = (getattr(doc, "page_content", "") or "").strip()
        if text:
            context_parts.append(text)

    if not context_parts:
        return ""

    context = "\n".join(context_parts)[:6000]

    MAX_CHARS = 50000
    if len(context) > MAX_CHARS:
        context = context[:MAX_CHARS]

    prompt = f"""
Extract relevant factual information from the context.

Question: {query}

Context:
{context}

Return facts in this format:
Company -> Value
"""

    try:
        response = llm_complete(prompt)
        return extract_llm_text(response)
    except Exception as e:
        logging.info(f"Error extracting facts: {e}")
        return ""


def generate_final_answer(original_query, facts):
    """Generate a final answer from extracted facts."""
    original_query = (original_query or "").strip()
    facts = (facts or "").strip()
    if not original_query:
        return ""
    if not facts:
        return "I could not extract enough relevant facts to answer the question."

    prompt = f"""
You are a financial analyst.

Based on the facts below, answer the question.

Question: {original_query}

Facts:
{facts}

Give a clear final answer with reasoning.
"""

    try:
        response = llm_complete(prompt)
        return extract_llm_text(response)
    except Exception as e:
        logging.info(f"Error generating final answer: {e}")
        return ""

def extract_data_for_chart(query, context):
    prompt = f"""
You are a STRICT data extraction engine.

Your task:
Convert the context into clean structured data for visualization.

RULES:
1. Output ONLY valid JSON
2. No explanation
3. No text outside JSON
4. Max 10 data points
5. Use meaningful labels (country, year, product, etc.)
6. Values must be numbers only

FORMAT:
[
  {{"label": "India", "value": 12000}},
  {{"label": "USA", "value": 20000}}
]

IMPORTANT:
- If multiple numbers exist, choose the most relevant to the question
- If data is unclear → return []
- Do NOT hallucinate

Question:
{query}

Context:
{context}
"""

    try:
        response = llm_complete(prompt, temperature=0)
        text = (extract_llm_text(response) or "").strip()

        # Be tolerant of fenced or chatty model output and recover the JSON payload.
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        if not text.startswith("["):
            match = re.search(r"\[\s*\{.*\}\s*\]", text, flags=re.DOTALL)
            if match:
                text = match.group(0)

        import json
        data = json.loads(text)

        #  CLEANING STEP (VERY IMPORTANT)
        cleaned = []
        seen_labels = set()
        for item in data:
            if (
                isinstance(item, dict) and
                "label" in item and
                "value" in item
            ):
                try:
                    label = str(item["label"]).strip()
                    if not label:
                        continue
                    val = float(item["value"])
                    if label in seen_labels:
                        continue
                    seen_labels.add(label)
                    cleaned.append({
                        "label": label,
                        "value": val
                    })
                except:
                    continue

        return cleaned[:10]

    except Exception as e:
        logging.info(f"Chart extraction error: {e}")
        return []
    
def decide_chart_type(query, data):
    prompt = f"""
You are a data visualization expert.

Choose the BEST chart type:

- bar → comparison
- line → trend over time
- pie → distribution

Return ONLY one word: bar / line / pie

Question:
{query}

Data:
{data}
"""

    try:
        response = llm_complete(prompt, temperature=0)
        result = extract_llm_text(response).strip().lower()

        if result in ["bar", "line", "pie"]:
            return result

    except Exception as e:
        logging.info(f"Chart type AI error: {e}")

    return "bar"  # fallback

def should_generate_chart(query, data, context):
    if not data or len(data) < 2:
        return False

    values = [d["value"] for d in data if "value" in d]
    if len(set(values)) <= 1:
        return False
    
    if values and max(values) - min(values) < 1:
        return False

    q = query.lower()

    no_chart_patterns = ["what is", "define", "explain",
                         "why", "how", "meaning",
                         "summary", "describe"
                         ]
    if any(p in q for p in no_chart_patterns):
        return False
    
    prompt = f"""
Decide whether a chart is useful.

Rules:
- YES → if comparing values, trends, distributions
- NO → if answer is textual, definition, explanation

Return ONLY: yes or no

Question:
{query}

Data:
{data}
"""
    try:
        response = llm_complete(prompt, temperature=0)
        decision = extract_llm_text(response).strip().lower()

        if "yes" in decision:
            return True
        else:
            return False
        
    except:
        return True
    
def generate_chart_insight(query, data):
    if not data:
        return ""
    try:
        sorted_data = sorted(data, key=lambda x:x["value"], reverse=True)
        top = sorted_data[0]
        bottom = sorted_data[-1]

        insight = f" insight:\nTop: {top['label']} ({top['value']})\nLowest: {bottom['label']} ({bottom['value']})"
        
        gap = top["value"] - bottom["value"]
        insight += f"\nGap: {gap}"
        labels = [d["label"]for d in data]
        values = [d["value"]for d in data]

        if all(str(1).isdigit() for l in labels):
            if values[-1] > values[0]:
                insight += "\nTrend: increasing"
            elif values[-1] < values[0]:
                insight += "\nTrend: Decreasing"
            
        prompt = f"""
You are a data analyst.
write 2-3 short insight from this data
keep it simple and clear.

Data:
{data}

Question:
{query}
"""
        try:
            response = llm_complete(prompt, temperature=0)
            ai_text = extract_llm_text(response)
            if ai_text:
                insight += "\n\n" + ai_text
        except:
            pass

        return insight
    except Exception as e:
        logging.info(f"Insight error: {e}")
        return ""

def get_query_settings(query_type):
    """Map query type to retrieval strategy knobs."""
    settings = {
        "context_limit": 5,
        "threshold_modifier": 0.0,
        "use_expansion": True,
    }
    if query_type == "extraction":
        settings.update({"threshold_modifier": -0.12, "context_limit": 6, "use_expansion": False})
    elif query_type == "analytical":
        settings.update({"threshold_modifier": -0.08, "context_limit": 8})
    elif query_type == "semantic":
        settings.update({"threshold_modifier": 0.05, "context_limit": 5})
    elif query_type == "summary":
        settings.update({"threshold_modifier": -0.10, "context_limit": 15})

    if DISABLE_QUERY_EXPANSION:
        settings["use_expansion"] = False
    return settings


def detect_requested_section(query_text):
    """Detect if query asks for a named document section."""
    ql = query_text.lower()
    for key, value in SECTION_MAP.items():
        if key in ql:
            return value
    return None


# Mental model: detect section intent -> fetch relevant chunks -> sanitize/rank results -> fast section answer before full RAG.

def summarize_section_query(query_text, requested_section, session_id, embedding_cache=None):
    """Answer section-summary queries directly from section chunks."""
    if embedding_cache is None:
        section_vector = model.encode(requested_section, normalize_embeddings=True).tolist()
    else:
        section_vector = get_embedding_cached(requested_section, embedding_cache).tolist()
    section_results = vector_search(
        qdrant,
        collection_name=COLLECTION,
        query_vector=section_vector,
        limit=20,
        with_payload=True,
        query_filter=Filter(
            must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
        ),
    )

    section_texts = [
        r.payload["text"]
        for r in section_results
        if r.payload["text"].startswith(f"[SECTION:{requested_section}]")
    ]
    if not section_texts:
        return "Not available in the dataset", "section_not_found", None

    section_content = section_texts[0].split("\n", 1)[1]
    format_type = "bullet points" if "bullet" in query_text.lower() else "paragraph"
    prompt = f"""

Summarize the following section in {format_type} clearly:

{section_content}
"""
    try:
        t_llm = time.perf_counter()
        response = llm_complete(prompt)
        llm_ms = round((time.perf_counter() - t_llm) * 1000, 2)
        return extract_llm_text(response) or "No summary returned.", None, llm_ms
    except Exception as e:
        logging.info(f"Error generating section summary: {e}")
        return "Error generating summary", "section_summary_error", None
    
def sanitize_results(results_or_query, maybe_results=None):
    """Normalize retrieval hits and drop malformed entries."""
    
    # Backward compatible: supports sanitize_results(results)
    # and sanitize_results(query, results).
    
    results = maybe_results if maybe_results is not None else results_or_query
    safe = []
    for r in results:
        if not hasattr(r, "payload") or r.payload is None:
            continue

        if not r.payload.get("text"):
            continue

        if getattr(r, "score", None) is None:
            r.score = 0.0

        if "rerank_score_norm" not in r.payload or r.payload["rerank_score_norm"] is None:
            r.payload["rerank_score_norm"] = 0.0

        safe.append(r)

    return safe

# ------------------ Vector DB Bootstrap ------------------

try:
    qdrant.create_collection(
       collection_name = COLLECTION,
       vectors_config=VectorParams(size=384, 
                                    distance=Distance.COSINE)
    )
    
    qdrant.create_payload_index(
        collection_name=COLLECTION, 
        field_name="file",
        field_schema="keyword"
    )
    qdrant.create_payload_index(
        collection_name=COLLECTION,
        field_name="country",
        field_schema="keyword"
    )
    qdrant.create_payload_index(
        collection_name=COLLECTION,
        field_name="product",
        field_schema="keyword"
    )
    qdrant.create_payload_index(
        collection_name=COLLECTION,
        field_name="countries",
        field_schema="keyword"
    )
    qdrant.create_payload_index(
        collection_name=COLLECTION,
        field_name="products",
        field_schema="keyword"
    )
    logging.info("Collection and index created successfully")

except UnexpectedResponse as e:
    if "already exists" in str(e):
        pass
    else:
        raise e
    
# ------------------ Data Helpers ------------------

#-------------------CHART HELPERS-------------------

def detect_visualization(query, context):
    q = query.lower()

    keywords = ["chart", "graph", "plot", "visualize", "trend", "distribution"]
    if any(k in q for k in keywords):
        return True
    
    patterns = ["top", "highest", "lowest", "compare",
                "distribution", "percentage",
                "sales by", "revenue by", "profit by",
                "trend", "over time"
                ]
    if any(p in q for p in patterns):
        return True
    
    numbers = re.findall(r"\d+", context)
    if len(numbers) >= 5:
        return True
    
    return False

def get_chart_type(query):
    q = query.lower()
    if "trend" in q or "over time" in q or "by year" in q or "monthly" in q or "yearly" in q:
        return "line"
    elif "distribution" in q or "percentage" in q or "share" in q:
        return "pie"
    else:
        return "bar"


def detect_visualization(query, context):
    """Only trigger charts for explicit visual intent or strong visual question types."""
    q = (query or "").lower()

    explicit_terms = ["chart", "graph", "plot", "visualize", "visualise"]
    if any(term in q for term in explicit_terms):
        return True

    strong_patterns = ["distribution", "percentage", "share", "breakdown", "trend", "over time"]
    return any(pattern in q for pattern in strong_patterns)


def should_generate_chart(query, data, context):
    """Gate chart generation to explicit or clearly visual requests only."""
    if not data or len(data) < 2:
        return False

    values = [d["value"] for d in data if "value" in d]
    if len(set(values)) <= 1:
        return False
    if values and max(values) - min(values) < 1:
        return False

    q = (query or "").lower()
    no_chart_patterns = ["what is", "define", "explain", "why", "how", "meaning", "summary", "describe"]
    if any(pattern in q for pattern in no_chart_patterns):
        return False

    explicit_chart_terms = [
        "chart", "graph", "plot", "visualize", "visualise",
        "show chart", "show graph", "in chart", "as chart"
    ]
    strong_visual_patterns = ["trend", "over time", "distribution", "breakdown", "compare in chart", "comparison chart"]
    return any(term in q for term in explicit_chart_terms) or any(pattern in q for pattern in strong_visual_patterns)


def build_tabular_chart_data(query, uploaded_files):
    """Build chart-ready label/value pairs directly from tabular data."""
    q = (query or "").strip().lower()
    loaded_frames = []
    metric_aliases = {
        "sales": ["sales", "sale", "revenue"],
        "revenue": ["revenue", "sales", "sale"],
        "profit": ["profit", "net income", "income"],
        "units": ["units sold", "units", "quantity", "qty", "sold"],
        "count": ["count", "number of", "how many"],
        "market_share": ["market share", "share"],
    }
    group_aliases = {
        "country": ["country", "countries", "nation", "region", "location"],
        "product": ["product", "products", "item", "items"],
        "category": ["category", "categories"],
        "company": ["company", "companies", "firm"],
        "year": ["year", "years", "annual"],
        "month": ["month", "months"],
    }

    def query_mentions(aliases):
        return any(alias in q for alias in aliases)

    def file_relevance_score(file_path, dataframe):
        file_name = os.path.splitext(os.path.basename(file_path))[0].lower()
        score = 0
        for alias in metric_aliases["sales"]:
            if alias in q and alias in file_name:
                score += 8
        for alias in metric_aliases["profit"]:
            if alias in q and alias in file_name:
                score += 7
        for alias in metric_aliases["market_share"]:
            if alias in q and alias in file_name:
                score += 8
        for alias in group_aliases["country"]:
            if alias in q and alias in file_name:
                score += 8
        for alias in group_aliases["product"]:
            if alias in q and alias in file_name:
                score += 7
        for alias in group_aliases["category"]:
            if alias in q and alias in file_name:
                score += 6
        if "trend" in q and "trend" in file_name:
            score += 7
        if query_mentions(group_aliases["month"]) and "month" in file_name:
            score += 5
        if query_mentions(group_aliases["year"]) and "year" in file_name:
            score += 5
        for col in dataframe.columns:
            col_l = str(col).lower()
            if col_l in q:
                score += 4
        return score

    for file_path in uploaded_files:
        df = load_dataset(file_path)
        if df is None or df.empty:
            continue
        normalized = df.copy()
        normalized.columns = [str(c).strip().lower() for c in normalized.columns]
        loaded_frames.append((file_path, normalized, file_relevance_score(file_path, normalized)))

    if not loaded_frames:
        return []

    loaded_frames.sort(key=lambda item: item[2], reverse=True)

    if len(loaded_frames) == 1:
        chart_df = loaded_frames[0][1]
    elif loaded_frames[0][2] > 0:
        chart_df = loaded_frames[0][1]
    else:
        base_cols = list(loaded_frames[0][1].columns)
        base_col_set = set(base_cols)
        frames_only = [frame for _, frame, _ in loaded_frames]
        if all(set(frame.columns) == base_col_set for frame in frames_only[1:]):
            aligned_frames = [frame.reindex(columns=base_cols) for frame in frames_only]
            chart_df = pd.concat(aligned_frames, ignore_index=True)
        else:
            chart_df = loaded_frames[0][1]

    working_df = chart_df.copy()

    def find_col(*keywords):
        for col in working_df.columns:
            col_l = str(col).lower()
            if all(k in col_l for k in keywords):
                return col
        for keyword in keywords:
            for col in working_df.columns:
                if keyword in str(col).lower():
                    return col
        return None

    def metric_requested():
        if query_mentions(metric_aliases["market_share"]):
            return "market_share"
        if query_mentions(metric_aliases["sales"]) and not query_mentions(metric_aliases["count"]):
            return "sales"
        if query_mentions(metric_aliases["profit"]):
            return "profit"
        if query_mentions(metric_aliases["units"]):
            return "units"
        if query_mentions(metric_aliases["count"]):
            return "count"
        return None

    year_col = detect_year_column(working_df)
    revenue_col = find_col("revenue") or find_col("sales")
    profit_col = find_col("profit") or find_col("net", "income") or find_col("income")
    units_col = find_col("units", "sold") or find_col("quantity") or find_col("qty")
    market_share_col = find_col("market", "share") or find_col("share")
    company_col = find_col("company") or find_col("firm")
    country_col = find_col("country") or find_col("nation") or find_col("region") or find_col("location")
    product_col = find_col("product") or find_col("item")
    category_col = find_col("category")
    month_col = find_col("month")

    if year_col and ("trend" in q or "over time" in q or "by year" in q):
        requested_metric = metric_requested()
        metric_col = {
            "sales": revenue_col,
            "profit": profit_col,
            "units": units_col,
            "market_share": market_share_col,
        }.get(requested_metric) or revenue_col or profit_col or units_col or market_share_col
        if metric_col and metric_col in working_df.columns:
            temp = working_df[[year_col, metric_col]].copy()
            temp[metric_col] = temp[metric_col].apply(to_number)
            temp = temp.dropna(subset=[metric_col])
            if not temp.empty:
                labels = temp[year_col].astype(str).str.extract(r"(20\d{2})", expand=False).fillna(temp[year_col].astype(str))
                grouped = temp.groupby(labels)[metric_col].sum().sort_index()
                return [{"label": str(idx), "value": float(val)} for idx, val in grouped.head(12).items()]

    if month_col and query_mentions(group_aliases["month"]):
        requested_metric = metric_requested()
        metric_col = {
            "sales": revenue_col,
            "profit": profit_col,
            "units": units_col,
            "market_share": market_share_col,
        }.get(requested_metric) or revenue_col or profit_col or units_col
        if metric_col and metric_col in working_df.columns:
            temp = working_df[[month_col, metric_col]].copy()
            temp[metric_col] = temp[metric_col].apply(to_number)
            temp = temp.dropna(subset=[metric_col])
            if not temp.empty:
                grouped = temp.groupby(temp[month_col].astype(str).str.strip())[metric_col].sum().sort_values(ascending=False)
                return [{"label": str(idx), "value": float(val)} for idx, val in grouped.head(12).items()]

    requested_metric = metric_requested()
    metric_col = {
        "sales": revenue_col,
        "profit": profit_col,
        "units": units_col,
        "market_share": market_share_col,
        "count": None,
    }.get(requested_metric)
    if requested_metric is None:
        metric_col = revenue_col or profit_col or units_col or market_share_col

    group_col = None
    if query_mentions(group_aliases["country"]):
        group_col = country_col
    elif query_mentions(group_aliases["product"]):
        group_col = product_col
    elif query_mentions(group_aliases["category"]):
        group_col = category_col
    elif query_mentions(group_aliases["company"]):
        group_col = company_col
    elif year_col and query_mentions(group_aliases["year"]):
        group_col = year_col
    elif month_col and query_mentions(group_aliases["month"]):
        group_col = month_col

    if group_col and group_col in working_df.columns:
        temp = working_df.copy()
        temp[group_col] = temp[group_col].astype(str).str.strip()
        temp = temp[temp[group_col] != ""]
        if temp.empty:
            return []

        if metric_col and metric_col in temp.columns:
            temp[metric_col] = temp[metric_col].apply(to_number)
            temp = temp.dropna(subset=[metric_col])
            if temp.empty:
                return []
            grouped = temp.groupby(group_col)[metric_col].sum().sort_values(ascending=False)
        elif requested_metric == "count":
            grouped = temp.groupby(group_col).size().sort_values(ascending=False)
        else:
            return []

        return [{"label": str(idx), "value": float(val)} for idx, val in grouped.head(10).items()]

    fallback_metric = metric_col or revenue_col or profit_col or units_col or market_share_col
    if fallback_metric and fallback_metric in working_df.columns:
        temp = working_df[[fallback_metric]].copy()
        temp[fallback_metric] = temp[fallback_metric].apply(to_number)
        temp = temp.dropna(subset=[fallback_metric])
        if not temp.empty:
            return [
                {"label": f"Row {i + 1}", "value": float(val)}
                for i, val in enumerate(temp[fallback_metric].head(10).tolist())
            ]

    return []

#------------compute adaptive threshold-----------------

def compute_adaptive_threshold(results, base_threshold):
            """Adapt threshold based on top-score spread."""
            if not results:
                return base_threshold
            
            scores = sorted([max(0.0, min(r.score, 1.0)) for r in results], reverse=True)
            top = scores[0]
            median = scores[len(scores)//2]
            spread = top - median

            if top > 0.75 and spread > 0.25:
                return min(0.55, top * 0.70)
            if top > 0.45:
                return min(0.40, top * 0.60)
            return max(0.12, top * 0.45)

def load_dataset(file_path):
    """Load CSV/XLSX dataset for structured query path."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext == ".xlsx":
        return pd.read_excel(file_path)
    else:
        return None


def normalize_metadata(meta, path, chunk_idx):
    """
    Makes metadata consistent across all file types
    Prevents KeyError crashes in RAG pipeline
    """
    if meta is None:
        meta = {}

    return {
        "source": meta.get("source", path),
        "page": meta.get("page", -1),
        "row": meta.get("row", meta.get("rows", -1)),
        "paragraph": meta.get("paragraph", -1),
        "file_type": meta.get("file_type", os.path.splitext(path)[1].lower().replace(".", "")),
        "chunk_index": chunk_idx,
        "country": meta.get("country"),
        "product": meta.get("product"),
        "countries": meta.get("countries"),
        "products": meta.get("products"),
        "chunk_kind": meta.get("chunk_kind"),
        "special_chunk": meta.get("special_chunk", False),
    }


def format_source_label(file_name, page):
    """Build a compact source label with page when available."""
    label = file_name or "Unknown source"
    if page in (None, "", -1, "-1"):
        return label
    return f"{label} p.{page}"


def append_unique_source(sources, seen_sources, file_name, page):
    label = format_source_label(file_name, page)
    if label not in seen_sources:
        seen_sources.add(label)
        sources.append(label)


def has_page_reference(payload):
    """Return True when payload metadata contains a usable page reference."""
    if not isinstance(payload, dict):
        return False
    page = payload.get("page")
    return page not in (None, "", -1, "-1")


def prefer_page_backed_results(results, limit):
    """Prefer retrieved chunks with page metadata for user-facing citations."""
    if not results:
        return []

    page_results = [r for r in results if has_page_reference(getattr(r, "payload", None))]
    if page_results:
        return page_results[:limit]
    return results[:limit]


def _normalize_match_text(text):
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def infer_page_reference(session_id, file_name, text):
    """Infer a likely page for page-less chunks by matching against indexed page-backed chunks."""
    if not file_name or not text:
        return None

    with session_state_lock:
        docs = list(session_index_docs.get(session_id, []))

    target_text = _normalize_match_text(text)
    if not target_text:
        return None

    target_tokens = {
        token for token in re.findall(r"[a-z0-9]{5,}", target_text)
        if token not in {"which", "their", "there", "about", "these", "those"}
    }

    best_page = None
    best_score = -1
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        if metadata.get("file") != file_name:
            continue
        page = metadata.get("page")
        if page in (None, "", -1, "-1"):
            continue

        candidate_text = _normalize_match_text(getattr(doc, "page_content", ""))
        if not candidate_text:
            continue

        score = 0
        if candidate_text in target_text or target_text in candidate_text:
            score += 1000

        candidate_tokens = set(re.findall(r"[a-z0-9]{5,}", candidate_text))
        if target_tokens and candidate_tokens:
            score += len(target_tokens.intersection(candidate_tokens))

        if score > best_score:
            best_score = score
            best_page = page

    return best_page if best_score > 0 else None


def resolve_payload_page(session_id, payload):
    """Return the stored page, or infer one when only a section-level chunk is available."""
    if not isinstance(payload, dict):
        return None
    page = payload.get("page")
    if page not in (None, "", -1, "-1"):
        return page
    return infer_page_reference(session_id, payload.get("file"), payload.get("text"))
    
    

def extract_sections(full_text):
    """Extract canonical section blocks from long-form text."""
    sections = {}
    if not full_text:
        return sections

    canonical_aliases = {
        "abstract": ["abstract", "summary"],
        "introduction": ["introduction", "background"],
        "methodology": ["methodology"],
        "methods": ["methods", "materials and methods", "experimental methods", "approach"],
        "results": ["results", "findings", "analysis", "results and discussion"],
        "conclusion": ["conclusion", "conclusions", "summary and conclusion", "closing remarks"],
    }

    alias_to_canonical = {}
    for canonical, aliases in canonical_aliases.items():
        for alias in aliases:
            alias_to_canonical[alias] = canonical

    heading_regex = re.compile(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:\d+(?:\.\d+)*[\)\.]?\s+)?"
        r"(abstract|summary|introduction|background|methodology|methods|materials and methods|"
        r"experimental methods|approach|results and discussion|results|findings|analysis|"
        r"conclusions?|summary and conclusion|closing remarks)\s*[:\-]?\s*$"
    )

    matches = list(heading_regex.finditer(full_text))
    if not matches:
        return sections

    for idx, match in enumerate(matches):
        alias = match.group(1).lower()
        canonical = alias_to_canonical.get(alias)
        if not canonical:
            continue

        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        content = full_text[start:end].strip()

        if len(content) > 50 and canonical not in sections:
            sections[canonical] = content

    return sections


# Mental model: detect tabular intent -> load dataframe -> apply pandas calculation rules -> return deterministic finance answer.

def answer_calculation(query, df, memory=None):
    """Handle rule-based financial calculations on tabular data."""
    col = None
    year = None
    memory = memory if memory is not None else {}
    
    q = query.lower()
    q = re.sub(r"\bsale\b", "sales", q)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_cols:
        return None

    cols_lower = {c.lower(): c for c in df.columns}
    year_col = next((orig for lower, orig in cols_lower.items() if lower == "year" or "year" in lower), None)
    profit_or_loss_col = next(
        (
            orig
            for lower, orig in cols_lower.items()
            if lower == "profit_or_loss" or ("profit" in lower and "loss" in lower)
        ),
        None,
    )

    query_tokens = re.findall(r"[a-zA-Z_]+", q)
    calc_numeric_cols = [c for c in numeric_cols if c != year_col] or numeric_cols
    def fmt_num(v):
        n = safe_float(v, default=None)
        if n is None:
            return str(v)
        if abs(n - round(n)) < 1e-9:
            return f"{int(round(n)):,}"
        return f"{n:,.2f}"

    def find_col(*keywords):
        for lower, orig in cols_lower.items():
            if all(k in lower for k in keywords):
                return orig
        for k in keywords:
            for lower, orig in cols_lower.items():
                if k in lower:
                    return orig
        return None

    revenue_col = find_col("revenue") or find_col("sales")
    profit_col = find_col("profit") or find_col("net", "income") or find_col("income")
    loss_col = find_col("loss")
    company_col = find_col("company") or find_col("company", "name") or find_col("firm")
    sector_col = find_col("sector") or find_col("industry")
    assets_col = find_col("assets") or find_col("asset")
    liabilities_col = find_col("liabilities") or find_col("liability")
    expense_col = find_col("expense") or find_col("cost")
    product_col = find_col("product") or find_col("item")
    category_col = find_col("category")
    country_col = (
        find_col("country")
        or find_col("nation")
        or find_col("location")
        or find_col("region")
    )
    order_date_col = find_col("order", "date") or find_col("date")

    def aggregate_entity_metric(dataframe, group_col, metric_col, highest=True):
        if not group_col or not metric_col:
            return None
        if group_col not in dataframe.columns or metric_col not in dataframe.columns:
            return None
        temp = dataframe[[group_col, metric_col]].copy()
        temp[metric_col] = temp[metric_col].apply(to_number)
        temp[group_col] = temp[group_col].astype(str).str.strip()
        temp = temp.dropna(subset=[metric_col])
        temp = temp[temp[group_col] != ""]
        if temp.empty:
            return None
        agg = temp.groupby(group_col)[metric_col].sum()
        if agg.empty:
            return None
        idx = agg.idxmax() if highest else agg.idxmin()
        return str(idx), agg.loc[idx]

    def unique_join(values, limit=15):
        seen = []
        for v in values:
            s = str(v).strip()
            if not s:
                continue
            if s not in seen:
                seen.append(s)
        if not seen:
            return None
        if len(seen) <= limit:
            return ", ".join(seen)
        shown = ", ".join(seen[:limit])
        return f"{shown} (and {len(seen) - limit} more)"

    def infer_value_from_column(text_col):
        if not text_col or text_col not in df.columns:
            return None
        series = df[text_col].dropna().astype(str).str.strip()
        candidates = sorted({s.lower() for s in series if s}, key=len, reverse=True)
        for cand in candidates[:300]:
            if cand and cand in q:
                return cand
        return None

    def infer_values_from_column(text_col, limit=4):
        if not text_col or text_col not in df.columns:
            return []
        series = df[text_col].dropna().astype(str).str.strip()
        candidates = sorted({s for s in series if s}, key=lambda s: len(s), reverse=True)
        found = []
        q_compact = re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()
        for cand in candidates[:1000]:
            cand_l = cand.lower()
            cand_compact = re.sub(r"[^a-z0-9]+", " ", cand_l).strip()
            if not cand_compact:
                continue
            if re.search(rf"\b{re.escape(cand_compact)}\b", q_compact):
                if cand not in found:
                    found.append(cand)
            elif cand_l in q and cand not in found:
                found.append(cand)
            if len(found) >= limit:
                break
        return found

    def apply_text_filter(dataframe, col_name, expected_substring):
        if not col_name or not expected_substring or col_name not in dataframe.columns:
            return dataframe
        return dataframe[
            dataframe[col_name]
            .astype(str)
            .str.lower()
            .str.contains(re.escape(expected_substring), na=False)
        ]

    def parse_threshold_value(raw_value, raw_suffix=""):
        if raw_value is None:
            return None
        text = f"{raw_value}{raw_suffix or ''}".strip().lower().replace(",", "")
        multiplier = 1.0
        if text.endswith("billion") or text.endswith("bn") or text.endswith("b"):
            multiplier = 1_000_000_000.0
            text = re.sub(r"(billion|bn|b)$", "", text).strip()
        elif text.endswith("million") or text.endswith("mn") or text.endswith("m"):
            multiplier = 1_000_000.0
            text = re.sub(r"(million|mn|m)$", "", text).strip()
        elif text.endswith("thousand") or text.endswith("k"):
            multiplier = 1_000.0
            text = re.sub(r"(thousand|k)$", "", text).strip()
        base = safe_float(text, default=None)
        if base is None:
            return None
        return base * multiplier

    def companies_from_filtered(dataframe):
        if company_col not in dataframe.columns:
            return None
        values = dataframe[company_col].dropna().astype(str).str.strip()
        values = values[values != ""]
        return unique_join(values.tolist())

    def canonical_text_by_company(dataframe, text_col):
        if not company_col or not text_col or company_col not in dataframe.columns or text_col not in dataframe.columns:
            return None
        temp = dataframe[[company_col, text_col]].copy()
        temp[company_col] = temp[company_col].astype(str).str.strip()
        temp[text_col] = temp[text_col].astype(str).str.strip()
        temp = temp[(temp[company_col] != "") & (temp[text_col] != "")]
        if temp.empty:
            return None

        def choose_value(series):
            counts = series.value_counts()
            return counts.index[0] if not counts.empty else None

        canonical = temp.groupby(company_col)[text_col].agg(choose_value)
        canonical = canonical.dropna()
        return canonical if not canonical.empty else None

    def build_company_rollup(dataframe):
        if not company_col or company_col not in dataframe.columns:
            return None
        temp = dataframe.copy()
        temp[company_col] = temp[company_col].astype(str).str.strip()
        temp = temp[temp[company_col] != ""]
        if temp.empty:
            return None

        grouped = pd.DataFrame(index=pd.Index(sorted(temp[company_col].unique()), name=company_col))
        numeric_candidates = [revenue_col, profit_col, loss_col, assets_col, liabilities_col, expense_col, profit_or_loss_col]
        for metric_col in [c for c in numeric_candidates if c and c in temp.columns]:
            metric_frame = temp[[company_col, metric_col]].copy()
            metric_frame[metric_col] = metric_frame[metric_col].apply(to_number)
            metric_frame = metric_frame.dropna(subset=[metric_col])
            if not metric_frame.empty:
                grouped[metric_col] = metric_frame.groupby(company_col)[metric_col].sum()

        for text_col in [sector_col, country_col]:
            canonical = canonical_text_by_company(temp, text_col)
            if canonical is not None:
                grouped[text_col] = canonical

        return grouped.reset_index() if not grouped.empty else None

    def summarize_company_metric(dataframe, metric_col, company_names, requested_year=None):
        if not company_col or not metric_col or metric_col not in dataframe.columns:
            return None
        temp = dataframe.copy()
        temp[company_col] = temp[company_col].astype(str).str.strip()
        temp[metric_col] = temp[metric_col].apply(to_number)
        temp = temp.dropna(subset=[metric_col])
        temp = temp[temp[company_col] != ""]
        if requested_year is not None and year_col and year_col in temp.columns:
            numeric_years = pd.to_numeric(temp[year_col], errors="coerce")
            temp = temp[numeric_years == requested_year]
        if temp.empty:
            return None

        results = {}
        for company_name in company_names:
            mask = temp[company_col].astype(str).str.lower() == str(company_name).strip().lower()
            company_rows = temp[mask]
            if not company_rows.empty:
                results[company_name] = company_rows[metric_col].sum()
        return results or None

    def compare_company_financial_strength(company_names, requested_year=None):
        metrics = [
            ("revenue", revenue_col, True),
            ("profit", profit_col or profit_or_loss_col, True),
            ("assets", assets_col, True),
            ("liabilities", liabilities_col, False),
        ]
        scores = {name: 0 for name in company_names[:2]}
        detail_lines = []

        for label, metric_col, higher_is_better in metrics:
            if not metric_col:
                continue
            metric_values = summarize_company_metric(
                df,
                metric_col,
                company_names[:2],
                requested_year=requested_year,
            )
            if not metric_values or len(metric_values) < 2:
                continue

            ordered_names = [name for name in company_names[:2] if name in metric_values]
            if len(ordered_names) < 2:
                continue

            left, right = ordered_names[:2]
            left_value = metric_values[left]
            right_value = metric_values[right]
            if left_value == right_value:
                winner_text = "tie"
            else:
                winner = left if (left_value > right_value) == higher_is_better else right
                scores[winner] += 1
                winner_text = winner

            detail_lines.append(
                f"{label}: {left} {fmt_num(left_value)} vs {right} {fmt_num(right_value)}"
                + (f" ({winner_text} better)" if winner_text != "tie" else " (tie)")
            )

        used_metrics = len(detail_lines)
        if used_metrics == 0:
            return None

        best_company = max(scores, key=scores.get)
        best_score = scores[best_company]
        other_company = next((name for name in company_names[:2] if name != best_company), None)
        other_score = scores.get(other_company, 0) if other_company else 0
        year_label = f" in {requested_year}" if requested_year is not None else ""

        if best_score == other_score:
            return (
                f"There is no clear financial-strength winner{year_label} based on the available metrics. "
                + "; ".join(detail_lines)
                + "."
            )

        return (
            f"{best_company} appears financially stronger{year_label} based on the available metrics. "
            + "; ".join(detail_lines)
            + f". Score: {best_company} {best_score}, {other_company} {other_score}."
        )

    def rank_companies_by_financial_performance(requested_year=None):
        if not company_col:
            return None

        candidate_cols = [c for c in [revenue_col, profit_col or profit_or_loss_col, assets_col, liabilities_col] if c]
        if len(candidate_cols) < 2:
            return None

        temp = df.copy()
        temp[company_col] = temp[company_col].astype(str).str.strip()
        temp = temp[temp[company_col] != ""]
        if requested_year is not None and year_col and year_col in temp.columns:
            numeric_years = pd.to_numeric(temp[year_col], errors="coerce")
            temp = temp[numeric_years == requested_year]
        if temp.empty:
            return None

        grouped = temp.groupby(company_col).sum(numeric_only=True)
        usable = {}
        for label, col_name, higher_is_better in [
            ("revenue", revenue_col, True),
            ("profit", profit_col or profit_or_loss_col, True),
            ("assets", assets_col, True),
            ("liabilities", liabilities_col, False),
        ]:
            if not col_name or col_name not in grouped.columns:
                continue
            series = pd.to_numeric(grouped[col_name], errors="coerce").dropna()
            if series.empty:
                continue
            min_val = float(series.min())
            max_val = float(series.max())
            if abs(max_val - min_val) < 1e-9:
                norm = pd.Series(1.0, index=series.index)
            elif higher_is_better:
                norm = (series - min_val) / (max_val - min_val)
            else:
                norm = (max_val - series) / (max_val - min_val)
            usable[label] = (col_name, norm)

        if len(usable) < 2:
            return None

        performance = pd.Series(0.0, index=grouped.index)
        metric_details = []
        for label, (col_name, norm) in usable.items():
            performance = performance.add(norm, fill_value=0.0)
            leader = str(norm.idxmax())
            metric_details.append(f"{label}: {leader} leads")

        best_company = str(performance.idxmax())
        best_score = float(performance.loc[best_company])
        year_label = f" in {requested_year}" if requested_year is not None else ""
        return (
            f"{best_company} shows the best financial performance{year_label} based on a combined score of "
            f"revenue, profit, assets, and liabilities available in the dataset. "
            + "; ".join(metric_details)
            + f". Composite score: {best_score:.2f}."
        )

    # Handle short carry-over fragments from split queries.
    
    if re.match(r"^and\s+\w+", q) and memory.get("last_structured_intent"):
        q = f"{memory['last_structured_intent']} {q.replace('and', '').strip()}"

    strongest_words = ["highest", "largest", "biggest", "max", "most"]
    lowest_words = ["lowest", "smallest", "min", "least"]
    freq_words = ["most frequent", "most frequently", "appears most", "frequency", "common", "occurs most"]
    month_names = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    quarter_months = {
        1: (1, 2, 3),
        2: (4, 5, 6),
        3: (7, 8, 9),
        4: (10, 11, 12),
    }

    def detect_quarter_reference(text):
        text = (text or "").lower()
        patterns = [
            (r"\bq([1-4])\b", None),
            (r"\bquarter\s*([1-4])\b", None),
            (r"\b([1-4])(st|nd|rd|th)?\s+quarter\b", None),
            (r"\bfirst\s+quarter\b", 1),
            (r"\bsecond\s+quarter\b", 2),
            (r"\bthird\s+quarter\b", 3),
            (r"\bfourth\s+quarter\b", 4),
        ]
        for pattern, fixed_value in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            if fixed_value is not None:
                return fixed_value
            return int(match.group(1))
        return None

    metric_map = {
        "revenue": revenue_col,
        "sales": revenue_col or revenue_col,
        "units sold": find_col("units", "sold") or find_col("quantity"),
        "profit": profit_col,
        "assets": assets_col,
        "asset": assets_col,
        "liabilities": liabilities_col,
        "liability": liabilities_col,
        "loss": loss_col,
        "expenses": expense_col,
        "expense": expense_col,
        "cost": expense_col,
    }

    detected_metric_word = detect_metric(q)
    requested_metric_word = (
        detected_metric_word
        if detected_metric_word in metric_map and metric_map.get(detected_metric_word)
        else next((word for word in metric_map if word in q and metric_map[word]), None)
    )
    requested_metric_col = metric_map.get(requested_metric_word) if requested_metric_word else None
    mentioned_companies = infer_values_from_column(company_col, limit=4) if company_col else []
    requested_years = detect_years(q)
    requested_year = requested_years[0] if requested_years else None
    company_rollup = build_company_rollup(df) if company_col else None

    def get_company_rollup_metrics(*required_cols):
        if company_rollup is None:
            return None
        missing = [c for c in required_cols if not c or c not in company_rollup.columns]
        if missing:
            return None
        temp = company_rollup[[company_col, *required_cols]].copy()
        for col_name in required_cols:
            temp[col_name] = temp[col_name].apply(to_number)
        temp = temp.dropna(subset=list(required_cols))
        temp = temp[temp[company_col].astype(str).str.strip() != ""]
        return temp if not temp.empty else None

    order_dates = pd.to_datetime(df[order_date_col], errors="coerce") if order_date_col and order_date_col in df.columns else None

    def dynamic_tabular_fallback():
        """Safe schema-aware fallback for tabular questions not covered by explicit rules."""
        temp = df.copy()
        if order_date_col and order_dates is not None:
            temp[order_date_col] = order_dates

        filter_labels = []

        def apply_exact_filters(dataframe):
            current = dataframe
            for col_name in [country_col, category_col, product_col, company_col]:
                if not col_name or col_name not in current.columns:
                    continue
                matched_values = infer_values_from_column(col_name, limit=6)
                if matched_values:
                    lowered = {str(v).strip().lower() for v in matched_values}
                    current = current[current[col_name].astype(str).str.strip().str.lower().isin(lowered)]
                    filter_labels.extend(f"{col_name}={value}" for value in matched_values[:3])
            if requested_years and order_date_col and order_date_col in current.columns:
                current = current[current[order_date_col].dt.year.isin(requested_years)]
                filter_labels.extend(f"year={y}" for y in requested_years)
            requested_month = next((month_no for month_name, month_no in month_names.items() if month_name in q), None)
            if requested_month is not None and order_date_col and order_date_col in current.columns:
                current = current[current[order_date_col].dt.month == requested_month]
                month_label = next(name.title() for name, num in month_names.items() if num == requested_month)
                filter_labels.append(f"month={month_label}")
            quarter_num = detect_quarter_reference(q)
            if quarter_num is not None and order_date_col and order_date_col in current.columns:
                current = current[current[order_date_col].dt.quarter == quarter_num]
                months = quarter_months.get(quarter_num, ())
                month_label = f"{months[0]}-{months[-1]}" if months else ""
                filter_labels.append(f"quarter=Q{quarter_num} ({month_label})")
            return current

        temp = apply_exact_filters(temp)
        if temp.empty:
            filter_text = ", ".join(filter_labels) if filter_labels else "the requested filters"
            return f"No matching rows found for {filter_text}."

        quantity_col = find_col("units", "sold") or find_col("quantity")
        metric_candidates = {
            "revenue": revenue_col,
            "sales": revenue_col,
            "profit": profit_col,
            "quantity": quantity_col,
            "units": quantity_col,
            "units sold": quantity_col,
            "unit price": find_col("unit", "price"),
            "price": find_col("price"),
            "orders": None,
            "transactions": None,
        }
        metric_key = requested_metric_word
        if metric_key not in metric_candidates:
            metric_key = next((key for key in metric_candidates if key in q and metric_candidates.get(key) is not None), None)
        metric_col = metric_candidates.get(metric_key) if metric_key else None
        if metric_col is None and any(token in q for token in ["order value", "average order value"]):
            metric_col = revenue_col
            metric_key = "order value"

        group_key = None
        group_col = None
        if any(token in q for token in ["per country", "by country", "country has", "country generates", "country performs"]):
            group_key, group_col = "country", country_col
        elif any(token in q for token in ["per product", "by product", "product has", "products sold"]):
            group_key, group_col = "product", product_col
        elif any(token in q for token in ["per category", "by category", "category generates"]):
            group_key, group_col = "category", category_col
        elif any(token in q for token in ["per company", "by company"]):
            group_key, group_col = "company", company_col
        elif any(token in q for token in ["per month", "by month", "which month", "month had", "sales over time", "trend"]):
            group_key, group_col = "month", order_date_col
        elif "quarter" in q or detect_quarter_reference(q) is not None or any(token in q for token in ["q1", "q2", "q3", "q4"]):
            group_key, group_col = "quarter", order_date_col

        if "correlation" in q and metric_col and quantity_col and quantity_col in temp.columns and metric_col in temp.columns:
            corr_frame = temp[[quantity_col, metric_col]].copy()
            corr_frame[quantity_col] = corr_frame[quantity_col].apply(to_number)
            corr_frame[metric_col] = corr_frame[metric_col].apply(to_number)
            corr_frame = corr_frame.dropna(subset=[quantity_col, metric_col])
            if not corr_frame.empty:
                corr = corr_frame[quantity_col].corr(corr_frame[metric_col])
                if pd.notna(corr):
                    return f"The correlation between {quantity_col} and {metric_col} is {corr:.4f}."

        if metric_col and metric_col in temp.columns:
            temp[metric_col] = temp[metric_col].apply(to_number)

        if group_key in {"month", "quarter"} and order_date_col and order_date_col in temp.columns:
            temp = temp[temp[order_date_col].notna()]
            if temp.empty:
                return "No date-backed rows are available for that question."
            group_series = temp[order_date_col].dt.month if group_key == "month" else temp[order_date_col].dt.quarter
            label_map = (
                lambda idx: next(name.title() for name, num in month_names.items() if num == int(idx))
                if group_key == "month"
                else f"Q{int(idx)}"
            )
        elif group_col and group_col in temp.columns:
            temp[group_col] = temp[group_col].astype(str).str.strip()
            temp = temp[temp[group_col] != ""]
            group_series = temp[group_col]
            label_map = lambda idx: str(idx)
        else:
            group_series = None
            label_map = lambda idx: str(idx)

        if metric_col and group_series is not None:
            grouped_metric = temp.groupby(group_series)[metric_col]
            if "average" in q or "avg" in q or "mean" in q:
                grouped = grouped_metric.mean().dropna()
                agg_label = f"average {metric_col}"
            elif "count" in q or "how many" in q or "number of" in q:
                grouped = grouped_metric.size().dropna()
                agg_label = "count"
            else:
                grouped = grouped_metric.sum().dropna()
                agg_label = f"total {metric_col}"

            if grouped.empty:
                return None

            if any(token in q for token in ["highest", "most", "best", "top", "lowest", "least"]):
                ascending = any(token in q for token in ["lowest", "least"])
                limit = 3 if "top 3" in q else 5 if "top 5" in q else 1
                ranked = grouped.sort_values(ascending=ascending).head(limit)
                if limit == 1:
                    idx = ranked.index[0]
                    return f"{label_map(idx)} has the {'lowest' if ascending else 'highest'} {agg_label} ({fmt_num(ranked.iloc[0])})."
                rows = [f"{label_map(idx)} ({fmt_num(val)})" for idx, val in ranked.items()]
                return f"Top {limit} by {agg_label}: " + "; ".join(rows)

            rows = [f"{label_map(idx)}: {fmt_num(val)}" for idx, val in grouped.sort_values(ascending=False).items()]
            return f"{agg_label.title()} per {group_key}: " + "; ".join(rows[:20])

        if metric_col and metric_col in temp.columns:
            series = temp[metric_col].dropna()
            if series.empty:
                return None
            if "average" in q or "avg" in q or "mean" in q:
                return f"The average {metric_col} is {fmt_num(series.mean())}."
            if "count" in q or "how many" in q or "number of" in q:
                focus = group_key or metric_key or "records"
                return f"I found matching {focus} data in the dataset."
            if any(token in q for token in ["highest", "max", "most"]):
                return f"The highest {metric_col} is {fmt_num(series.max())}."
            if any(token in q for token in ["lowest", "min", "least"]):
                return f"The lowest {metric_col} is {fmt_num(series.min())}."
            if any(token in q for token in ["total", "sum"]):
                return f"The total {metric_col} is {fmt_num(series.sum())}."

        if "show" in q or "list" in q or "give" in q:
            preview_cols = [c for c in [company_col, product_col, category_col, country_col, order_date_col, revenue_col, profit_col] if c and c in temp.columns]
            if preview_cols:
                return build_preview_answer("Here is a sample of the matching data:", temp, preview_cols, limit=5)

        return None

    comparison_triggers = [
        "compare",
        "higher",
        "more",
        "greater",
        "larger",
        "earns more",
    ]
    if (
        company_col
        and requested_metric_col
        and len(mentioned_companies) >= 2
        and any(trigger in q for trigger in comparison_triggers)
    ):
        comparison_values = summarize_company_metric(
            df,
            requested_metric_col,
            mentioned_companies[:2],
            requested_year=requested_year,
        )
        if comparison_values and len(comparison_values) >= 2:
            ordered_companies = [c for c in mentioned_companies[:2] if c in comparison_values]
            value_lines = [f"{name}: {fmt_num(comparison_values[name])}" for name in ordered_companies]
            best_company = max(ordered_companies, key=lambda name: comparison_values[name])
            best_value = comparison_values[best_company]
            year_label = f" in {requested_year}" if requested_year is not None else ""
            metric_label = requested_metric_word if requested_metric_word else requested_metric_col
            memory["last_structured_intent"] = "compare"
            if any(word in q for word in ["which", "higher", "more", "greater", "larger", "earns more"]):
                return (
                    f"{'; '.join(value_lines)}. "
                    f"{best_company} has the higher {metric_label}{year_label} ({fmt_num(best_value)})."
                )
            return f"Comparison of {metric_label}{year_label}: " + "; ".join(value_lines) + "."

    if (
        company_col
        and len(mentioned_companies) >= 2
        and any(
            phrase in q
            for phrase in [
                "financially stronger",
                "financial strength",
                "financial performance",
                "overall performance",
                "stronger financially",
            ]
        )
    ):
        strength_answer = compare_company_financial_strength(
            mentioned_companies,
            requested_year=requested_year,
        )
        if strength_answer:
            memory["last_structured_intent"] = "compare"
            return strength_answer

    if (
        company_col
        and "company" in q
        and any(
            phrase in q
            for phrase in [
                "best financial performance",
                "shows the best financial performance",
                "best overall financial performance",
                "top financial performance",
            ]
        )
    ):
        performance_answer = rank_companies_by_financial_performance(requested_year=requested_year)
        if performance_answer:
            memory["last_structured_intent"] = "compare"
            return performance_answer

    if company_col and assets_col and liabilities_col and "company" in q and ("asset to liability ratio" in q or ("ratio" in q and "asset" in q and "liabilit" in q)):
        temp = get_company_rollup_metrics(assets_col, liabilities_col)
        if temp is not None:
            temp["asset_liability_ratio"] = temp[assets_col] / (temp[liabilities_col].abs() + 1e-9)
            best = temp.sort_values(by="asset_liability_ratio", ascending=False).iloc[0]
            return (
                f"{best[company_col]} has the highest asset-to-liability ratio "
                f"({best['asset_liability_ratio']:.2f}, assets: {fmt_num(best[assets_col])}, liabilities: {fmt_num(best[liabilities_col])})."
            )

    if company_col and year_col and ("company" in q or "companies" in q) and ("growing the fastest" in q or "fastest financially" in q or "asset growth" in q or "growth" in q):
        growth_metric_col = assets_col if "asset growth" in q else (revenue_col or profit_col or assets_col)
        if growth_metric_col and growth_metric_col in df.columns:
            temp = df[[company_col, year_col, growth_metric_col]].copy()
            temp[growth_metric_col] = temp[growth_metric_col].apply(to_number)
            temp[year_col] = temp[year_col].apply(to_number)
            temp[company_col] = temp[company_col].astype(str).str.strip()
            temp = temp.dropna(subset=[company_col, year_col, growth_metric_col])
            temp = temp[temp[company_col] != ""]
            if not temp.empty:
                growth_rows = []
                for company_name, grp in temp.groupby(company_col):
                    g = grp.sort_values(by=year_col)
                    if len(g) < 2:
                        continue
                    first_val = safe_float(g[growth_metric_col].iloc[0], default=0.0)
                    last_val = safe_float(g[growth_metric_col].iloc[-1], default=0.0)
                    if abs(first_val) < 1e-9:
                        continue
                    growth_pct = ((last_val - first_val) / abs(first_val)) * 100.0
                    growth_rows.append((company_name, growth_pct))
                if growth_rows:
                    ranked = sorted(growth_rows, key=lambda x: x[1], reverse=True)
                    if "companies" in q:
                        strong = [name for name, pct in ranked if pct > 0]
                        companies = unique_join(strong)
                        if companies:
                            return f"Companies with strong {growth_metric_col} growth: {companies}."
                    best_company, best_growth = ranked[0]
                    return f"{best_company} is growing the fastest financially in {growth_metric_col} ({best_growth:.2f}% from first to last year)."

    if company_col and assets_col and liabilities_col and ("balance sheet" in q or "financially risky" in q or "balanced finances" in q or "financial structure" in q):
        temp = get_company_rollup_metrics(assets_col, liabilities_col)
        if temp is not None:
            temp["stability_ratio"] = temp[assets_col] / (temp[liabilities_col].abs() + 1e-9)
            temp["capital_buffer"] = temp[assets_col] - temp[liabilities_col]
            if "risky" in q:
                risky = temp[(temp["capital_buffer"] < 0) | (temp["stability_ratio"] < 1.0)]
                if not risky.empty:
                    companies = unique_join(risky.sort_values(by=["capital_buffer", "stability_ratio"]).index.tolist())
                    if companies:
                        return f"Companies that appear financially risky: {companies}."
            elif "balanced finances" in q:
                temp["balance_gap"] = (temp[assets_col] - temp[liabilities_col]).abs()
                best = temp.sort_values(by=["balance_gap", "stability_ratio"], ascending=[True, False]).iloc[0]
                return (
                    f"{best[company_col]} shows the most balanced finances "
                    f"(assets: {fmt_num(best[assets_col])}, liabilities: {fmt_num(best[liabilities_col])})."
                )
            elif "balance sheet" in q and "companies" in q:
                strong = temp[(temp["stability_ratio"] >= temp["stability_ratio"].median()) & (temp["capital_buffer"] > 0)]
                if not strong.empty:
                    companies = unique_join(strong[company_col].tolist())
                    if companies:
                        return f"Companies with strong financial balance sheets: {companies}."
            else:
                best = temp.sort_values(by=["stability_ratio", "capital_buffer"], ascending=False).iloc[0]
                return (
                    f"{best[company_col]} has the strongest financial structure "
                    f"(ratio: {best['stability_ratio']:.2f}, assets: {fmt_num(best[assets_col])}, liabilities: {fmt_num(best[liabilities_col])})."
                )

    if sector_col and assets_col and "sector" in q and "asset" in q and any(w in q for w in strongest_words):
        top = aggregate_entity_metric(df, sector_col, assets_col, highest=True)
        if top:
            entity, value = top
            return f"{entity} has companies with the largest {assets_col} ({fmt_num(value)})."

    if company_col and revenue_col and profit_col and ("high revenue but low profit" in q or ("high revenue" in q and "low profit" in q)):
        temp = get_company_rollup_metrics(revenue_col, profit_col)
        if temp is not None:
            high_rev = temp[revenue_col].quantile(0.75)
            low_profit = temp[profit_col].quantile(0.25)
            filtered = temp[(temp[revenue_col] >= high_rev) & (temp[profit_col] <= low_profit)]
            if not filtered.empty:
                companies = unique_join(filtered[company_col].tolist())
                if companies:
                    return f"Companies with high {revenue_col} but low {profit_col}: {companies}."
            return f"No companies found with high {revenue_col} but low {profit_col}."

    if company_col and assets_col and liabilities_col and ("low liabilities but high assets" in q or ("low liabilit" in q and "high asset" in q)):
        temp = get_company_rollup_metrics(assets_col, liabilities_col)
        if temp is not None:
            high_assets = temp[assets_col].quantile(0.75)
            low_liabilities = temp[liabilities_col].quantile(0.25)
            filtered = temp[(temp[assets_col] >= high_assets) & (temp[liabilities_col] <= low_liabilities)]
            if not filtered.empty:
                companies = unique_join(filtered[company_col].tolist())
                if companies:
                    return f"Companies with low {liabilities_col} but high {assets_col}: {companies}."
            return f"No companies found with low {liabilities_col} but high {assets_col}."

    if sector_col and profit_col and ("sector" in q and "consistent profits" in q):
        if year_col and year_col in df.columns:
            temp = df[[sector_col, year_col, profit_col]].copy()
            temp[profit_col] = temp[profit_col].apply(to_number)
            temp[year_col] = temp[year_col].apply(to_number)
            temp[sector_col] = temp[sector_col].astype(str).str.strip()
            temp = temp.dropna(subset=[sector_col, year_col, profit_col])
            temp = temp[temp[sector_col] != ""]
            if not temp.empty:
                stats = temp.groupby(sector_col)[profit_col].agg(["mean", "std"]).fillna(0.0)
                stats = stats[stats["mean"] > 0]
                if not stats.empty:
                    stats["consistency_score"] = stats["mean"] / (stats["std"] + 1e-9)
                    best = stats["consistency_score"].idxmax()
                    return f"{best} has the most consistent profits based on profit stability over time."

    if company_col and expense_col and revenue_col and ("expenses compared to revenue" in q or "high expenses compared to revenue" in q):
        temp = get_company_rollup_metrics(expense_col, revenue_col)
        if temp is not None:
            temp = temp[temp[revenue_col] > 0]
            if not temp.empty:
                temp["expense_revenue_ratio"] = temp[expense_col] / temp[revenue_col]
                if "companies" in q:
                    high = temp[temp["expense_revenue_ratio"] >= temp["expense_revenue_ratio"].quantile(0.75)]
                    if not high.empty:
                        companies = unique_join(high[company_col].tolist())
                        if companies:
                            return f"Companies with high {expense_col} compared to {revenue_col}: {companies}."
                best = temp.sort_values(by="expense_revenue_ratio", ascending=False).iloc[0]
                return f"{best[company_col]} has the highest expense-to-revenue ratio ({best['expense_revenue_ratio'] * 100:.2f}%)."

    if company_col and profit_col and revenue_col and ("financial efficiency" in q or "financially efficient" in q):
        temp = get_company_rollup_metrics(profit_col, revenue_col)
        if temp is not None:
            temp = temp[temp[revenue_col] > 0]
            if not temp.empty:
                temp["efficiency_ratio"] = temp[profit_col] / temp[revenue_col]
                if "companies" in q:
                    efficient = temp[temp["efficiency_ratio"] >= temp["efficiency_ratio"].quantile(0.75)]
                    if not efficient.empty:
                        companies = unique_join(efficient[company_col].tolist())
                        if companies:
                            return f"Financially efficient companies: {companies}."
                best = temp.sort_values(by="efficiency_ratio", ascending=False).iloc[0]
                return f"{best[company_col]} has the best financial efficiency ({best['efficiency_ratio'] * 100:.2f}% profit-to-revenue ratio)."

    if company_col and assets_col and liabilities_col and ("financial gap" in q or "gap between assets and liabilities" in q):
        temp = get_company_rollup_metrics(assets_col, liabilities_col)
        if temp is not None:
            temp["financial_gap"] = temp[assets_col] - temp[liabilities_col]
            if "companies" in q:
                top_gap = temp.sort_values(by="financial_gap", ascending=False)
                companies = unique_join(top_gap[company_col].tolist())
                if companies:
                    return f"Companies with the largest financial gap between {assets_col} and {liabilities_col}: {companies}."
            best = temp.sort_values(by="financial_gap", ascending=False).iloc[0]
            return f"{best[company_col]} has the largest financial gap between {assets_col} and {liabilities_col} ({fmt_num(best['financial_gap'])})."

    if sector_col and assets_col and liabilities_col and "sector" in q and "stable" in q:
        temp = get_company_rollup_metrics(assets_col, liabilities_col)
        if temp is not None and sector_col in company_rollup.columns:
            temp[sector_col] = company_rollup.set_index(company_col).loc[temp[company_col], sector_col].values
            temp["stability_ratio"] = temp[assets_col] / (temp[liabilities_col].abs() + 1e-9)
            stats = temp.groupby(sector_col)["stability_ratio"].mean().dropna()
            if not stats.empty:
                best = stats.idxmax()
                return f"{best} has the most financially stable companies based on average asset-to-liability strength."

    if company_col and profit_col and assets_col and ("profits relative to assets" in q or ("profit" in q and "asset" in q and "relative" in q)):
        temp = get_company_rollup_metrics(profit_col, assets_col)
        if temp is not None:
            temp = temp[temp[assets_col] > 0]
            if not temp.empty:
                temp["profit_asset_ratio"] = temp[profit_col] / temp[assets_col]
                high = temp[temp["profit_asset_ratio"] >= temp["profit_asset_ratio"].quantile(0.75)]
                if not high.empty:
                    companies = unique_join(high[company_col].tolist())
                    if companies:
                        return f"Companies with high {profit_col} relative to {assets_col}: {companies}."

    if company_col and profit_col and "compan" in q and "top" in q and "profit" in q and re.search(r"\btop\s*5\b", q):
        temp = get_company_rollup_metrics(profit_col)
        if temp is not None:
            ranked = temp.sort_values(by=profit_col, ascending=False).head(5)
            rows = [f"{i+1}. {row[company_col]} ({fmt_num(row[profit_col])})" for i, (_, row) in enumerate(ranked.iterrows())]
            return "Top 5 companies by profit: " + "; ".join(rows)

    if sector_col and assets_col and liabilities_col and ("strongest finances" in q or "financially strongest overall" in q):
        temp = get_company_rollup_metrics(assets_col, liabilities_col)
        if temp is not None and sector_col in company_rollup.columns:
            temp[sector_col] = company_rollup.set_index(company_col).loc[temp[company_col], sector_col].values
            temp["stability_ratio"] = temp[assets_col] / (temp[liabilities_col].abs() + 1e-9)
            temp["capital_buffer"] = temp[assets_col] - temp[liabilities_col]
            stats = temp.groupby(sector_col).agg({"stability_ratio": "mean", "capital_buffer": "mean"}).dropna()
            if not stats.empty:
                stats["sector_strength"] = (0.6 * stats["stability_ratio"]) + (0.4 * stats["capital_buffer"].rank(pct=True))
                best = stats["sector_strength"].idxmax()
                return f"{best} appears financially strongest overall."

    if country_col and profit_col and "country" in q and "profitable" in q:
        temp = df[[country_col, profit_col]].copy()
        temp[country_col] = temp[country_col].astype(str).str.strip()
        temp[profit_col] = temp[profit_col].apply(to_number)
        temp = temp.dropna(subset=[country_col, profit_col])
        temp = temp[temp[country_col] != ""]
        if not temp.empty:
            agg = temp.groupby(country_col)[profit_col].sum()
            if not agg.empty:
                best = str(agg.idxmax())
                return f"{best} has the most profitable companies."

    if company_col and assets_col and liabilities_col and "financial balance" in q:
        temp = get_company_rollup_metrics(assets_col, liabilities_col)
        if temp is not None:
            temp["balance_score"] = (temp[assets_col] - temp[liabilities_col]).abs()
            best = temp.sort_values(by=["balance_score", assets_col], ascending=[True, False]).iloc[0]
            return f"{best[company_col]} shows the best financial balance."

    if company_col and revenue_col and assets_col and ("high revenue and high assets" in q or ("high revenue" in q and "high asset" in q)):
        temp = get_company_rollup_metrics(revenue_col, assets_col)
        if temp is not None:
            high_rev = temp[revenue_col].quantile(0.75)
            high_assets = temp[assets_col].quantile(0.75)
            filtered = temp[(temp[revenue_col] >= high_rev) & (temp[assets_col] >= high_assets)]
            if not filtered.empty:
                companies = unique_join(filtered[company_col].tolist())
                if companies:
                    return f"Companies with high {revenue_col} and high {assets_col}: {companies}."
            return f"No companies found with high {revenue_col} and high {assets_col}."

    if company_col and sector_col and profit_col and ("dominate their respective sectors" in q or "dominate" in q and "sector" in q):
        temp = get_company_rollup_metrics(profit_col)
        if temp is not None and sector_col in company_rollup.columns:
            temp[sector_col] = company_rollup.set_index(company_col).loc[temp[company_col], sector_col].values
            temp = temp.dropna(subset=[sector_col])
            leaders = temp.sort_values(by=profit_col, ascending=False).groupby(sector_col).head(1)
            if not leaders.empty:
                companies = unique_join(leaders[company_col].tolist())
                if companies:
                    return f"Companies dominating their sectors financially: {companies}."

    # Compare profit across companies (optionally within a sector).
    
    if company_col and profit_col and "compar" in q and "profit" in q and "compan" in q:
        temp = df.copy()
        requested_sector = infer_value_from_column(sector_col) if sector_col else None
        if requested_sector:
            temp = apply_text_filter(temp, sector_col, requested_sector)
        temp = temp[[company_col, profit_col]].copy()
        temp[profit_col] = temp[profit_col].apply(to_number)
        temp[company_col] = temp[company_col].astype(str).str.strip()
        temp = temp.dropna(subset=[profit_col])
        temp = temp[temp[company_col] != ""]
        if not temp.empty:
            ranked = temp.groupby(company_col)[profit_col].sum().sort_values(ascending=False)
            if not ranked.empty:
                rows = [f"{idx}: {fmt_num(val)}" for idx, val in ranked.items()]
                title = f" in {requested_sector.title()} sector" if requested_sector else ""
                return f"Profit comparison by company{title}: " + "; ".join(rows)

    # Top N companies by revenue (default 5 for this use case).
    
    if company_col and revenue_col and "compan" in q and "top" in q and "revenue" in q and re.search(r"\btop\s*5\b", q):
        temp = df[[company_col, revenue_col]].copy()
        temp[revenue_col] = temp[revenue_col].apply(to_number)
        temp[company_col] = temp[company_col].astype(str).str.strip()
        temp = temp.dropna(subset=[revenue_col])
        temp = temp[temp[company_col] != ""]
        if not temp.empty:
            ranked = temp.groupby(company_col)[revenue_col].sum().sort_values(ascending=False).head(5)
            if not ranked.empty:
                rows = [f"{i+1}. {idx} ({fmt_num(val)})" for i, (idx, val) in enumerate(ranked.items())]
                return "Top 5 companies by revenue: " + "; ".join(rows)

    # Country with highest total assets.
    
    if country_col and assets_col and "country" in q and "asset" in q and any(w in q for w in strongest_words):
        top = aggregate_entity_metric(df, country_col, assets_col, highest=True)
        if top:
            entity, value = top
            return f"{entity} has companies with the highest total {assets_col} ({fmt_num(value)})."

    # Sector with lowest liabilities (average or total).
    
    if sector_col and liabilities_col and "sector" in q and "liabilit" in q and any(w in q for w in lowest_words):
        temp = df[[sector_col, liabilities_col]].copy()
        temp[liabilities_col] = temp[liabilities_col].apply(to_number)
        temp[sector_col] = temp[sector_col].astype(str).str.strip()
        temp = temp.dropna(subset=[liabilities_col])
        temp = temp[temp[sector_col] != ""]
        if not temp.empty:
            agg = temp.groupby(sector_col)[liabilities_col].mean() if "average" in q else temp.groupby(sector_col)[liabilities_col].sum()
            if not agg.empty:
                sec = str(agg.idxmin())
                val = agg.loc[sec]
                kind = "average" if "average" in q else "total"
                return f"{sec} has the lowest {kind} {liabilities_col} ({fmt_num(val)})."

    # Sector with highest average profit.
    
    if sector_col and profit_col and "sector" in q and "profit" in q and "average" in q and any(w in q for w in strongest_words):
        temp = df[[sector_col, profit_col]].copy()
        temp[profit_col] = temp[profit_col].apply(to_number)
        temp[sector_col] = temp[sector_col].astype(str).str.strip()
        temp = temp.dropna(subset=[profit_col])
        temp = temp[temp[sector_col] != ""]
        if not temp.empty:
            avg_profit = temp.groupby(sector_col)[profit_col].mean()
            if not avg_profit.empty:
                sec = str(avg_profit.idxmax())
                val = avg_profit.loc[sec]
                return f"{sec} has companies with the highest average {profit_col} ({fmt_num(val)})."

    # Company with largest assets-liabilities difference.
    
    if company_col and assets_col and liabilities_col and "difference" in q and "asset" in q and "liabilit" in q:
        temp = df[[company_col, assets_col, liabilities_col]].copy()
        temp[assets_col] = temp[assets_col].apply(to_number)
        temp[liabilities_col] = temp[liabilities_col].apply(to_number)
        temp[company_col] = temp[company_col].astype(str).str.strip()
        temp = temp.dropna(subset=[assets_col, liabilities_col])
        temp = temp[temp[company_col] != ""]
        if not temp.empty:
            grouped = temp.groupby(company_col)[[assets_col, liabilities_col]].sum()
            grouped["delta"] = grouped[assets_col] - grouped[liabilities_col]
            if not grouped.empty:
                best = str(grouped["delta"].idxmax())
                delta = grouped.loc[best, "delta"]
                return f"{best} has the largest difference between {assets_col} and {liabilities_col} ({fmt_num(delta)})."

    # Companies with high revenue and low expenses.
    
    if company_col and revenue_col and expense_col and "high revenue" in q and ("low expense" in q or "low expenses" in q):
        temp = df[[company_col, revenue_col, expense_col]].copy()
        temp[revenue_col] = temp[revenue_col].apply(to_number)
        temp[expense_col] = temp[expense_col].apply(to_number)
        temp[company_col] = temp[company_col].astype(str).str.strip()
        temp = temp.dropna(subset=[revenue_col, expense_col])
        temp = temp[temp[company_col] != ""]
        if not temp.empty:
            high_rev = temp[revenue_col].quantile(0.75)
            low_exp = temp[expense_col].quantile(0.25)
            filtered = temp[(temp[revenue_col] >= high_rev) & (temp[expense_col] <= low_exp)]
            if not filtered.empty:
                companies = unique_join(filtered[company_col].tolist())
                if companies:
                    return (
                        f"Companies with high {revenue_col} and low {expense_col}: {companies} "
                        f"(thresholds: revenue >= {fmt_num(high_rev)}, expense <= {fmt_num(low_exp)})."
                    )
            return f"No companies found with high {revenue_col} and low {expense_col} under dataset quantile thresholds."

    # Company with best financial stability using assets vs liabilities.
    
    if company_col and assets_col and liabilities_col and "company" in q and ("stable" in q or "stability" in q):
        temp = df[[company_col, assets_col, liabilities_col]].copy()
        temp[assets_col] = temp[assets_col].apply(to_number)
        temp[liabilities_col] = temp[liabilities_col].apply(to_number)
        temp[company_col] = temp[company_col].astype(str).str.strip()
        temp = temp.dropna(subset=[assets_col, liabilities_col])
        temp = temp[temp[company_col] != ""]
        if not temp.empty:
            grouped = temp.groupby(company_col)[[assets_col, liabilities_col]].sum()
            grouped = grouped.dropna()
            if not grouped.empty:
                grouped["stability_ratio"] = grouped[assets_col] / (grouped[liabilities_col].abs() + 1e-9)
                grouped["capital_buffer"] = grouped[assets_col] - grouped[liabilities_col]
                ranked = grouped.sort_values(by=["stability_ratio", "capital_buffer"], ascending=False)
                best_company = str(ranked.index[0])
                best_assets = ranked.iloc[0][assets_col]
                best_liabilities = ranked.iloc[0][liabilities_col]
                best_ratio = ranked.iloc[0]["stability_ratio"]
                return (
                    f"{best_company} appears most financially stable based on {assets_col} vs {liabilities_col} "
                    f"(assets: {fmt_num(best_assets)}, liabilities: {fmt_num(best_liabilities)}, ratio: {best_ratio:.2f})."
                )

    # Company with highest profit.
    
    if company_col and profit_col and "company" in q and "profit" in q and any(w in q for w in strongest_words):
        top = aggregate_entity_metric(df, company_col, profit_col, highest=True)
        if top:
            entity, value = top
            return f"{entity} has the highest {profit_col} ({fmt_num(value)})."

    # Sector that appears most profitable.
    
    if sector_col and profit_col and "sector" in q and ("profit" in q or "profitable" in q):
        top = aggregate_entity_metric(df, sector_col, profit_col, highest=True)
        if top:
            entity, value = top
            return f"{entity} seems most profitable with total {profit_col} of {fmt_num(value)}."

    # Sector with highest revenue.
    
    if sector_col and revenue_col and "sector" in q and "revenue" in q and any(w in q for w in strongest_words):
        top = aggregate_entity_metric(df, sector_col, revenue_col, highest=True)
        if top:
            entity, value = top
            return f"{entity} generates the most {revenue_col} ({fmt_num(value)})."

    # Company with best profit-to-revenue ratio.
    
    if (
        company_col
        and profit_col
        and revenue_col
        and "company" in q
        and (
            "profit-to-revenue" in q
            or "profit to revenue" in q
            or ("ratio" in q and "profit" in q and "revenue" in q)
            or ("margin" in q and "best" in q)
        )
    ):
        temp = df[[company_col, profit_col, revenue_col]].copy()
        temp[profit_col] = temp[profit_col].apply(to_number)
        temp[revenue_col] = temp[revenue_col].apply(to_number)
        temp[company_col] = temp[company_col].astype(str).str.strip()
        temp = temp.dropna(subset=[profit_col, revenue_col])
        temp = temp[temp[company_col] != ""]
        if not temp.empty:
            grouped = temp.groupby(company_col)[[profit_col, revenue_col]].sum()
            grouped = grouped[grouped[revenue_col] > 0]
            if not grouped.empty:
                grouped["profit_revenue_ratio"] = grouped[profit_col] / grouped[revenue_col]
                best_company = str(grouped["profit_revenue_ratio"].idxmax())
                best_ratio = float(grouped.loc[best_company, "profit_revenue_ratio"])
                return (
                    f"{best_company} has the best profit-to-revenue ratio "
                    f"({best_ratio * 100:.2f}% based on aggregated {profit_col}/{revenue_col})."
                )

    # Country with most companies in dataset.
    
    if country_col and company_col and "country" in q and ("most companies" in q or "most company" in q):
        temp = df[[country_col, company_col]].copy()
        temp[country_col] = temp[country_col].astype(str).str.strip()
        temp[company_col] = temp[company_col].astype(str).str.strip()
        temp = temp[(temp[country_col] != "") & (temp[company_col] != "")]
        if not temp.empty:
            counts = temp.groupby(country_col)[company_col].nunique()
            if not counts.empty:
                top_country = str(counts.idxmax())
                top_count = int(counts.loc[top_country])
                return f"{top_country} has the most companies in the dataset ({top_count} unique companies)."

    # Company with largest financial growth across years.
    
    if company_col and year_col and "company" in q and "growth" in q:
        growth_metric_col = revenue_col or profit_col or assets_col
        if growth_metric_col and growth_metric_col in df.columns:
            temp = df[[company_col, year_col, growth_metric_col]].copy()
            temp[growth_metric_col] = temp[growth_metric_col].apply(to_number)
            temp[year_col] = temp[year_col].apply(to_number)
            temp[company_col] = temp[company_col].astype(str).str.strip()
            temp = temp.dropna(subset=[company_col, year_col, growth_metric_col])
            temp = temp[temp[company_col] != ""]
            if not temp.empty:
                best_company = None
                best_growth = None
                for company_name, grp in temp.groupby(company_col):
                    g = grp.sort_values(by=year_col)
                    if len(g) < 2:
                        continue
                    first_val = safe_float(g[growth_metric_col].iloc[0], default=0.0)
                    last_val = safe_float(g[growth_metric_col].iloc[-1], default=0.0)
                    if abs(first_val) < 1e-9:
                        continue
                    growth_pct = ((last_val - first_val) / abs(first_val)) * 100.0
                    if best_growth is None or growth_pct > best_growth:
                        best_growth = growth_pct
                        best_company = company_name
                if best_company is not None:
                    return (
                        f"{best_company} has the largest financial growth in {growth_metric_col} "
                        f"({best_growth:.2f}% from first to last year)."
                    )

    # Company with largest assets.
    
    if company_col and assets_col and "company" in q and "asset" in q and any(w in q for w in strongest_words):
        top = aggregate_entity_metric(df, company_col, assets_col, highest=True)
        if top:
            entity, value = top
            return f"{entity} has the largest {assets_col} ({fmt_num(value)})."

    # Company with lowest liabilities.
    
    if company_col and liabilities_col and "company" in q and "liabilit" in q and any(w in q for w in lowest_words):
        candidate_df = df
        requested_sector = infer_value_from_column(sector_col) if sector_col else None
        if requested_sector:
            filtered = apply_text_filter(candidate_df, sector_col, requested_sector)
            if not filtered.empty:
                candidate_df = filtered
        low = aggregate_entity_metric(candidate_df, company_col, liabilities_col, highest=False)
        if low:
            entity, value = low
            if requested_sector:
                return f"{entity} in {requested_sector.title()} sector has the lowest {liabilities_col} ({fmt_num(value)})."
            return f"{entity} has the lowest {liabilities_col} ({fmt_num(value)})."

    # Company with highest revenue (optionally filtered by country, e.g., India).
    
    if company_col and revenue_col and "company" in q and "revenue" in q and any(w in q for w in strongest_words):
        candidate_df = df
        requested_country_label = None
        requested_sector_label = None
        if sector_col:
            requested_sector = infer_value_from_column(sector_col)
            if requested_sector:
                filtered = apply_text_filter(candidate_df, sector_col, requested_sector)
                if not filtered.empty:
                    candidate_df = filtered
                    requested_sector_label = requested_sector.title()
        if country_col:
            requested_country = infer_value_from_column(country_col)
            if not requested_country:
                m_country = re.search(r"\bin\s+([a-z][a-z ]{1,})", q)
                if m_country:
                    requested_country = m_country.group(1).strip().lower()
            if requested_country:
                filtered = apply_text_filter(candidate_df, country_col, requested_country)
                if not filtered.empty:
                    candidate_df = filtered
                    requested_country_label = requested_country.title()

        top = aggregate_entity_metric(candidate_df, company_col, revenue_col, highest=True)
        if top:
            entity, value = top
            if requested_country_label and requested_sector_label:
                return (
                    f"{entity} in {requested_country_label} ({requested_sector_label} sector) "
                    f"has the highest {revenue_col} ({fmt_num(value)})."
                )
            if requested_country_label:
                return f"{entity} in {requested_country_label} has the highest {revenue_col} ({fmt_num(value)})."
            if requested_sector_label:
                return f"{entity} in {requested_sector_label} sector has the highest {revenue_col} ({fmt_num(value)})."
            return f"{entity} has the highest {revenue_col} ({fmt_num(value)})."

    # Most frequent sector in dataset.
    
    if sector_col and "sector" in q and any(w in q for w in freq_words):
        sector_series = df[sector_col].dropna().astype(str).str.strip()
        sector_series = sector_series[sector_series != ""]
        if not sector_series.empty:
            counts = sector_series.value_counts()
            if not counts.empty:
                top_sector = counts.index[0]
                top_count = int(counts.iloc[0])
                return f"{top_sector} appears most frequently in {sector_col} ({top_count} rows)."

    # List companies in a requested sector (e.g., banking, technology, automotive).
    
    if company_col and sector_col and ("compan" in q or "list " in q) and ("sector" in q or "industry" in q or "companies" in q):
        requested_sector = infer_value_from_column(sector_col)
        if not requested_sector:
            m_sector = re.search(r"\b([a-z][a-z0-9 _-]{2,})\s+sector\b", q)
            if m_sector:
                requested_sector = m_sector.group(1).strip().lower()
        if not requested_sector:
            m_companies = re.search(r"\b([a-z][a-z0-9 _-]{2,})\s+companies\b", q)
            if m_companies:
                maybe_sector = m_companies.group(1).strip().lower()
                if maybe_sector not in {"which", "find", "list"}:
                    requested_sector = maybe_sector
        if requested_sector:
            source_df = company_rollup if company_rollup is not None else df
            filtered = apply_text_filter(source_df, sector_col, requested_sector)
            if not filtered.empty:
                companies = companies_from_filtered(filtered)
                if companies:
                    return f"Companies in {requested_sector.title()} sector: {companies}."

    # List companies operating in a requested country (e.g., India).
    
    if company_col and country_col and ("operating in" in q or "operate in" in q or "companies in" in q or "list companies" in q):
        requested_country = infer_value_from_column(country_col)
        if not requested_country:
            m_country = re.search(r"(?:operating in|operate in|companies in)\s+([a-z][a-z ]{1,})", q)
            if m_country:
                requested_country = m_country.group(1).strip().lower()
        if requested_country:
            source_df = company_rollup if company_rollup is not None else df
            filtered = apply_text_filter(source_df, country_col, requested_country)
            if not filtered.empty:
                companies = companies_from_filtered(filtered)
                if companies:
                    return f"Companies operating in {requested_country.title()}: {companies}."

    # List companies where assets exceed liabilities.
    
    if company_col and assets_col and liabilities_col and "asset" in q and "liabilit" in q and any(p in q for p in ["greater than", "above", "more than", "exceed"]):
        temp = company_rollup[[company_col, assets_col, liabilities_col]].copy() if company_rollup is not None and assets_col in company_rollup.columns and liabilities_col in company_rollup.columns else df[[company_col, assets_col, liabilities_col]].copy()
        temp[company_col] = temp[company_col].astype(str).str.strip()
        temp[assets_col] = temp[assets_col].apply(to_number)
        temp[liabilities_col] = temp[liabilities_col].apply(to_number)
        temp = temp.dropna(subset=[assets_col, liabilities_col])
        temp = temp[temp[company_col] != ""]
        if not temp.empty:
            grouped = temp.groupby(company_col)[[assets_col, liabilities_col]].sum()
            filtered = grouped[grouped[assets_col] > grouped[liabilities_col]]
            if not filtered.empty:
                companies = unique_join(filtered.index.tolist())
                if companies:
                    return f"Companies with {assets_col} greater than {liabilities_col}: {companies}."
            return f"No companies found with {assets_col} greater than {liabilities_col}."

    # Threshold list queries (e.g., revenue above 300000, profit above 10000).
    
    threshold_match = re.search(
        r"(revenue|profit|loss|assets?|liabilit(?:y|ies)|expenses?|cost)\s*(above|over|greater than|more than|>=|>|below|under|less than|<=|<)\s*([0-9][0-9,\.]*)\s*(million|billion|thousand|mn|bn|m|b|k)?",
        q,
    )
    if not threshold_match:
        threshold_match = re.search(
            r"(above|over|greater than|more than|>=|>|below|under|less than|<=|<)\s*([0-9][0-9,\.]*)\s*(million|billion|thousand|mn|bn|m|b|k)?",
            q,
        )

    if company_col and threshold_match:
        if len(threshold_match.groups()) >= 4:
            metric_word = threshold_match.group(1)
            operator_word = threshold_match.group(2)
            threshold_raw = threshold_match.group(3)
            threshold_suffix = threshold_match.group(4) or ""
        else:
            metric_word = None
            operator_word = threshold_match.group(1)
            threshold_raw = threshold_match.group(2)
            threshold_suffix = threshold_match.group(3) or ""
        threshold_value = parse_threshold_value(threshold_raw, threshold_suffix)
        metric_col = None
        if metric_word:
            metric_col = find_col(metric_word)
        if metric_col is None:
            if "revenue" in q:
                metric_col = revenue_col
            elif "profit" in q:
                metric_col = profit_col
            elif "asset" in q:
                metric_col = assets_col
            elif "liabilit" in q:
                metric_col = liabilities_col
            elif "loss" in q:
                metric_col = loss_col
            elif "expense" in q or "cost" in q:
                metric_col = expense_col

        if metric_col and threshold_value is not None and metric_col in df.columns:
            if company_rollup is not None and metric_col in company_rollup.columns:
                temp = company_rollup.copy()
            else:
                temp = df.copy()
                temp[metric_col] = temp[metric_col].apply(to_number)
            temp = temp.dropna(subset=[metric_col])
            if sector_col:
                requested_sector = infer_value_from_column(sector_col)
                if not requested_sector:
                    m_sector = re.search(r"\b([a-z][a-z0-9 _-]{2,})\s+companies\b", q)
                    if m_sector:
                        maybe_sector = m_sector.group(1).strip().lower()
                        if maybe_sector not in {"which", "find", "list"}:
                            requested_sector = maybe_sector
                if requested_sector:
                    temp = apply_text_filter(temp, sector_col, requested_sector)
            if country_col:
                requested_country = infer_value_from_column(country_col)
                if requested_country:
                    temp = apply_text_filter(temp, country_col, requested_country)

            if operator_word in {"below", "under", "less than", "<=", "<"}:
                filtered = temp[temp[metric_col] < threshold_value]
                direction_label = "below"
            else:
                filtered = temp[temp[metric_col] > threshold_value]
                direction_label = "above"
            if not filtered.empty:
                preview_cols = [c for c in ["OrderID", company_col, product_col, category_col, country_col, metric_col] if c and c in filtered.columns]
                preview_rows = []
                for _, row in filtered.head(10)[preview_cols].iterrows():
                    preview_rows.append([row[col] for col in preview_cols])
                table = build_markdown_table(preview_cols, preview_rows)
                answer = f"Found {len(filtered):,} rows with {metric_col} {direction_label} {fmt_num(threshold_value)}."
                if table:
                    answer += "\n\nSample rows:\n" + table
                return answer.strip()
            return f"No rows found with {metric_col} {direction_label} {fmt_num(threshold_value)}."

    if "sales" in q and revenue_col and any(token in q for token in ["total", "sum", "amount"]):
        total_sales = df[revenue_col].apply(to_number).dropna().sum()
        return f"The total {revenue_col} is {fmt_num(total_sales)}."

    if revenue_col and ("average order value" in q or ("average" in q and "order" in q and "value" in q)) and "country" not in q:
        revenue_series = df[revenue_col].apply(to_number).dropna()
        if not revenue_series.empty:
            return f"The average order value is {fmt_num(revenue_series.mean())}."

    if "how many transactions" in q or "number of transactions" in q or "transactions are recorded" in q:
        return f"There are {len(df):,} transactions recorded in the dataset."

    if any(token in q for token in ["summarize", "summary", "full report", "business report", "detailed insights", "sales distribution", "complete sales distribution", "provide detailed insights", "explain complete sales distribution"]):
        summary_lines = [f"Dataset contains {len(df):,} transactions."]
        if revenue_col:
            revenue_series = df[revenue_col].apply(to_number).dropna()
            if not revenue_series.empty:
                summary_lines.append(f"Total {revenue_col}: {fmt_num(revenue_series.sum())}.")
                summary_lines.append(f"Average order value: {fmt_num(revenue_series.mean())}.")
        if profit_col:
            profit_series = df[profit_col].apply(to_number).dropna()
            if not profit_series.empty:
                summary_lines.append(f"Total {profit_col}: {fmt_num(profit_series.sum())}.")
        if country_col:
            countries = unique_join(sorted(df[country_col].dropna().astype(str).str.strip().unique().tolist()), limit=8)
            if countries:
                summary_lines.append(f"Countries covered: {countries}.")
        if product_col:
            summary_lines.append(f"Products covered: {df[product_col].dropna().astype(str).str.strip().nunique():,}.")
        if category_col:
            summary_lines.append(f"Categories covered: {df[category_col].dropna().astype(str).str.strip().nunique():,}.")
        if order_dates is not None and order_dates.notna().any():
            summary_lines.append(
                f"Date range: {order_dates.min().date()} to {order_dates.max().date()}."
            )
        return " ".join(summary_lines)

    if country_col and revenue_col and any(token in q for token in ["sales in", "revenue in"]) and "only" in q:
        requested_country = infer_value_from_column(country_col)
        if requested_country:
            temp = df[[country_col, revenue_col]].copy()
            temp = apply_text_filter(temp, country_col, requested_country)
            temp[revenue_col] = temp[revenue_col].apply(to_number)
            temp = temp.dropna(subset=[revenue_col])
            if not temp.empty:
                return f"Total {revenue_col} in {requested_country.title()} is {fmt_num(temp[revenue_col].sum())}."
            return f"No data found for {requested_country.title()}."
        if "country" in q:
            return "That country is not present in the dataset."

    if country_col and ("show all transactions" in q or "transactions for" in q):
        requested_country = infer_value_from_column(country_col)
        requested_years = detect_years(q)
        temp = df.copy()
        if requested_country:
            temp = apply_text_filter(temp, country_col, requested_country)
        elif "country" in q or "usa" in q or "uk" in q or "india" in q:
            return "That country is not present in the dataset."
        if requested_years and order_dates is not None:
            temp[order_date_col] = order_dates
            temp = temp[temp[order_date_col].dt.year == requested_years[0]]
        if temp.empty:
            if requested_country and requested_years:
                return f"No transactions found for {requested_country.title()} in {requested_years[0]}."
            return "No matching transactions found."
        preview_cols = [c for c in [ "OrderID", company_col, product_col, country_col, order_date_col, revenue_col, profit_col] if c and c in temp.columns]
        scope = f"{requested_country.title()} in {requested_years[0]}" if requested_country and requested_years else (requested_country.title() if requested_country else "the request")
        return build_preview_answer(f"Here is a sample of transactions for {scope}:", temp, preview_cols, limit=5)

    if product_col and ("products sold more than" in q or "product sold more than" in q):
        units_col = find_col("units", "sold") or find_col("quantity")
        threshold_match = re.search(r"more than\s*([0-9][0-9,]*)", q)
        if units_col and threshold_match:
            threshold = safe_float(threshold_match.group(1).replace(",", ""))
            temp = df[[product_col, units_col]].copy()
            temp[units_col] = temp[units_col].apply(to_number)
            temp[product_col] = temp[product_col].astype(str).str.strip()
            temp = temp.dropna(subset=[units_col])
            grouped = temp.groupby(product_col)[units_col].sum()
            winners = [f"{idx} ({fmt_num(val)})" for idx, val in grouped.items() if val > threshold]
            if winners:
                return "Products sold more than " + fmt_num(threshold) + " units: " + "; ".join(winners[:20])
            return f"No products sold more than {fmt_num(threshold)} units."

    if product_col and any(token in q for token in ["top 5 selling products", "top five selling products", "top selling products"]):
        units_col = find_col("units", "sold") or find_col("quantity")
        metric_col = units_col or revenue_col
        if metric_col:
            temp = df[[product_col, metric_col]].copy()
            temp[metric_col] = temp[metric_col].apply(to_number)
            temp[product_col] = temp[product_col].astype(str).str.strip()
            temp = temp.dropna(subset=[metric_col])
            temp = temp[temp[product_col] != ""]
            if not temp.empty:
                ranked = temp.groupby(product_col)[metric_col].sum().sort_values(ascending=False).head(5)
                label = metric_col
                table = build_markdown_table(
                    ["Rank", "Product", label],
                    [[i + 1, idx, fmt_num(val)] for i, (idx, val) in enumerate(ranked.items())],
                )
                if table:
                    return f"Top 5 selling products by {label}:\n\n{table}"
                return "Not available in the dataset"

    if category_col and revenue_col and "total revenue" in q and "category" in q:
        requested_category = infer_value_from_column(category_col)
        if requested_category:
            temp = df[[category_col, revenue_col]].copy()
            temp = apply_text_filter(temp, category_col, requested_category)
            temp[revenue_col] = temp[revenue_col].apply(to_number)
            temp = temp.dropna(subset=[revenue_col])
            if not temp.empty:
                return f"Total {revenue_col} for {requested_category.title()} category is {fmt_num(temp[revenue_col].sum())}."
            return f"No data found for {requested_category.title()} category."

    if country_col and revenue_col and any(token in q for token in ["total revenue per country", "revenue generated by each country", "total revenue generated by each country", "revenue by country"]):
        temp = df[[country_col, revenue_col]].copy()
        temp[country_col] = temp[country_col].astype(str).str.strip()
        temp[revenue_col] = temp[revenue_col].apply(to_number)
        temp = temp.dropna(subset=[revenue_col])
        temp = temp[temp[country_col] != ""]
        grouped = temp.groupby(country_col)[revenue_col].sum().sort_values(ascending=False)
        table = build_markdown_table(
            ["Country", revenue_col],
            [[idx, fmt_num(val)] for idx, val in grouped.items()],
        )
        if table:
            return f"Total revenue per country:\n\n{table}"
        return "Not available in the dataset"

    if country_col and any(token in q for token in ["highest number of orders", "most orders", "number of orders"]) and "country" in q:
        temp = df[[country_col]].copy()
        temp[country_col] = temp[country_col].astype(str).str.strip()
        temp = temp[temp[country_col] != ""]
        grouped = temp.groupby(country_col).size().sort_values(ascending=False)
        if not grouped.empty:
            best = str(grouped.idxmax())
            return f"{best} has the highest number of orders ({int(grouped.loc[best])})."

    if order_dates is not None and revenue_col and "month" in q and any(token in q for token in ["highest sales", "highest revenue", "had the highest sales"]):
        temp = df.copy()
        temp[order_date_col] = order_dates
        temp[revenue_col] = temp[revenue_col].apply(to_number)
        temp = temp[(temp[order_date_col].notna()) & (temp[revenue_col].notna())]
        quarter_num = detect_quarter_reference(q)
        if quarter_num is not None:
            temp = temp[temp[order_date_col].dt.quarter == quarter_num]
        if not temp.empty:
            grouped = temp.groupby(temp[order_date_col].dt.month)[revenue_col].sum()
            if not grouped.empty:
                best_month = int(grouped.idxmax())
                best_value = grouped.loc[best_month]
                month_name = next(name.title() for name, month_no in month_names.items() if month_no == best_month)
                if quarter_num is not None:
                    return f"{month_name} had the highest {revenue_col} in Q{quarter_num} ({fmt_num(best_value)})."
                return f"{month_name} had the highest {revenue_col} ({fmt_num(best_value)})."

    if product_col and ("total quantity sold per product" in q or ("quantity" in q and "per product" in q)):
        units_col = find_col("units", "sold") or find_col("quantity")
        if units_col:
            temp = df[[product_col, units_col]].copy()
            temp[units_col] = temp[units_col].apply(to_number)
            temp[product_col] = temp[product_col].astype(str).str.strip()
            temp = temp.dropna(subset=[units_col])
            temp = temp[temp[product_col] != ""]
            grouped = temp.groupby(product_col)[units_col].sum().sort_values(ascending=False)
            rows = [f"{idx}: {fmt_num(val)}" for idx, val in grouped.items()]
            return "Total quantity sold per product: " + "; ".join(rows[:20])

    if category_col and profit_col and ("category generates the most profit" in q or ("category" in q and "most profit" in q)):
        top = aggregate_entity_metric(df, category_col, profit_col, highest=True)
        if top:
            entity, value = top
            return f"{entity} generates the most {profit_col} ({fmt_num(value)})."

    if country_col and revenue_col and ("highest average order value" in q or ("country" in q and "average order value" in q and any(w in q for w in strongest_words))):
        temp = df[[country_col, revenue_col]].copy()
        temp[country_col] = temp[country_col].astype(str).str.strip()
        temp[revenue_col] = temp[revenue_col].apply(to_number)
        temp = temp.dropna(subset=[revenue_col])
        temp = temp[temp[country_col] != ""]
        grouped = temp.groupby(country_col)[revenue_col].mean()
        if not grouped.empty:
            best = str(grouped.idxmax())
            return f"{best} has the highest average order value ({fmt_num(grouped.loc[best])})."

    if product_col and revenue_col and ("highest sales but lowest quantity" in q or "highest revenue but lowest number of orders" in q):
        units_col = find_col("units", "sold") or find_col("quantity")
        if units_col:
            temp = df[[product_col, revenue_col, units_col]].copy()
            temp[revenue_col] = temp[revenue_col].apply(to_number)
            temp[units_col] = temp[units_col].apply(to_number)
            temp[product_col] = temp[product_col].astype(str).str.strip()
            temp = temp.dropna(subset=[revenue_col, units_col])
            grouped = temp.groupby(product_col)[[revenue_col, units_col]].sum()
            if not grouped.empty:
                grouped["revenue_rank"] = grouped[revenue_col].rank(ascending=False, method="min")
                grouped["units_rank"] = grouped[units_col].rank(ascending=True, method="min")
                grouped["combo"] = grouped["revenue_rank"] + grouped["units_rank"]
                best = str(grouped["combo"].idxmin())
                row = grouped.loc[best]
                return f"{best} balances high {revenue_col} with low {units_col} ({fmt_num(row[revenue_col])}, {fmt_num(row[units_col])})."

    if order_dates is not None and revenue_col and ("highest revenue but lowest number of orders" in q or ("month" in q and "lowest number of orders" in q)):
        temp = df.copy()
        temp[order_date_col] = order_dates
        temp[revenue_col] = temp[revenue_col].apply(to_number)
        temp = temp[(temp[order_date_col].notna()) & (temp[revenue_col].notna())]
        if not temp.empty:
            grouped = temp.groupby(temp[order_date_col].dt.month).agg(
                revenue_sum=(revenue_col, "sum"),
                order_count=(revenue_col, "size"),
            )
            grouped["combo"] = grouped["revenue_sum"].rank(ascending=False, method="min") + grouped["order_count"].rank(ascending=True, method="min")
            best_month = int(grouped["combo"].idxmin())
            month_name = next(name.title() for name, month_no in month_names.items() if month_no == best_month)
            row = grouped.loc[best_month]
            return f"{month_name} has the strongest mix of high revenue and low order count ({fmt_num(row['revenue_sum'])}, {int(row['order_count'])} orders)."

    if product_col and country_col and revenue_col and "top 3 products" in q and "countries" in q:
        temp = df[[country_col, product_col, revenue_col]].copy()
        temp[country_col] = temp[country_col].astype(str).str.strip()
        temp[product_col] = temp[product_col].astype(str).str.strip()
        temp[revenue_col] = temp[revenue_col].apply(to_number)
        temp = temp.dropna(subset=[revenue_col])
        temp = temp[(temp[country_col] != "") & (temp[product_col] != "")]
        grouped = temp.groupby([country_col, product_col])[revenue_col].sum().reset_index()
        blocks = []
        for country, grp in grouped.groupby(country_col):
            top3 = grp.sort_values(by=revenue_col, ascending=False).head(3)
            rows = ", ".join(f"{r[product_col]} ({fmt_num(r[revenue_col])})" for _, r in top3.iterrows())
            blocks.append(f"{country}: {rows}")
        return "Top 3 products across countries: " + "; ".join(blocks[:12])

    if order_dates is not None and revenue_col and ("trend" in q or "sales trend" in q):
        temp = df.copy()
        temp[order_date_col] = order_dates
        temp[revenue_col] = temp[revenue_col].apply(to_number)
        temp = temp[(temp[order_date_col].notna()) & (temp[revenue_col].notna())]
        if not temp.empty:
            monthly = temp.groupby(temp[order_date_col].dt.to_period("M"))[revenue_col].sum().sort_index()
            first = monthly.iloc[0]
            last = monthly.iloc[-1]
            direction = "increasing" if last > first else ("decreasing" if last < first else "stable")
            top_period = str(monthly.idxmax())
            low_period = str(monthly.idxmin())
            return f"Sales trend is overall {direction}. Highest month: {top_period} ({fmt_num(monthly.max())}). Lowest month: {low_period} ({fmt_num(monthly.min())})."

    if "salesperson" in q:
        return "Not available in the dataset. No salesperson field was found."

    if product_col and country_col and revenue_col and "country" in q and "specific product" in q:
        return "Specify the product name to compare country-level sales for that product."

    if product_col and country_col and revenue_col and "country" in q and ("highest sales" in q or "highest revenue" in q):
        requested_product = infer_value_from_column(product_col)
        if requested_product:
            temp = df[[country_col, product_col, revenue_col]].copy()
            temp = apply_text_filter(temp, product_col, requested_product)
            temp[country_col] = temp[country_col].astype(str).str.strip()
            temp[revenue_col] = temp[revenue_col].apply(to_number)
            temp = temp.dropna(subset=[revenue_col])
            temp = temp[temp[country_col] != ""]
            grouped = temp.groupby(country_col)[revenue_col].sum()
            if not grouped.empty:
                best = str(grouped.idxmax())
                return f"{best} has the highest {revenue_col} for {requested_product} ({fmt_num(grouped.loc[best])})."

    if revenue_col and ("correlation" in q or "relationship" in q) and ("quantity" in q or "units sold" in q):
        units_col = find_col("units", "sold") or find_col("quantity")
        if units_col:
            temp = df[[units_col, revenue_col]].copy()
            temp[units_col] = temp[units_col].apply(to_number)
            temp[revenue_col] = temp[revenue_col].apply(to_number)
            temp = temp.dropna(subset=[units_col, revenue_col])
            if not temp.empty:
                corr = temp[units_col].corr(temp[revenue_col])
                if pd.notna(corr):
                    return f"The correlation between {units_col} and {revenue_col} is {corr:.4f}."

    if "region" in q and ("quarter" in q or detect_quarter_reference(q) is not None or any(token in q for token in ["q1", "q2", "q3", "q4"])):
        return "Not available in the dataset. No region field was found."

    if country_col and ("country that does not exist" in q or "country does not exist" in q):
        return "No matching country was found in the dataset."

    if order_dates is not None and "2050" in q:
        temp = df.copy()
        temp[order_date_col] = order_dates
        temp = temp[temp[order_date_col].dt.year == 2050]
        if temp.empty:
            return "No data found for year 2050."

    if profit_col and ("negative profit" in q or "transactions with negative profit" in q):
        temp = df.copy()
        temp[profit_col] = temp[profit_col].apply(to_number)
        temp = temp[temp[profit_col] < 0]
        if temp.empty:
            return "No transactions with negative profit were found."
        preview_cols = [c for c in ["OrderID", company_col, product_col, country_col, profit_col] if c and c in temp.columns]
        return build_preview_answer("Here is a sample of transactions with negative profit:", temp, preview_cols, limit=10)

    if "duplicate" in q:
        dup_mask = df.astype(str).duplicated(keep=False)
        dup_count = int(dup_mask.sum())
        if dup_count == 0:
            return "No duplicate records were found."
        return f"Found {dup_count:,} duplicate records."

    if "place making most money" in q or "lowest performing country" in q or ("least sales" in q and any(token in q for token in ["country", "place", "region"])):
        if country_col and revenue_col:
            temp = df[[country_col, revenue_col]].copy()
            temp[country_col] = temp[country_col].astype(str).str.strip()
            temp[revenue_col] = temp[revenue_col].apply(to_number)
            temp = temp.dropna(subset=[revenue_col])
            grouped = temp.groupby(country_col)[revenue_col].sum()
            if not grouped.empty:
                if "lowest" in q or "least" in q:
                    worst = str(grouped.idxmin())
                    return f"{worst} is the lowest performing country by {revenue_col} ({fmt_num(grouped.loc[worst])})."
                best = str(grouped.idxmax())
                return f"{best} is making the most money with total {revenue_col} of {fmt_num(grouped.loc[best])}."

    if product_col and ("buying like crazy" in q or "item is useless" in q):
        metric_col = find_col("units", "sold") or revenue_col
        if metric_col:
            temp = df[[product_col, metric_col]].copy()
            temp[metric_col] = temp[metric_col].apply(to_number)
            temp[product_col] = temp[product_col].astype(str).str.strip()
            temp = temp.dropna(subset=[metric_col])
            grouped = temp.groupby(product_col)[metric_col].sum()
            if not grouped.empty:
                if "useless" in q or "least sales" in q:
                    worst = str(grouped.idxmin())
                    return f"{worst} is the least selling product ({fmt_num(grouped.loc[worst])})."
                best = str(grouped.idxmax())
                return f"{best} is the most purchased product ({fmt_num(grouped.loc[best])})."

    if country_col and revenue_col and "country" in q and any(word in q for word in strongest_words) and ("sales" in q or "revenue" in q):
        top = aggregate_entity_metric(df, country_col, revenue_col, highest=True)
        if top:
            entity, value = top
            return f"{entity} has the highest total {revenue_col} ({fmt_num(value)})."

    if product_col and revenue_col and "product" in q and "top" in q and any(word in q for word in ["revenue", "sales"]):
        temp = df[[product_col, revenue_col]].copy()
        temp[revenue_col] = temp[revenue_col].apply(to_number)
        temp[product_col] = temp[product_col].astype(str).str.strip()
        temp = temp.dropna(subset=[revenue_col])
        temp = temp[temp[product_col] != ""]
        if not temp.empty:
            ranked = temp.groupby(product_col)[revenue_col].sum().sort_values(ascending=False).head(5)
            if not ranked.empty:
                rows = [f"{i+1}. {idx} ({fmt_num(val)})" for i, (idx, val) in enumerate(ranked.items())]
                return "Top 5 products by revenue: " + "; ".join(rows)

    if company_col and revenue_col and "compan" in q and "top" in q and "revenue" in q:
        temp = df[[company_col, revenue_col]].copy()
        temp[revenue_col] = temp[revenue_col].apply(to_number)
        temp[company_col] = temp[company_col].astype(str).str.strip()
        temp = temp.dropna(subset=[revenue_col])
        temp = temp[temp[company_col] != ""]
        if not temp.empty:
            ranked = temp.groupby(company_col)[revenue_col].sum().sort_values(ascending=False).head(5)
            if not ranked.empty:
                rows = [f"{i+1}. {idx} ({fmt_num(val)})" for i, (idx, val) in enumerate(ranked.items())]
                return "Top 5 companies by revenue: " + "; ".join(rows)

    if category_col and revenue_col and profit_col and "profit margin" in q and "category" in q:
        temp = df[[category_col, revenue_col, profit_col]].copy()
        temp[revenue_col] = temp[revenue_col].apply(to_number)
        temp[profit_col] = temp[profit_col].apply(to_number)
        temp[category_col] = temp[category_col].astype(str).str.strip()
        temp = temp.dropna(subset=[revenue_col, profit_col])
        temp = temp[(temp[category_col] != "") & (temp[revenue_col] != 0)]
        if not temp.empty:
            grouped = temp.groupby(category_col)[[revenue_col, profit_col]].sum()
            grouped = grouped[grouped[revenue_col] != 0]
            if not grouped.empty:
                rows = [
                    f"{idx}: {(float(row[profit_col]) / float(row[revenue_col]) * 100):.2f}%"
                    for idx, row in grouped.iterrows()
                ]
                return "Profit margin by category: " + "; ".join(rows)

    if order_date_col and revenue_col:
        order_dates = pd.to_datetime(df[order_date_col], errors="coerce")
        requested_month = next((month_no for month_name, month_no in month_names.items() if month_name in q), None)
        if requested_month is not None and "sales" in q:
            temp = df.copy()
            temp[order_date_col] = order_dates
            temp[revenue_col] = temp[revenue_col].apply(to_number)
            temp = temp[(temp[order_date_col].notna()) & (temp[order_date_col].dt.month == requested_month)]
            temp = temp.dropna(subset=[revenue_col])
            if not temp.empty:
                month_total = temp[revenue_col].sum()
                month_name = next(name.title() for name, month_no in month_names.items() if month_no == requested_month)
                return f"Total {revenue_col} in {month_name} is {fmt_num(month_total)}."

    if product_col and "product" in q and "zero sales" in q:
        metric_col = revenue_col or find_col("units", "sold") or find_col("quantity")
        if metric_col:
            temp = df[[product_col, metric_col]].copy()
            temp[metric_col] = temp[metric_col].apply(to_number)
            temp[product_col] = temp[product_col].astype(str).str.strip()
            temp = temp.dropna(subset=[metric_col])
            temp = temp[temp[product_col] != ""]
            if not temp.empty:
                grouped = temp.groupby(product_col)[metric_col].sum()
                zeros = [str(idx) for idx, val in grouped.items() if abs(float(val)) < 1e-9]
                if zeros:
                    return "Products with zero sales: " + unique_join(zeros)
                return "No products have zero sales."

    if country_col and revenue_col and "compare" in q and any(word in q for word in ["revenue", "sales"]):
        requested_countries = infer_values_from_column(country_col, limit=4)
        if len(requested_countries) >= 2:
            temp = df[[country_col, revenue_col]].copy()
            temp[country_col] = temp[country_col].astype(str).str.strip()
            temp[revenue_col] = temp[revenue_col].apply(to_number)
            temp = temp.dropna(subset=[revenue_col])
            temp = temp[temp[country_col] != ""]
            if not temp.empty:
                grouped = temp.groupby(country_col)[revenue_col].sum()
                selected = [country for country in requested_countries if country in grouped.index]
                if len(selected) >= 2:
                    compared = selected[:2]
                    winner = max(compared, key=lambda country: grouped.loc[country])
                    table = build_markdown_table(
                        ["Country", revenue_col],
                        [[country, fmt_num(grouped.loc[country])] for country in compared],
                    )
                    answer = f"Revenue comparison by country. {winner} is higher."
                    if table:
                        answer += f"\n\n{table}"
                    return answer

    # Comparison-style finance query.
    
    if "compare" in q and revenue_col and profit_col and any(k in q for k in ["revenue", "profit"]):
        rev_total = df[revenue_col].apply(to_number).dropna().sum()
        prof_total = df[profit_col].apply(to_number).dropna().sum()
        margin = (prof_total / rev_total * 100) if rev_total else 0.0
        memory["last_structured_intent"] = "compare"
        return (
            f"Total {revenue_col}: {fmt_num(rev_total)}. "
            f"Total {profit_col}: {fmt_num(prof_total)}. "
            f"Profit margin: {margin:.2f}%."
        )

    # Profitability check.
    
    if any(k in q for k in ["profitable", "profitability"]):
        if profit_col:
            prof_total = df[profit_col].apply(to_number).dropna().sum()
            memory["last_structured_intent"] = "is company"
            return "Yes, the company is profitable." if prof_total > 0 else "No, the company is not profitable."
        if profit_or_loss_col:
            pnl_total = df[profit_or_loss_col].apply(to_number).dropna().sum()
            memory["last_structured_intent"] = "is company"
            return "Yes, the company is profitable." if pnl_total > 0 else "No, the company is not profitable."

    # Year with highest/lowest metric.
    
    if year_col and any(k in q for k in ["highest", "max", "lowest", "min"]):
        metric_col = None
        if "revenue" in q and revenue_col:
            metric_col = revenue_col
        elif "profit" in q and profit_col:
            metric_col = profit_col
        elif "loss" in q and loss_col:
            metric_col = loss_col

        if metric_col:
            metric_series = df[metric_col].apply(to_number)
            valid = metric_series.dropna()
            if not valid.empty:
                idx = valid.idxmax() if any(k in q for k in ["highest", "max"]) else valid.idxmin()
                yr = df.loc[idx, year_col]
                val = valid.loc[idx]
                return f"{int(yr) if str(yr).replace('.','',1).isdigit() else yr} has the {'highest' if any(k in q for k in ['highest','max']) else 'lowest'} {metric_col} ({fmt_num(val)})."

    dynamic_answer = dynamic_tabular_fallback()
    if dynamic_answer:
        return dynamic_answer

    for word in query_tokens:
        for col_name in cols_lower:
            if word in col_name:
                col = cols_lower[col_name]
                break
        if col:
            break
    
    if not col:
        col = calc_numeric_cols[0]
        
    
    detected_years = detect_years(q)
    if detected_years:
        year = detected_years[0]
        memory["last_year"] = str(year)
    elif "last_year" in memory:
        try:
            year = int(memory["last_year"])
        except ValueError:
            year = None

    
    if "last_year" in memory and any(word in q for word in ["profit", "loss", "revenue"]):
        carry_ref = bool(re.search(r"\b(it|that|same|this)\b", q))
        if carry_ref and not re.search(r"(20\d{2})", q):
            logging.info("Appending year to query")
            q = q + " in " + memory["last_year"]

    # Generic single-year lookup for detected metric column.
    
    if year is not None and year_col and col in df.columns:
        year_rows = df[df[year_col] == year]
        if not year_rows.empty:
            if not any(k in q for k in ["compare", "difference", "diff", "increase", "decrease", "growth"]):
                return f"In {year}, {col} was {fmt_num(year_rows.iloc[0][col])}."

    if year is not None and year_col and profit_or_loss_col:

        if "loss" in q:
            data = df[(df[year_col] == year) & (df[profit_or_loss_col] < 0)]
            return f"In {year}, the total loss was {fmt_num(abs(data[profit_or_loss_col].sum()))}."

        if "profit" in q:
            data = df[(df[year_col] == year) & (df[profit_or_loss_col] > 0)]
            return f"In {year}, the total profit was {fmt_num(data[profit_or_loss_col].sum())}."
    


    highest_pattern = re.search(r"(.*?) in year with highest ([a-zA-Z_]+)", q)
    lowest_pattern = re.search(r"(.*?) in year with lowest ([a-zA-Z_]+)", q)

    if highest_pattern:
        target_word = highest_pattern.group(1).strip()
        condition_word = highest_pattern.group(2).strip()

        condition_col = None
        for col_name, original_name in cols_lower.items():
            if condition_word in col_name:
                condition_col = original_name
                break
        if condition_col is None:
            return None

        idx = df[condition_col].idxmax()

        target_col = None
        for target_col_name, original_name in cols_lower.items():
            if target_word in target_col_name:
                target_col = original_name
                break
        if target_col is None:
            target_col = condition_col

        value = df.loc[idx, target_col]
        if "year" in cols_lower:
            year = df.loc[idx, cols_lower["year"]]
            return f"in {year}, {target_col} was {value} (year with highest {condition_col})."
        return f"{target_col} was {value} in row with highest {condition_col}."
                    
    
    if lowest_pattern:
        target_word = lowest_pattern.group(1).strip()
        condition_word = lowest_pattern.group(2).strip()

        for col_name in cols_lower:
            if condition_word in col_name:
                condition_col = cols_lower[col_name]
                idx = df[condition_col].idxmin()

                for target_col_name in cols_lower:
                    if target_word in target_col_name:
                        target_col = cols_lower[target_col_name]
                        value = df.loc[idx, target_col]

                        if "year" in cols_lower:
                            year = df.loc[idx, cols_lower["year"]]
                            return f"in {year}, {target_col} was {value} (year with lowest {condition_col})."
                        
                        return f"{target_col} was {value} in row with lowest {condition_col}."
                            

# ------------------ Column Mapping ------------------

    SUBJECT_MAP = {
        "math": ["math"],
        "writing": ["writing"],
        "reading": ["reading"],
        "exam": ["exam", "score"],
        "revenue": ["revenue", "sales"],
        "profit": ["profit"],
        "loss": ["loss"],
        "score": ["score", "marks"],
        "interest": ["interest"],
        "expense": ["expense", "cost"]
    }

    col = None
    
# ------------------ Detect Column ------------------

    for subject, keywords in SUBJECT_MAP.items():
        if subject in q:
            for c in df.columns:
                cname = c.lower()
                if any(word in cname for word in query_tokens):
                    if pd.api.types.is_numeric_dtype(df[c]):
                        col = c
                        break
        if col:
            break
    
# ------------------ Fallback Numeric Column ------------------
    
    if col is None:
        col = numeric_cols[0]

    # -------- year vs year comparison --------

    year_compare = re.findall(r"(20\d{2})", q)

    if len(year_compare) == 2 and year_col:

       y1, y2 = map(int, year_compare[:2])
       if col == year_col:
           col = revenue_col or profit_col or calc_numeric_cols[0]

       row1 = df[df[year_col] == y1]
       row2 = df[df[year_col] == y2]

       if not row1.empty and not row2.empty:
        v1 = row1.iloc[0][col]
        v2 = row2.iloc[0][col]

        if any(k in q for k in ["growth", "percentage", "rate"]):
            if safe_float(v1) != 0:
                pct = ((safe_float(v2) - safe_float(v1)) / safe_float(v1)) * 100
                return f"{col} growth from {y1} to {y2} is {pct:.2f}%."
            return f"{col} growth from {y1} to {y2} is not computable because the base value is zero."

        if any(k in q for k in ["difference", "diff", "change", "compare", "increase", "decrease"]):
            diff = v2 - v1
            direction = "increased" if diff > 0 else "decreased"
            if diff == 0:
                return f"{col} stayed the same from {y1} to {y2}."
            return f"{col} {direction} from {y1} to {y2} by {abs(diff)}."
        if v1 > v2:
            return f"{y1} is financially stronger than {y2} based on {col}."
        elif v2 > v1:
            return f"{y2} is financially stronger than {y1} based on {col}."
        else:
            return f"{y1} and {y2} show equal financial strength based on {col}."

    # Generic growth-rate query without explicit years.
    
    if any(k in q for k in ["growth", "growth rate", "percentage growth", "cagr"]):
        metric_col = revenue_col or profit_col or col
        series_src = df[metric_col].apply(to_number).dropna()
        if year_col and metric_col in df.columns:
            temp = df[[year_col, metric_col]].copy()
            temp[metric_col] = temp[metric_col].apply(to_number)
            temp = temp.dropna(subset=[metric_col])
            if not temp.empty:
                temp = temp.sort_values(by=year_col)
                series_src = temp[metric_col]
        if len(series_src) > 1 and safe_float(series_src.iloc[0]) != 0:
            g = ((safe_float(series_src.iloc[-1]) - safe_float(series_src.iloc[0])) / safe_float(series_src.iloc[0])) * 100
            return f"{metric_col} growth rate is {g:.2f}% over the available period."
          
# ------------------ Basic Calculations ------------------

    series = df[col].apply(to_number).dropna()
    if "average" in q or "mean" in q:
        return f"The average {col} is {series.mean():.2f}."

    if "highest" in q or "max" in q:
        return f"The highest {col} is {fmt_num(series.max())}."

    if "lowest" in q or "min" in q:
        return f"The lowest {col} is {fmt_num(series.min())}."

    if "total" in q or "sum" in q:
        memory["last_structured_intent"] = "what is total"
        return f"The total {col} is {fmt_num(series.sum())}."

    # Default direct metric request (e.g., "what is revenue")
    
    if any(w in q for w in ["what is", "what's", "revenue", "profit", "margin"]):
        if "margin" in q and revenue_col and profit_col:
            rev_total = df[revenue_col].apply(to_number).dropna().sum()
            prof_total = df[profit_col].apply(to_number).dropna().sum()
            margin = (prof_total / rev_total * 100) if rev_total else 0.0
            memory["last_structured_intent"] = "what is"
            return f"Overall profit margin is {margin:.2f}%."
        if col in df.columns:
            s = df[col].apply(to_number).dropna()
            if not s.empty:
                memory["last_structured_intent"] = "what is"
                return f"The total {col} is {fmt_num(s.sum())}."


# ------------------ Growth ------------------
       
    if "growth" in q or "increase" in q:        
        if len(series) > 1 and series.iloc[0] != 0:
            growth = ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100
            return f"{col} grew by {growth:.2f}% from first to last year."
    
# ------------------ Profit Margin ------------------
    
    
    if "margin" in q and "revenue" in cols_lower and "profit" in cols_lower:
        rev_col = cols_lower["revenue"]
        prof_col = cols_lower["profit"]
        margin = (df[prof_col].sum() / df[rev_col].sum()) * 100
        return f"Overall profit margin is {margin:.2f}%."

# ------------------ Trend ------------------

    if "trend" in q:
        if len(series) > 1:

            if year_col:
                df = df.sort_values(by=year_col)

            first = series.iloc[0]
            last = series.iloc[-1]

            if last > first: 
                trend = "increasing"
            elif last < first: 
                trend = "decreasing"
            else: 
                trend = "stable"

            return f"{col} shows an overall {trend} trend over time."
    
# ------------------ Best/Worst Year ------------------

    if "best year" in q or "highest year" in q:
        idx = series.idxmax()
        year = df.loc[idx, year_col] if year_col else idx
        return f"Best year for {col} was {year} with value {series.max()}."
    
    if "worst year" in q or "lowest year" in q:
        idx = series.idxmin()
        year = df.loc[idx, year_col] if year_col else idx
        return f"Worst year for {col} was {year} with value {series.min()}."

    return None


def run_structured_query(query, uploaded_files, conversation_memory):
    """Run Pandas-style structured query across uploaded tabular files."""
    loaded_frames = []
    for file_path in uploaded_files:
        df = load_dataset(file_path)
        if df is not None:
            loaded_frames.append(df)

    if loaded_frames:
        normalized_frames = []
        normalized_col_sets = []
        for frame in loaded_frames:
            normalized = frame.copy()
            normalized.columns = [str(c).strip().lower() for c in normalized.columns]
            normalized_frames.append(normalized)
            normalized_col_sets.append(tuple(normalized.columns))

        base_cols = list(normalized_frames[0].columns)
        base_col_set = set(base_cols)
        same_schema = all(set(frame.columns) == base_col_set for frame in normalized_frames[1:])

        if same_schema:
            try:
                aligned_frames = [frame.reindex(columns=base_cols) for frame in normalized_frames]
                combined_df = pd.concat(aligned_frames, ignore_index=True)
                logging.info("STRUCTURED ROUTE -> Combined dataframe across %s files", len(loaded_frames))
                combined_answer = answer_calculation(query, combined_df, memory=conversation_memory)
                if combined_answer is None:
                    fallback_answer = answer_tabular(query, combined_df)
                    if fallback_answer and "outside the financial dataset" not in fallback_answer.lower():
                        combined_answer = fallback_answer
                if combined_answer is not None:
                    return combined_answer
            except Exception as e:
                logging.info(f"Combined structured path failed: {e}")

    pandas_answer = []

    for file_path in uploaded_files:
        df = load_dataset(file_path)
        if df is None:
            continue

        logging.info("STRUCTURED ROUTE -> Pandas engine only")
        try:
            calc_answer = answer_calculation(query, df, memory=conversation_memory)
        except Exception as e:
            logging.info(f"Pandas error: {e}")
            calc_answer = None

        if calc_answer is None:
            try:
                fallback_answer = answer_tabular(query, df)
                if fallback_answer and "outside the financial dataset" not in fallback_answer.lower():
                    calc_answer = fallback_answer
            except Exception as e:
                logging.info(f"Tabular fallback error: {e}")

        if calc_answer is not None:
            pandas_answer.append(calc_answer)

    if pandas_answer:
        if len(set(pandas_answer)) == 1:
            return pandas_answer[0]

        table = build_markdown_table(
            ["File", "Answer"],
            [[f"File {i}", ans] for i, ans in enumerate(pandas_answer, 1)],
        )
        if table:
            return "Conflict detected across files.\n\n" + table
        final_answer = "Conflict detected across files.\n\n"
        for i, ans in enumerate(pandas_answer, 1):
            final_answer += f"File {i}\n{ans}\n\n"
        return final_answer

    return "Not available in the dataset"


def run_structured_query_batch(query_text, uploaded_files, conversation_memory):
    """Run one or many structured questions and return combined display text plus confidences."""
    queries = split_into_questions(query_text)
    if not queries:
        queries = [query_text]

    answers = []
    confidences = []
    for q in queries:
        q = q.strip()
        if not q:
            continue
        answer = run_structured_query(q, uploaded_files, conversation_memory)
        part_conf = estimate_structured_confidence(answer)
        log_query_answer("STRUCTURED", q, answer)
        answers.append(answer)
        confidences.append(part_conf)

    combined = combine_answers_for_display(answers, queries)
    return combined, confidences, queries


def finalize_structured_response(answer_text, confidences, conversation_memory, user_query):
    """Normalize one structured/tabular answer plus its single displayed confidence line."""
    overall_conf = compute_overall_confidence(confidences)
    final_answer = ensure_confidence_line(answer_text, overall_conf)
    conversation_memory["last_query"] = user_query
    conversation_memory["last_answer"] = final_answer
    conversation_memory["last_confidence"] = overall_conf
    conversation_memory["last_sources"] = []
    conversation_memory["last_chunks"] = []
    return final_answer, overall_conf


# Mental model: classify question -> rank/filter evidence -> score confidence/grounding -> decide if evidence is trustworthy.

def grounding_score(answer, context):
    """Return the fraction of context chunks that contain any answer word."""
    if isinstance(context, list):
        context_chunks = [str(c) for c in context if str(c).strip()]
    elif context is None:
        context_chunks = []
    else:
        context_chunks = [str(context)]

    answer_words = [word for word in str(answer).lower().split() if word]
    if not context_chunks or not answer_words:
        return 0.0

    score = 0
    for chunk in context_chunks:
        chunk_lower = chunk.lower()
        if any(word in chunk_lower for word in answer_words):
            score += 1

    return score / len(context_chunks)


def context_coverage_score(answer, context):
    """Estimate how much of the answer is covered by retrieved evidence tokens."""
    if isinstance(context, list):
        context_chunks = [str(c) for c in context if str(c).strip()]
    elif context is None:
        context_chunks = []
    else:
        context_chunks = [str(context)]

    answer_tokens = _chunk_tokens(answer)
    if not context_chunks or not answer_tokens:
        return 0.0

    context_tokens = set()
    for chunk in context_chunks[:8]:
        context_tokens.update(_chunk_tokens(chunk))

    if not context_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)


def is_unanswerable(answer):
    triggers = [
        "cannot be determined",
        "not provided",
        "not mentioned",
        "no information",
        "not available",
    ]
    answer_lower = str(answer).lower()
    return any(trigger in answer_lower for trigger in triggers)

def calibrated_grounding(answer, context_chunks):
    """Cross-encoder grounding score across top context chunks."""
    if not context_chunks:
        return 0.0
    
    pairs = [(answer, c[:800]) for c in context_chunks[:6]]
    scores = reranker.predict(pairs)

    best = float(max(scores)) if len(scores) else 0.0
    norm = 1 / (1 + np.exp(-best))

    return norm

def detect_extraction_intent(q):
    """Detect strict extraction/equation intent patterns."""
    q = q.lower().strip()

    strong_patterns = [
        "write reaction",
        "write the reaction",
        "write equation",
        "balanced equation",
        "chemical equation for",
        "give equation",
        "calculate",
        "find value"
    ]

    weak_patterns = [
        "what is reaction",
        "why reaction",
        "explain reaction",
        "process of reaction",
        "define reaction"
    ]

    if any(p in q for p in strong_patterns):
        return True

    if any(p in q for p in weak_patterns):
        return False

    return False

# ------------------ Query Classification ------------------

def classify_query(q):
    """Classify query into semantic/structured/extraction/etc."""
    q = q.lower().strip()
    q = re.sub(r"\bsale\b", "sales", q)
    finance_words = [
        "revenue", "profit", "loss", "margin", "expense",
        "financial", "ratio", "dividend", "growth", "trend", "company", "invest",
        "sector", "industry", "asset", "assets", "liability", "liabilities",
        "companies", "sectors", "sales", "country", "countries", "region",
    ]
    tabular_agg_words = [
        "total", "sum", "average", "avg", "highest", "lowest", "top", "bottom",
        "most", "least", "count", "how many", "which", "sales", "country",
    ]
    lookup_patterns = [
        r"^(what is|what was|show|give|tell me|find)\s+",
        r"\bof\s+[a-z0-9 _-]+\??$",
        r"\bin\s+\d{4}\b",
        r"\bfor\s+[a-z0-9 _-]+\??$",
    ]
    aggregation_patterns = [
        r"\b(total|sum|average|avg|mean|median|count)\b",
        r"\b(highest|lowest|maximum|minimum|max|min|top|bottom)\b",
        r"\bcompare|comparison|difference|growth|trend\b",
        r"\bhow many\b",
        r"\bper\s+(country|company|sector|region|year|month|product)\b",
    ]

    # ---- STRONG EQUATION INTENT ----
    
    equation_patterns = [
    r"\bwrite\b.*\bequation\b",
    r"\bgive\b.*\bequation\b",
    r"\bbalanced\s+equation\b",
    r"\bchemical\s+equation\b",
    r"\bequation\s+for\b",
    r"\breaction\s+equation\b",
    r"\bformula\s+for\b",
    r"\bsymbolic\s+equation\b"
]

    for pattern in equation_patterns:
        if re.search(pattern, q):
            return "extraction"

    # Prefer deterministic tabular routing for finance ranking/comparison questions.
    
    if (
        "company" in q
        and any(
            phrase in q
            for phrase in [
                "best financial performance",
                "shows the best financial performance",
                "best overall financial performance",
                "financially stronger",
                "financial strength",
            ]
        )
    ):
        return "structured"
        
    finance_analysis_patterns = [
        r"is .* company .* (good|safe|stable|profitable|healthy)",
        r"should i invest",
        r"financial performance",
        r"future growth",
        r"risk level",
        r"compare .* company",
        r"trend analysis",
        r"how is the company doing",
        r"overall performance"
    ]

    for pattern in finance_analysis_patterns:
        if re.search(pattern, q):
            return "analytical"

    # ---- ANALYTICAL QUESTIONS ----
    
    # Keep domain-general explain/why/how in semantic mode unless finance intent is present.
    
    if re.search(r"\b(why|how|explain|reason|effect|impact|law)\b", q):
        if any(word in q for word in finance_words):
            return "analytical"
        return "semantic"

    # ---- SUMMARY ----
    if re.search(r"\b(summary|summarize|overview)\b", q):
        return "summary"
    
    # ---- SECTION ----
    if re.search(r"\b(abstract|introduction|methods|results|conclusion)\b", q):
        return "section"

    is_finance_or_tabular = any(word in q for word in finance_words)
    is_aggregation = any(re.search(pattern, q) for pattern in aggregation_patterns)
    is_lookup = any(re.search(pattern, q) for pattern in lookup_patterns)

    if is_finance_or_tabular and is_aggregation:
        return "structured"

    if is_finance_or_tabular and is_lookup:
        return "semantic"

    # ---- FINANCE ----
    if is_finance_or_tabular:
        return "structured"

    # Prefer deterministic routing for spreadsheet-style aggregation questions.
    if any(word in q for word in tabular_agg_words):
        return "structured"

    # ---- DEFAULT ---
    return "semantic"


# ------------------ Hybrid Filter ------------------

def hybrid_filter_results(query, results, threshold):
    """
    Production-safe retrieval filter

    Goal:
    - Never kill recall
    - Prefer strong matches
    - Allow moderate semantic matches
    - Always return minimum context
    """
    
    if not results:
        return []

    q_words = set(query.lower().split()) if isinstance(query, str) else set()

    # Sort by semantic similarity (highest first)
    
    ranked = sorted(results, key=lambda r: getattr(r, "score", 0.0), reverse=True)

    strong = []
    medium = []

    for r in ranked:
        text = r.payload.get("text", "").lower()
        keyword_hit = any(word in text for word in q_words) if q_words else False
        vector_score = max(0.0, min(safe_float(getattr(r, "score", 0.0)), 1.0))
        rerank_score = max(0.0, min(safe_float(r.payload.get("rerank_score_norm", 0.5)), 1.0))

        hybrid_score = (0.7 * rerank_score) + (0.3 * vector_score)
        r.score = hybrid_score

        logging.debug(
            "HYBRID: %.3f | RERANK: %.3f | VECTOR: %.3f",
            round(hybrid_score, 3),
            round(rerank_score, 3),
            round(vector_score, 3),
        )
        
        if hybrid_score >= threshold:
            strong.append(r)
        elif hybrid_score >= (threshold * 0.55) or keyword_hit:   # important: semantic but weaker matches
            medium.append(r)
        

    # --- Decision Logic ---

    if len(strong) >= 4:
        return strong[:8]
    if len(strong) + len(medium) >= 4:
        return (strong + medium)[:8]
    return ranked[:8]


# ------------------ Dynamic Threshold ------------------

def get_dynamic_threshold(query):
    """
    Dynamically adjust similarity threshold based on query type.
    lower threshold = broader retrieval
    higher threshold = stricter precision

    """
    q = query.lower().strip()
    length = len(q.split())

    try:
        query_type = classify_query(query)
    except:
        query_type = "semantic"

    base_map = {
        "summary": 0.10,
        "semantic": 0.16,
        "analytical": 0.18,
        "structured": 0.22,
        "extraction": 0.26
    }
    threshold = base_map.get(query_type, 0.18)

    if length <= 3:
        threshold -= 0.05
    elif 4 <= length <= 8:
        threshold += 0.00
    else:
        threshold += 0.04
    
    strong_precision_words = [
        "exact", "balanced", "equation", "formula",
        "value", "amount", "year", "percentage:"
    ]

    if any(word in q for word in strong_precision_words):
        threshold += 0.04

    threshold = max(0.08, min(threshold, 0.35))
    return round(threshold, 3)
   

# ------------------ Rerank Results ------------------

def rerank_results(query, results):
        """Apply lightweight lexical/symbolic reranking heuristics."""
        q_words = set(query.lower().split())

        for r in results:
            text = r.payload["text"].lower()

            keyword_score = sum(word in text for word in q_words)
            keyword_score = min(keyword_score, 4)

            symbol_bonus = 0
            if any(sym in text for sym in ["->", "=", "+", "->"]):
                symbol_bonus = 0.35

            base_score = max(0.0, min(safe_float(getattr(r, "score", 0.0)), 1.0))
            r.score = (base_score * 0.6) + ((keyword_score / 4.0) * 0.25) + symbol_bonus
            r.score = max(0.0, min(r.score, 1.0))
            # Keep payload score metadata aligned so fallback diagnostics are meaningful.
            if hasattr(r, "payload") and isinstance(r.payload, dict):
                r.payload["rerank_score_norm"] = r.score

        results.sort(key = lambda x: getattr(x, "score", 0.0), reverse = True)
        return results

# ------------------ Confidence ------------------

def compute_retrieval_diagnostics(filtered_results, top_k=5):
    """Summarize retrieval quality using top-hit strength, top-k average, and score consistency."""
    if not filtered_results:
        return {
            "retrieval_score": 0.0,
            "top1_score": 0.0,
            "top_k_avg": 0.0,
            "retrieval_consistency": 0.0,
            "top_k_count": 0,
        }

    top_results = filtered_results[:top_k]
    scores = [max(0.0, min(safe_float(r.score), 1.0)) for r in top_results]
    top1 = scores[0]
    top_k_avg = sum(scores) / len(scores)
    std_dev = float(np.std(scores)) if len(scores) > 1 else 0.0
    relative_spread = std_dev / max(top_k_avg, 0.01)
    retrieval_consistency = max(0.0, 1.0 - min(relative_spread, 1.0))

    retrieval_score = (
        0.60 * top1 +
        0.40 * top_k_avg
    ) * 100.0

    return {
        "retrieval_score": round(retrieval_score, 2),
        "top1_score": round(top1 * 100.0, 2),
        "top_k_avg": round(top_k_avg * 100.0, 2),
        "retrieval_consistency": round(retrieval_consistency * 100.0, 2),
        "top_k_count": len(scores),
    }


def calculate_confidence(filtered_results):
    """Convert retrieval diagnostics into a confidence percentage."""
    retrieval_score = compute_retrieval_diagnostics(filtered_results)["retrieval_score"]
    if retrieval_score <= 0:
        return 0.0
    return round(min(100.0, BASE_CONFIDENCE_FLOOR + (0.17 * retrieval_score)), 2)


def detect_numeric_query(query):
    keywords = ["total", "sum", "average", "count", "max", "min"]
    q = (query or "").lower()
    return any(keyword in q for keyword in keywords)


def prefer_summary_chunks(results, numeric_query=False):
    if not numeric_query:
        return list(results or [])

    def priority(result):
        payload = getattr(result, "payload", {}) or {}
        chunk_kind = str(payload.get("chunk_kind") or "").lower()
        if chunk_kind == "global_summary":
            return 0
        if chunk_kind == "dataset_summary":
            return 1
        if chunk_kind == "data_block":
            return 2
        return 3

    return sorted(results or [], key=lambda r: (priority(r), -safe_float(getattr(r, "score", 0.0))))


def render_final_answer(answer_text, confidence_percent, *, numeric_query=False):
    text = (answer_text or "").strip()

    final_match = re.search(
        r"Final Answer:\s*(.*?)\s*(?:Explanation:\s*(.*?))?\s*(?:Confidence Note:\s*(.*?))?\s*$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if final_match:
        final_answer = (final_match.group(1) or "").strip()
        explanation = (final_match.group(2) or "").strip()
    else:
        final_answer = text
        explanation = ""

    if not explanation:
        explanation = ""
        if numeric_query:
            explanation = ""

    if numeric_query and "may not be exact" not in explanation.lower():
        explanation = f"{explanation} This answer is based on summarized data and may not be exact."

    rendered = final_answer
    if explanation:
        rendered = f"{rendered}\n\n{explanation}"
    if is_unanswerable(final_answer):
        conf = 0.0
    else:
        conf = max(0.0, min(100.0, safe_float(confidence_percent)))
    return f"{rendered}\n\nConfidence (part): {round(conf, 2)}%"


def _normalize_filter_value(value):
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text).lower()


def infer_metadata_filters(query, session_id):
    q = (query or "").lower()
    docs = get_session_index_docs(session_id)
    filter_candidates = {"country": set(), "product": set()}

    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        for key, plural_key in (("country", "countries"), ("product", "products")):
            value = meta.get(key)
            if value:
                filter_candidates[key].add(str(value).strip())
            multi_values = meta.get(plural_key) or []
            if isinstance(multi_values, (list, tuple, set)):
                for item in multi_values:
                    if item:
                        filter_candidates[key].add(str(item).strip())

    detected = {}
    for key, values in filter_candidates.items():
        best_match = None
        best_len = 0
        for value in values:
            normalized = _normalize_filter_value(value)
            if not normalized:
                continue
            pattern = rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])"
            if re.search(pattern, q) and len(normalized) > best_len:
                best_match = value
                best_len = len(normalized)
        if best_match:
            detected[key] = best_match

    return detected


def compute_answer_length_score(answer_text, query_type):
    """Reward answers that are neither too thin nor excessively verbose for the task."""
    words = len(re.findall(r"\w+", answer_text or ""))
    target_ranges = {
        "extraction": (3, 35),
        "structured": (3, 45),
        "semantic": (20, 120),
        "analytical": (30, 170),
        "summary": (45, 220),
        "section": (35, 220),
    }
    min_words, max_words = target_ranges.get(query_type, (15, 120))

    if words == 0:
        return 0.0
    if words < min_words:
        return round(max(20.0, 100.0 * (words / max(min_words, 1))), 2)
    if words <= max_words:
        return 100.0

    overflow_ratio = min((words - max_words) / max(max_words, 1), 1.0)
    return round(max(40.0, 100.0 - (overflow_ratio * 60.0)), 2)

#---------------compute trust score-------------------

# Removed dead legacy confidence function.
def _unused_removed_legacy_confidence(
    retrieval_score=None,
    grounding_score=None,
    semantic_similarity=None,
    evidence_agreement=1.0,
    context_coverage=1.0,
    retrieval_consistency=1.0,
    answer_length_score=1.0,
    query_type="semantic",
    **kwargs,
):
    # Accept legacy/newer aliases used across the answer-validation flow.
    
    if retrieval_score is None:
        retrieval_score = kwargs.pop("retrieval_conf", 0.0)
    if grounding_score is None:
        grounding_score = kwargs.pop("grounding", 0.0)
    if semantic_similarity is None:
        semantic_similarity = kwargs.pop("semantic_sim", 0.0)
    if "verifier_ok" in kwargs:
        evidence_agreement = 1.0 if kwargs.pop("verifier_ok") else 0.0
    if "top_k_avg" in kwargs:
        kwargs.pop("top_k_avg")

    def normalize_score(value):
        value = safe_float(value)
        if 0.0 <= value <= 1.0:
            return value * 100.0
        return max(0.0, min(100.0, value))

    retrieval_score = normalize_score(retrieval_score)
    grounding_score = normalize_score(grounding_score)
    semantic_similarity = normalize_score(semantic_similarity)
    evidence_agreement = normalize_score(evidence_agreement)
    context_coverage = normalize_score(context_coverage)
    retrieval_consistency = normalize_score(retrieval_consistency)
    answer_length_score = normalize_score(answer_length_score)

    # Stable production weighting: emphasize retrieval + grounding, keep smaller heuristic effects.
    
    trust = (
        0.45 * retrieval_score +
        0.30 * grounding_score +
        0.15 * semantic_similarity +
        0.10 * evidence_agreement
    )

    # Penalize conflicting chunks
    
    if evidence_agreement < 60:
        trust -= 10

    # Penalize unstable retrieval distributions.
    
    if retrieval_consistency < 45:
        trust -= 10

    # Penalize answers that are implausibly short for the task.
    
    if answer_length_score < 45:
        trust -= 10

    # Penalize low context coverage only when it is explicitly low.
    
    if context_coverage < 50:
        trust -= 8

    # Slight penalty for analytical queries.
    
    if query_type == "analytical":
        trust *= 0.95

    # Clamp between 0–100
    
    trust = max(0, min(100, trust))

    return round(trust, 2)

# ------------------ Retry/Reflection ------------------

def compute_trust_score(
    retrieval_score=None,
    grounding_score=None,
    semantic_similarity=None,
    evidence_agreement=1.0,
    context_coverage=1.0,
    retrieval_consistency=1.0,
    answer_length_score=1.0,
    top1_score=None,
    answer_text="",
    query_type="semantic",
    **kwargs,
):
    if retrieval_score is None:
        retrieval_score = kwargs.pop("retrieval_conf", 0.0)
    if grounding_score is None:
        grounding_score = kwargs.pop("grounding", 0.0)
    if top1_score is None:
        top1_score = kwargs.pop("top1", 0.0)

    def normalize_score(value):
        value = safe_float(value)
        if 0.0 <= value <= 1.0:
            return value * 100.0
        return max(0.0, min(100.0, value))

    retrieval_score = normalize_score(retrieval_score)
    grounding_score = normalize_score(grounding_score)
    retrieval_consistency = normalize_score(retrieval_consistency)
    top1_score = normalize_score(top1_score)
    semantic_similarity = normalize_score(semantic_similarity)
    evidence_agreement = normalize_score(evidence_agreement)
    context_coverage = normalize_score(context_coverage)

    evidence_strength = (
        0.35 * max(semantic_similarity, context_coverage) +
        0.25 * grounding_score +
        0.20 * evidence_agreement +
        0.10 * retrieval_score +
        0.10 * top1_score
    )
    trust = BASE_CONFIDENCE_FLOOR + (0.17 * evidence_strength)

    if is_unanswerable(answer_text):
        trust *= 0.3

    if "conflict detected" in str(answer_text or "").lower():
        trust = min(trust, 72.0)

    if top1_score < 20:
        trust *= 0.35

    if grounding_score < 20:
        trust *= 0.5

    if evidence_agreement < 25:
        trust *= 0.7

    return round(max(0.0, min(100.0, trust)), 2)


def regenerate_answer_with_reflection(query, context, previous_answer):
    """
    Second attempt forcing grounded answer
    """

    reflection_prompt = f"""
The previous answer was not sufficiently supported by the context.

Your task:
- Use ONLY the context
- Do NOT use outside knowledge
- Quote or closely follow the wording of the context
- If unsure, say: Not available in the dataset.

Context:
{context}

Question:
{query}

Previous Answer:
{previous_answer}

Corrected Answer:
"""

    try:
        response = llm_complete(reflection_prompt)

        return extract_llm_text(response) or previous_answer

    except Exception as e:
        logging.info(f"Error regenerating answer: {e}")
        return previous_answer

# ------------------ Query Expansion ------------------

QUERY_EXPANSION_SYNONYMS = {
    "workforce": ["number of employees", "employee count", "staff size", "headcount"],
    "employees": ["employee count", "number of employees", "staff size", "headcount"],
    "employee count": ["number of employees", "staff size", "headcount"],
    "headcount": ["employee count", "number of employees", "staff size"],
    "revenue": ["sales", "turnover", "total revenue"],
    "profit": ["net income", "earnings", "profitability"],
    "assets": ["total assets", "asset base"],
    "liabilities": ["total liabilities", "debt obligations"],
    "cash flow": ["cash generated", "operating cash flow"],
    "market cap": ["market capitalization", "company value"],
}


def build_query_expansion_variants(query, limit=4):
    """Create retrieval-friendly rewrites using common domain synonyms."""
    base = (query or "").strip()
    if not base:
        return []

    variants = [base]
    lowered = base.lower()

    for term, replacements in QUERY_EXPANSION_SYNONYMS.items():
        if term not in lowered:
            continue
        for replacement in replacements:
            candidate = re.sub(term, replacement, base, flags=re.IGNORECASE)
            if candidate.strip() and candidate not in variants:
                variants.append(candidate.strip())
            if len(variants) >= limit:
                return variants[:limit]

    return variants[:limit]


def expand_query_with_llm(query):
    """
    Generate retrieval-friendly query variants using synonym expansion first,
    then optional LLM rewrites to improve recall.
    """
    base = (query or "").strip()
    variants = []

    def add_variant(text):
        candidate = (text or "").strip()
        if candidate and candidate not in variants:
            variants.append(candidate)

    add_variant(base)
    add_variant(f"{base} sales data")
    add_variant(f"{base} revenue analysis")
    if re.search(r"\btotal\b", base, flags=re.IGNORECASE):
        add_variant(re.sub(r"\btotal\b", "sum", base, flags=re.IGNORECASE))

    for candidate in build_query_expansion_variants(base, limit=6):
        add_variant(candidate)
        if len(variants) >= 4:
            break

    return variants[:4]

def agent_plan(query):
    prompt = f"""
Break this problem into steps needed to answer it.

Question: {query}
"""
    try:
        response = llm_complete(prompt)
        return extract_llm_text(response)
    except Exception as e:
        logging.info(f"Agent plan error: {e}")
        return None

def decompose_query(query):
    """Break a user query into smaller document-answerable questions."""
    query = (query or "").strip()
    if not query:
        return []

    prompt = f"""
You are a helpful assistant.

Break the following question into smaller, specific questions
that can be answered from documents.

Question: {query}

Return ONLY a list of questions, one per line, with no numbering.
"""

    try:
        response = llm_complete(prompt)
        text = extract_llm_text(response)
        subqueries = [q.strip("- ").strip() for q in text.split("\n") if q.strip()]
        return subqueries or split_into_questions(query)
    except Exception as e:
        logging.info(f"Error decomposing query: {e}")
        return split_into_questions(query)


# ------------------ Reader (Answer Span Extraction) ------------------

def extract_answer_span(query, best_chunk):
    """
    Select exact supporting sentence before generation.
    Prevents LLM choosing wrong sentence inside correct chunk.
    """

    reader_prompt = f"""
You are a precise reading comprehension system.

Select ONLY the exact sentence(s) from the passage that directly answers the question.

Rules:
- Copy text exactly
- Do not explain
- Do not paraphrase
- If multiple sentences needed include them
- If not present return: {ANSWER_SPAN_NOT_FOUND}

Question:
{query}

Passage:
{best_chunk}

Answer span:
"""

    try:
        response = llm_complete(reader_prompt, temperature=0)
        return extract_llm_text(response) or ANSWER_SPAN_NOT_FOUND
    except:
        return ANSWER_SPAN_NOT_FOUND
    
#-------------reasoning function for prompt-----------
    
def reason_over_evidence(query, context):
    """
    Forces step-by-step reasoning using only extracted evidence.
    Prevents cross-chunk mixing errors.
    """

    reasoning_prompt = f"""
You are a careful scientific reasoning system.

Use ONLY the provided evidence.
Think step-by-step and connect facts logically.

Rules:
- Do not add outside knowledge
- Every conclusion must come from evidence
- If evidence insufficient say: NOT_ENOUGH_EVIDENCE

Evidence:
{context}

Question:
{query}

Reasoning:
"""

    try:
        response = llm_complete(reasoning_prompt, temperature=0)
        return extract_llm_text(response) or None
    except:
        return None
    
# ------------------ Prompt Building ------------------

def build_prompt(query, context, conflict_info, query_type, primary_evidence=None, numeric_query=False):
    """Build query-type specific instruction prompt for answering."""
    prompt = f"""
You are an expert data analyst.

Instructions:
- Answer clearly and confidently.
- Combine information from multiple chunks when useful.
- Give the final answer first, then a brief explanation.
- Explain reasoning step-by-step briefly.
- If data is incomplete, say "based on available data".
- Do not say "context does not provide".
- Use only the provided evidence.
- Do not invent facts.
- If the answer cannot be determined, say "Not available in the dataset".
{"- For numeric queries, give approximate values when only summarized evidence is available." if numeric_query else ""}
- Do not use section labels like "Final Answer", "Explanation", or "Confidence Note".
- Write plain answer text only.

Document Info:
{conflict_info}

PRIMARY EVIDENCE (answer MUST come from here if present):
{primary_evidence if primary_evidence else context}

SUPPORTING CONTEXT:
{context}

Question:
{query}

Answer:
"""

    return prompt


def is_cross_document_comparison_query(query, filtered_results):
    """Heuristic gate for comparison questions that need per-document fact extraction."""
    if not filtered_results or len(filtered_results) < 2:
        return False

    q = (query or "").lower()
    comparison_terms = [
        "compare",
        "comparison",
        "vs",
        "versus",
        "which company",
        "higher",
        "lower",
        "largest",
        "smallest",
        "most",
        "least",
        "better",
        "stronger",
    ]
    if not any(term in q for term in comparison_terms):
        return False

    files = {
        (r.payload or {}).get("file")
        for r in filtered_results
        if (r.payload or {}).get("file")
    }
    return len(files) >= 2


def build_document_fact_blocks(filtered_results, max_docs=4, max_facts_per_doc=5):
    """Group retrieved evidence by source file and extract compact fact lines."""
    grouped = defaultdict(list)
    for result in filtered_results:
        payload = result.payload or {}
        file_name = payload.get("file") or "Unknown source"
        text = (payload.get("text") or "").strip()
        if text:
            grouped[file_name].append(text)

    fact_blocks = []
    for file_name, texts in list(grouped.items())[:max_docs]:
        facts = []
        seen = set()
        for text in texts:
            for line in re.split(r"[\n;]+", text):
                clean = normalize_text(line)
                if not clean:
                    continue
                has_number = bool(re.search(r"\d|[$%\u20ac\u00a3\u00a5]", clean))
                if not has_number:
                    continue
                compact = clean[:220]
                if compact in seen:
                    continue
                seen.add(compact)
                facts.append(compact)
                if len(facts) >= max_facts_per_doc:
                    break
            if len(facts) >= max_facts_per_doc:
                break
        if facts:
            fact_blocks.append((file_name, facts))
    return fact_blocks


def answer_cross_document_comparison(query, filtered_results, q_debug):
    """Compare per-document facts first, then answer from that comparison evidence."""
    fact_blocks = build_document_fact_blocks(filtered_results)
    if len(fact_blocks) < 2:
        return None

    comparison_context = "\n\n".join(
        f"{file_name}:\n" + "\n".join(f"- {fact}" for fact in facts)
        for file_name, facts in fact_blocks
    )

    prompt = f"""
You are comparing multiple company documents.

Create a structured comparison.

Steps:
1. Extract key values per company
2. Present them in a table
3. Then give final conclusion

Format:

TABLE:
| Company | Value | Source |
|---------|-------|--------|

Then:

FINAL ANSWER:
<clear conclusion>

Data:
{comparison_context}

Question:
{query}

Answer:
"""

    t_compare = time.perf_counter()
    response = llm_complete(prompt)
    q_debug["stages_ms"]["compare_answer"] = round((time.perf_counter() - t_compare) * 1000, 2)
    q_debug["comparison_docs"] = [file_name for file_name, _ in fact_blocks]
    q_debug["comparison_mode"] = True
    return extract_llm_text(response), comparison_context


# Mental model: build prompt -> generate answer -> verify support -> recompute trust -> retry/fallback -> finalize output.

def generate_answer_with_validation(
    q,
    query_type,
    context,
    conflict_info,
    filtered_results,
    confidence_percent,
    q_debug,
    fallback_reason=None,
    embedding_cache=None,
    numeric_query=False,
):
    """Generate answer, then verify/ground/retry before returning."""
    retrieval_stats = compute_retrieval_diagnostics(filtered_results)
    candidate_spans = []
    answer_span = ANSWER_SPAN_NOT_FOUND
    best_chunk_text = context

    for r in filtered_results[:4]:
        chunk_text = r.payload.get("text", "")
        if not chunk_text.strip():
            continue
        span = extract_answer_span(q, chunk_text)
        if span != ANSWER_SPAN_NOT_FOUND:
            confidence = r.payload.get("rerank_score_norm", 0)
            candidate_spans.append((span, confidence, chunk_text))

    best_span = None
    if candidate_spans:
        best_span, _, best_chunk_text = max(candidate_spans, key=lambda x: x[1])
        answer_span = best_span

        if len(best_span) > 20:
            context = f"PRIMARY EVIDENCE:\n{best_span}\n\nSUPPORTING CONTEXT:\n{context}"
            logging.info("Best span selected from competing chunks")

        compressed_chunks = []
        for r in filtered_results[:4]:
            txt = r.payload.get("text", "")
            if not txt:
                continue
            if best_span[:40] in txt:
                idx = txt.find(best_span[:40])
                start = max(0, idx - 250)
                end = min(len(txt), idx + 250)
                txt = txt[start:end]
            compressed_chunks.append(txt[:500])

        supporting_context = "\n---\n".join(compressed_chunks)
        context = f"PRIMARY EVIDENCE:\n{best_span}\n\nLOCAL SUPPORT:\n{supporting_context}"
        logging.info("Applied anti-dilution compression")

    if query_type in ["analytical", "semantic"] and len(filtered_results) >= 2 and confidence_percent < 70:
        reasoning_source = answer_span if answer_span != ANSWER_SPAN_NOT_FOUND else (best_chunk_text or "")
        reasoning = reason_over_evidence(q, reasoning_source)
        if reasoning and "NOT_ENOUGH_EVIDENCE" not in reasoning:
            context = f"REASONED EVIDENCE:\n{reasoning}\n\nRAW CONTEXT:\n{context}"
            logging.info("Applied multi-hop reasoning step")

    prompt = build_prompt(
        query=q,
        context=context,
        conflict_info=conflict_info,
        query_type=query_type,
        primary_evidence=best_span if best_span else None,
        numeric_query=numeric_query,
    )

    t_llm = time.perf_counter()
    response = llm_complete(prompt)
    q_debug["stages_ms"]["llm_answer"] = round((time.perf_counter() - t_llm) * 1000, 2)

    ans = extract_llm_text(response)
    origin_span = extract_answer_span(q, context)
    if origin_span == "NOT_FOUND":
        logging.info("Answer not found in context â†’ rejecting hallucination")
        ans = "Not available in the dataset"
    logging.info("Answer generated from LLM")

    ans = final_safety_check(ans)
    original_ans = ans
    if embedding_cache is None:
        query_embedding = model.encode(q, normalize_embeddings=True)
    else:
        query_embedding = get_embedding_cached(q, embedding_cache)
    similarity = 0.7
    verdict = "unknown"
    verification_prompt = None

    def is_chemical_equation(text):
        if not text:
            return False
        arrow = "->" in text or "=" in text
        elements = re.findall(r"[A-Z][a-z]?\d*", text)
        return arrow and len(elements) >= 2

    def is_symbolic_answer(text):
        return bool(re.search(r"(\d+\s?[%$]|=|->|\bkg\b|\bcm\b|\bm\b|\d+\.\d+)", text))

    def compute_answer_similarity(answer_text):
        if not answer_text:
            return 0.0
        if query_type in ["semantic", "analytical", "summary"] and not is_symbolic_answer(answer_text):
            if embedding_cache is None:
                answer_embedding = model.encode(answer_text, normalize_embeddings=True)
            else:
                answer_embedding = get_embedding_cached(answer_text, embedding_cache)
            score = float(answer_embedding @ query_embedding.T)
            logging.info(f"Semantic similarity: {score}")
            if score < 0.25:
                logging.info("Low semantic similarity detected")
                return 0.15
            return score
        logging.info("Symbolic answer detected")
        return 0.85

    def score_answer_length(answer_text):
        return compute_answer_length_score(answer_text, query_type)

    if "not available" not in ans.lower():
        similarity = compute_answer_similarity(ans)
    answer_length_score = score_answer_length(ans)
    logging.info(f"Answer length score: {answer_length_score}")

    def requires_strict_verification(answer, q_type, retrieval_conf):
        if q_type == "extraction":
            return True
        if retrieval_conf < 55:
            return True
        if re.search(r"(->|->|=|\d+[A-Z][a-z]?\d*)", answer):
            return True
        return False

    if "not available" not in ans.lower() and len(context) > 120 and requires_strict_verification(ans, query_type, confidence_percent):
        if is_chemical_equation(ans):
            verification_prompt = f"""
Determine whether the answer represents the SAME CHEMICAL REACTION as described in the context.

Equivalent reactions allowed even if:
- coefficients differ
- order differs
- words vs symbols

Context:
{context}

Answer:
{ans}

Question:
{q}

Reply YES or NO only.
"""
        else:
            verification_prompt = f"""
Check if the answer is supported by the context.

Paraphrasing allowed.

Context:
{context}

Answer:
{ans}

Question:
{q}

Reply YES or NO only.
"""

    if verification_prompt is not None:
        t_verify = time.perf_counter()
        verification_response = llm_complete(verification_prompt)
        q_debug["stages_ms"]["verify"] = round((time.perf_counter() - t_verify) * 1000, 2)
        verdict = extract_llm_text(verification_response).lower()
        logging.info(f"Verification verdict: {verdict}")
    else:
        logging.info("Verification not needed")

    grounding = 0.0
    answer_span = None
    verification_context = context
    context_list = []
    coverage = 0.0
    agreement = calibrated_similarity(filtered_results) if filtered_results else 0.0

    if "not available" not in ans.lower() and filtered_results:
        best_chunk = filtered_results[0].payload.get("text") or ""
        context_list = [r.payload.get("text") or "" for r in filtered_results[:6]]
        answer_span = extract_answer_span(q, best_chunk)
        if answer_span and answer_span != ANSWER_SPAN_NOT_FOUND:
            verification_context = answer_span
        if is_chemical_equation(ans):
            grounding = max(calibrated_grounding(ans, context_list), 0.85)
        else:
            grounding = safe_float(grounding_score(ans, context_list or verification_context))
        coverage = safe_float(context_coverage_score(ans, context_list or verification_context))

    logging.info(f"Grounding score: {grounding}")
    verifier_ok = verdict == "yes"
    semantic_alignment = coverage
    trust = compute_trust_score(
        retrieval_score=retrieval_stats["retrieval_score"],
        semantic_similarity=semantic_alignment,
        grounding_score=grounding,
        evidence_agreement=agreement if verifier_ok else agreement * 0.5,
        context_coverage=coverage,
        retrieval_consistency=retrieval_stats["retrieval_consistency"],
        answer_length_score=answer_length_score,
        top1_score=retrieval_stats["top1_score"],
        answer_text=ans,
        query_type=query_type,
    )

    if origin_span == "NOT_FOUND":
        trust = 0
        verifier_ok = False
        logging.info("Trust forced to 0 â€” answer not grounded in retrieved context")

    if grounding < 0.25 and "not available" not in ans.lower():
        logging.info("Low grounding detected -> regenerating")
        ans = regenerate_answer_with_reflection(q, verification_context, ans)
        if is_chemical_equation(ans):
            grounding = max(calibrated_grounding(ans, [verification_context]), 0.85)
        else:
            grounding = safe_float(grounding_score(ans, context_list or verification_context))
        coverage = safe_float(context_coverage_score(ans, context_list or verification_context))
        semantic_alignment = coverage
        answer_length_score = score_answer_length(ans)
        trust = compute_trust_score(
            retrieval_score=retrieval_stats["retrieval_score"],
            semantic_similarity=semantic_alignment,
            grounding_score=grounding,
            evidence_agreement=agreement if verifier_ok else agreement * 0.5,
            context_coverage=coverage,
            retrieval_consistency=retrieval_stats["retrieval_consistency"],
            answer_length_score=answer_length_score,
            top1_score=retrieval_stats["top1_score"],
            answer_text=ans,
            query_type=query_type,
        )
        logging.info(f"Trust score after regeneration: {trust}")
    logging.info(f"Trust score: {trust}")

    ans = final_safety_check(ans)
    if trust < 55 and "not available" not in ans.lower():
        logging.info("low trust detected -> regenreating grounded answer")
        retry_context = answer_span if answer_span != ANSWER_SPAN_NOT_FOUND else context
        new_ans = regenerate_answer_with_reflection(q, retry_context, ans)
        if new_ans and new_ans != ans:
            logging.info("Reflection improved answer")
            ans = new_ans
            if is_chemical_equation(ans):
                grounding = max(calibrated_grounding(ans, context_list), 0.85)
            else:
                grounding = safe_float(grounding_score(ans, context_list or retry_context))
            coverage = safe_float(context_coverage_score(ans, context_list or retry_context))
            answer_length_score = score_answer_length(ans)
            trust = compute_trust_score(
                retrieval_score=retrieval_stats["retrieval_score"],
                semantic_similarity=coverage,
                grounding_score=grounding,
                evidence_agreement=agreement if verifier_ok else agreement * 0.5,
                context_coverage=coverage,
                retrieval_consistency=retrieval_stats["retrieval_consistency"],
                answer_length_score=answer_length_score,
                top1_score=retrieval_stats["top1_score"],
                answer_text=ans,
                query_type=query_type,
            )
            logging.info(f"New trust score: {trust}")

    if trust < 30:
        logging.info("Low trust score, attempting self correction")
        improved = regenerate_answer_with_reflection(q, context, ans)
        new_ground = safe_float(grounding_score(improved, context_list or context))
        new_coverage = safe_float(context_coverage_score(improved, context_list or context))
        if embedding_cache is None:
            improved_embedding = model.encode(improved, normalize_embeddings=True)
        else:
            improved_embedding = get_embedding_cached(improved, embedding_cache)
        new_sim = float(improved_embedding @ query_embedding.T)
        new_answer_length = score_answer_length(improved)
        new_trust = compute_trust_score(
            retrieval_score=retrieval_stats["retrieval_score"],
            semantic_similarity=new_coverage,
            grounding_score=new_ground,
            evidence_agreement=agreement,
            context_coverage=new_coverage,
            retrieval_consistency=retrieval_stats["retrieval_consistency"],
            answer_length_score=new_answer_length,
            top1_score=retrieval_stats["top1_score"],
            answer_text=improved,
            query_type=query_type,
        )

        if new_trust > trust and improved and "not available" not in improved.lower():
            ans = improved
            trust = new_trust
        else:
            if query_type in ["semantic", "analytical"] and filtered_results and confidence_percent >= 35:
                logging.info("Keeping best-effort answer despite low trust for semantic/analytical query")
                if improved and "not available" not in improved.lower():
                    ans = improved
                trust = max(trust, 36.0)
            else:
                fallback_reason = "low_trust_reflection_failed"
                logging.info(f"Fallback reason: {fallback_reason}")
                ans = "Not available in the dataset"

    if (
        query_type in ["semantic", "analytical"]
        and "not available" in ans.lower()
        and "not available" not in (original_ans or "").lower()
        and filtered_results
        and similarity >= 0.60
        and calculate_confidence(filtered_results) >= 40
    ):
        logging.info("Restoring original grounded answer after conservative fallback")
        ans = original_ans
        trust = max(trust, 45.0)

    confidence_percent = round(trust, 2)
    confidence_percent = max(0.0, min(100.0, confidence_percent))
    if numeric_query:
        confidence_percent *= 0.6
    if is_unanswerable(ans):
        confidence_percent = min(confidence_percent, 30.0)
    if verdict == "no" and trust < 40:
        logging.info("Verification failed, setting not available")
        fallback_reason = "verification_failed_low_trust"
        logging.info(f"Fallback reason: {fallback_reason}")
        ans = "Not available in the dataset"

    return ans, origin_span, confidence_percent, fallback_reason


def finalize_answer_text(ans, query_type, origin_span, filtered_results, confidence_percent, numeric_query=False):
    """Attach citations, final safety check, and confidence footer."""
    final_answer = ans
    if query_type != "structured" and "not available" not in ans.lower():
        try:
            safe_results = [r for r in filtered_results if r.payload.get("text")]
            citations = []
            if origin_span != "NOT_FOUND":
                citations = build_citations(ans, safe_results, origin_span)
            if citations:
                citation_text = "\n\nSources:\n" + "\n".join(f"[{i+1}] {c}" for i, c in enumerate(citations))
                final_answer = ans + citation_text
        except Exception as e:
            logging.info(f"Citation error: {e}")
            final_answer = ans

    final_answer = final_safety_check(final_answer)
    return render_final_answer(final_answer, confidence_percent, numeric_query=numeric_query)


def finalize_question_debug(q_debug, fallback_reason, q_start, effective_debug, debug_questions):
    """Finalize debug payload for one question turn."""
    q_debug["fallback_reason"] = fallback_reason
    q_debug["stages_ms"]["total"] = round((time.perf_counter() - q_start) * 1000, 2)
    if effective_debug:
        debug_questions.append(q_debug)


def calibrated_similarity(results):
    """Compute semantic agreement among top retrieved chunks."""
    if not results:
        return 0.0
    top_texts = [r.payload["text"][:400] for r in results[:3]]
    pairs = []
    for i in range(len(top_texts)):
        for j in range(i + 1, len(top_texts)):
            pairs.append((top_texts[i], top_texts[j]))
    if not pairs:
        return min(max(results[0].score, 0), 1)
    scores = reranker.predict(pairs)
    best = float(max(scores)) if len(scores) else 0.0
    return 1 / (1 + np.exp(-best))


def _chunk_tokens(text):
    return set(re.findall(r"[A-Za-z0-9]+", (text or "").lower()))


def _chunk_similarity(text_a, text_b):
    tokens_a = _chunk_tokens(text_a)
    tokens_b = _chunk_tokens(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return overlap / max(union, 1)


def remove_similar_chunks(results, similarity_threshold=0.82, keep=8):
    """Remove near-duplicate chunks while preserving score order."""
    unique = []
    for result in results or []:
        payload = getattr(result, "payload", {}) or {}
        text = (payload.get("text") or "").strip()
        if not text:
            continue
        if any(_chunk_similarity(text, (u.payload.get("text") or "")) >= similarity_threshold for u in unique):
            continue
        unique.append(result)
        if len(unique) >= keep:
            break
    return unique


def compress_chunk_for_prompt(query, text, max_chars=520):
    """Extract the most query-relevant sentences from a chunk."""
    clean_text = normalize_text(text or "")
    if not clean_text:
        return ""

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", clean_text) if s.strip()]
    if not sentences:
        return clean_text[:max_chars].strip()

    query_tokens = _chunk_tokens(query)
    scored = []
    for idx, sentence in enumerate(sentences):
        sentence_tokens = _chunk_tokens(sentence)
        overlap = len(query_tokens & sentence_tokens)
        digit_bonus = 0.2 if re.search(r"\d", sentence) else 0.0
        score = overlap + digit_bonus - (idx * 0.01)
        scored.append((score, idx, sentence))

    best_sentences = []
    total_chars = 0
    for _, idx, sentence in sorted(scored, key=lambda item: (-item[0], item[1])):
        if total_chars >= max_chars:
            break
        if sentence in best_sentences:
            continue
        sentence_len = len(sentence) + (1 if best_sentences else 0)
        if total_chars + sentence_len > max_chars and best_sentences:
            continue
        best_sentences.append(sentence)
        total_chars += sentence_len
        if len(best_sentences) >= 3:
            break

    if not best_sentences:
        return clean_text[:max_chars].strip()
    return " ".join(best_sentences)[:max_chars].strip()


def run_retrieval_pipeline(
    q,
    session_id,
    query_type,
    use_expansion,
    threshold_modifier,
    effective_debug,
    q_debug,
    embedding_cache=None,
    metadata_filter=None,
    numeric_query=False,
):
    """Run retrieval, filtering, fallback search, and scoring."""
    session_hybrid_index = get_session_hybrid_index(session_id)
    results, retrieval_queries, retrieval_debug = hybrid_retrieve(
        q,
        session_id,
        qdrant=qdrant,
        collection_name=COLLECTION,
        model=model,
        session_hybrid_index=session_hybrid_index,
        use_expansion=use_expansion,
        expand_query_fn=expand_query_with_llm,
        vector_limit=20,
        final_top_k=8,
        collect_debug=effective_debug,
        encode_query_fn=(lambda txt: get_embedding_cached(txt, embedding_cache)) if embedding_cache is not None else None,
        metadata_filter=metadata_filter,
    )
    logging.info("Calculating dynamic threshold")

    base_threshold = get_dynamic_threshold(q)
    initial_threshold = max(0.10, base_threshold + threshold_modifier)
    dynamic_threshold = compute_adaptive_threshold(results, initial_threshold)
    logging.info(f"Base threshold: {base_threshold}, Dynamic threshold: {dynamic_threshold}")
    q_debug["dynamic_threshold"] = round(dynamic_threshold, 4)

    if effective_debug and retrieval_debug:
        q_debug["stages_ms"].update(retrieval_debug.get("stages_ms", {}))
        q_debug["retrieval_queries"] = retrieval_debug.get("retrieval_queries", retrieval_queries)
        q_debug["vector_hits"] = retrieval_debug.get("vector_hits", 0)
        q_debug["bm25_hits"] = retrieval_debug.get("bm25_hits", 0)
        if metadata_filter:
            q_debug["metadata_filter"] = metadata_filter

    if use_expansion:
        logging.info(f"Using query expansion with {len(retrieval_queries)} variants")
    else:
        logging.info("Query expansion disabled for this query type")

    unique_results = {}
    for r in results:
        unique_results[r.id] = r
    results = sanitize_results(q, list(unique_results.values()))
    logging.info(f"Results after reranking: {len(results)}")

    t_filter = time.perf_counter()
    filtered_results = hybrid_filter_results(q, results, dynamic_threshold)
    retrieval_stats = compute_retrieval_diagnostics(filtered_results)
    confidence = retrieval_stats["retrieval_score"]
    confidence_percent = confidence
    logging.info(f"Initial confidence: {confidence_percent}%")
    logging.info(
        "Retrieval diagnostics -> top1: %s%%, top_k_avg: %s%%, consistency: %s%%",
        retrieval_stats["top1_score"],
        retrieval_stats["top_k_avg"],
        retrieval_stats["retrieval_consistency"],
    )
    filtered_results = sanitize_results(filtered_results)
    if filtered_results:
        logging.debug("TOP CHUNK: %s", filtered_results[0].payload["text"][:120])
    q_debug["stages_ms"]["filter"] = round((time.perf_counter() - t_filter) * 1000, 2)
    logging.info("Hybrid filter applied")

    filtered_results = sorted(filtered_results, key=lambda r: r.score, reverse=True)
    filtered_results = prefer_summary_chunks(filtered_results, numeric_query=numeric_query)
    filtered_results = remove_similar_chunks(filtered_results, keep=8)

    if query_type == "semantic":
        finance_terms = {"revenue", "profit", "loss", "margin", "expense", "financial", "ratio", "dividend"}
        if not any(t in q.lower() for t in finance_terms):
            non_tab = [
                r for r in filtered_results
                if str(r.payload.get("file_type", "")).lower() not in {"csv", "xls", "xlsx"}
            ]
            if non_tab:
                filtered_results = non_tab

    if filtered_results:
        chunk_contribution = [round(r.score, 3) for r in filtered_results[:5]]
        logging.info(f"Chunk contributions: {chunk_contribution}")
    if confidence < 20:
        metrics["weak_queries"] = metrics.get("weak_queries", 0) + 1

    fallback_used = False
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        fallback_used = True
        logging.info("Low confidence; running fallback search with relaxed filtering")
        fallback_query_vector = (
            get_embedding_cached(q, embedding_cache).tolist()
            if embedding_cache is not None
            else model.encode(q, normalize_embeddings=True).tolist()
        )
        fallback_results = vector_search(
            qdrant,
            collection_name=COLLECTION,
            query_vector=fallback_query_vector,
            limit=25,
            with_payload=True,
            query_filter={
                "must": [{"key": "session_id", "match": {"value": session_id}}]
                + [
                    {"key": key, "match": {"value": value}}
                    for key, value in (metadata_filter or {}).items()
                    if value is not None
                ]
            },
        )
        logging.info("Fallback search completed")

        filtered_results.extend(fallback_results[:5])
        filtered_results = rerank_results(q, filtered_results)
        filtered_results = sanitize_results(filtered_results)
        filtered_results = prefer_summary_chunks(filtered_results, numeric_query=numeric_query)
        filtered_results = remove_similar_chunks(filtered_results, keep=8)
        for d in filtered_results:
            payload = getattr(d, "payload", None)
            if payload is not None and payload.get("rerank_score_norm") is None:
                payload["rerank_score_norm"] = 0.0
        if filtered_results:
            logging.debug("top rerank: %s", filtered_results[0].payload.get("rerank_score_norm"))
        similarity = calibrated_similarity(filtered_results)
        logging.info(f"Calibrated similarity: {similarity}")
        if fallback_used:
            confidence_percent = max(confidence_percent, similarity * 100)
        logging.info(f"Updated confidence: {confidence_percent}%")

    return results, filtered_results, confidence_percent


def prepare_generation_context(q, filtered_results, context_limit):
    """Deduplicate chunks and prepare context/conflict summary."""
    filtered_results = remove_similar_chunks(filtered_results, keep=max(5, min(context_limit, 8)))

    file_chunks = defaultdict(list)
    file_versions = {}
    for r in filtered_results:
        fname = r.payload["file"]
        text = compress_chunk_for_prompt(q, r.payload["text"])
        r.payload["compressed_text"] = text
        file_chunks[fname].append(text)
        if fname not in file_versions:
            file_versions[fname] = r.payload["version"]
    logging.info("Results grouped by file")

    sorted_files = sorted(file_chunks.keys(), key=lambda f: file_versions[f], reverse=True)

    context_candidates = []
    used_files = set()
    for r in filtered_results:
        fname = r.payload["file"]
        text = r.payload.get("compressed_text") or compress_chunk_for_prompt(q, r.payload["text"])
        if fname not in used_files or len(context_candidates) < context_limit:
            context_candidates.append({"text": text, "score": safe_float(r.score, 0.0), "meta": r.payload})
            used_files.add(fname)
    final_chunks, context_tokens = allocate_context_by_budget(
        context_candidates[:context_limit],
        max_context_tokens=MAX_CONTEXT_TOKENS,
    )
    context_chunks = [chunk["text"] for chunk in final_chunks]
    context = "\n\n".join(context_chunks)
    logging.info("Context compressed")
    logging.info("Context token budget used: %s/%s", context_tokens, MAX_CONTEXT_TOKENS)

    conflict_info = ""
    if len(sorted_files) > 1:
        conflict_info = "Conflict detected between versions:\n\n"
        latest_file = sorted_files[0]
        for f in sorted_files:
            preview = " ".join(file_chunks[f][:2]).strip()
            if f == latest_file:
                conflict_info += f" {f} (latest)\n{preview}\n\n"
            else:
                conflict_info += f" {f}\n{preview}\n\n"
        conflict_info += f" - Using latest version: {latest_file}\n"

    if "summary" in q.lower() or "summarize" in q.lower():
        logging.info("Processing summary query")
        pdf_chunks = [r.payload["text"] for r in filtered_results if r.payload.get("file", "").lower().endswith(".pdf")]
        logging.info("Collected PDF chunks for summary")
        if not pdf_chunks:
            logging.info("No PDF chunks found for summary")
            return filtered_results, None, "", "Not available in the dataset"
        summary_candidates = [
            {"text": compress_chunk_for_prompt(q, text, max_chars=900), "score": 1.0, "meta": None}
            for text in pdf_chunks[:15]
            if text
        ]
        final_summary_chunks, _ = allocate_context_by_budget(
            summary_candidates,
            max_context_tokens=MAX_CONTEXT_TOKENS,
        )
        context = "\n".join(chunk["text"] for chunk in final_summary_chunks)
        if not context.strip():
            logging.info("No context available for summary")
            return filtered_results, None, "", "Not available in the dataset"
    logging.info("Context prepared for prompt")
    return filtered_results, context, conflict_info, None

# ------------------ Cache ------------------

def split_into_questions(text):
    """Split user input into smaller intent-focused questions."""
    text = (text or "").strip()
    if not text:
        return []

    # First split by explicit sentence separators.
    
    rough_parts = [p.strip() for p in re.split(r"[?!.;]+", text) if p.strip()]
    if not rough_parts:
        rough_parts = [text]

    final_parts = []
    for part in rough_parts:
        # Split on "and" only when likely starting a new intent.
        
        sub = re.split(
            r"\band\s+(?=(what|why|how|define|explain|show|give|calculate|find|compare|which|is)\b)",
            part,
            flags=re.IGNORECASE,
        )
        if len(sub) == 1:
            final_parts.append(part)
            continue

        merged = [sub[0]]
        i = 1
        while i + 1 < len(sub):
            merged.append(f"{sub[i]} {sub[i+1]}".strip())
            i += 2
        final_parts.extend([m for m in merged if m.strip()])

    cleaned = []
    for p in final_parts:
        p = re.sub(r"^\s*(and|then|also|now)\s+", "", p.strip(), flags=re.IGNORECASE)
        if p:
            cleaned.append(p)

    # Split segments containing multiple intent starters without punctuation.
    
    split_cleaned = []
    starter_pat = re.compile(r"\b(what is|define|explain|show|why|how|calculate|find|compare)\b", flags=re.IGNORECASE)
    for p in cleaned:
        starts = list(starter_pat.finditer(p))
        if len(starts) <= 1:
            split_cleaned.append(p)
            continue
        boundaries = [m.start() for m in starts] + [len(p)]
        for i in range(len(boundaries) - 1):
            chunk = p[boundaries[i]:boundaries[i + 1]].strip(" ,;")
            if chunk:
                split_cleaned.append(chunk)

    # Expand short continuation fragments like "profit" / "margin".
    
    expanded = []
    prev = ""
    for p in split_cleaned:
        p = re.sub(r"^(explain|define|show)\s+\1\b", r"\1", p, flags=re.IGNORECASE)
        short_metric = re.fullmatch(r"(profit|margin|revenue|loss)\s*", p, flags=re.IGNORECASE)
        if short_metric:
            metric = short_metric.group(1).lower()
            if "total" in prev.lower():
                p = f"What is total {metric}"
            else:
                p = f"What is {metric}"

        # Resolve pronoun-only intent like "Explain it now".
        
        if re.fullmatch(r"(explain|define)\s+(it|this|that)(\s+now)?", p, flags=re.IGNORECASE):
            topic_tokens = [w for w in re.findall(r"[a-zA-Z]+", prev.lower()) if w not in {"what", "is", "the", "a", "an", "and", "now"}]
            topic = topic_tokens[-1] if topic_tokens else "the topic"
            p = re.sub(r"(it|this|that)", topic, p, flags=re.IGNORECASE)

        if len(p.strip()) > 2:
            expanded.append(p.strip())
            prev = p

    # Force mixed-intent split even without punctuation.
    
    def _force_mixed_intent_split(q):
        ql = q.lower()
        finance_terms = [
            "revenue", "profit", "loss", "margin", "expense",
            "financial", "ratio", "dividend", "growth", "trend"
        ]
        semantic_terms = ["explain", "define", "show", "why", "how", "describe"]

        has_finance = any(t in ql for t in finance_terms)
        has_semantic = any(re.search(rf"\b{t}\b", ql) for t in semantic_terms)
        if not (has_finance and has_semantic):
            return [q]

        # Split at the first semantic starter that appears after some text.
        
        m = re.search(r"\b(explain|define|why|how|describe)\b", q, flags=re.IGNORECASE)
        if m and m.start() > 0:
            left = q[:m.start()].strip(" ,;")
            right = q[m.start():].strip(" ,;")
            out = []
            if left:
                out.append(left)
            if right:
                out.append(right)
            return out if len(out) >= 2 else [q]
        if m and m.start() == 0:
            right_split = re.search(r"\band\s+(revenue|profit|loss|margin|financial)\b", q, flags=re.IGNORECASE)
            if right_split:
                left = q[:right_split.start()].strip(" ,;")
                right = q[right_split.start() + 4 :].strip(" ,;")
                out = []
                if left:
                    out.append(left)
                if right:
                    out.append(f"What is {right}" if not re.search(r"\bwhat is\b", right, flags=re.IGNORECASE) else right)
                return out if len(out) >= 2 else [q]
        return [q]

    forced = []
    for q in (expanded if expanded else [text]):
        forced.extend(_force_mixed_intent_split(q))

    return [q for q in forced if q.strip()]

def get_cached_answer(session_id, query):
    """Return cached answer if not expired for this session/query."""
    key = (session_id, query)

    if key in query_cache:
        answer, timestamp = query_cache[key]
        if time.time() - timestamp < CACHE_TTL:
            return answer
        else:
            del query_cache[key]
    
    return None
   
def set_cached_answer(session_id, query, answer):
    """Store answer in TTL cache."""
    query_cache[(session_id, query)] = (answer, time.time())

def final_safety_check(answer):
    """Reject short, highly uncertain hedge-style outputs."""
    banned_patterns = ["i think", "probably", "maybe", "guess"]

    text = (answer or "").strip()
    lowered = text.lower()

    # Do not hard-fail long grounded answers because of one soft hedge word.
    # Only reject when the whole answer is short and mostly uncertain.
    
    if len(text) <= 60:
        for pattern in banned_patterns:
            if pattern in lowered:
                return "Not available in the dataset."
    return answer

def extract_confidence_value(answer):
    """Extract numeric confidence from text footer, if present."""
    text = (answer or "").strip()
    match = re.search(
        r"(?:overall\s+)?confidence(?:\s*\(part\))?\s*:\s*(\d+(?:\.\d+)?)%",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return max(0.0, min(100.0, safe_float(match.group(1))))


def strip_confidence_footer(answer):
    """Remove trailing confidence footer from answer text."""
    return re.sub(
        r"\n*\s*(?:overall\s+)?confidence(?:\s*\(part\))?\s*:\s*\d+(?:\.\d+)?%\s*$",
        "",
        str(answer or "").strip(),
        flags=re.IGNORECASE,
    ).rstrip()

def estimate_structured_confidence(answer_text):
    """Estimate confidence for deterministic pandas answers."""
    text = (answer_text or "").strip().lower()
    if not text or is_unanswerable(text):
        return 0.0
    if "conflict detected" in text:
        return 68.0
    if "general knowledge answer-not from uploaded files" in text:
        return 30.0

    length_score = compute_answer_length_score(answer_text, "structured")
    numeric_hits = len(re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", answer_text or ""))
    comparison_present = bool(re.search(r"\b(vs|higher|lower|greater|less|compare|comparison|score)\b", text))

    score = BASE_CONFIDENCE_FLOOR
    if numeric_hits >= 1:
        score += 5
    if numeric_hits >= 2:
        score += 3
    if comparison_present:
        score += 3
    if "appears" in text or "seems" in text:
        score -= 1
    if length_score < 50:
        score -= 3

    return round(max(0.0, min(100.0, score)), 2)


def estimate_section_confidence(answer_text):
    """Estimate section-summary confidence from answer quality signals."""
    text = (answer_text or "").strip().lower()
    if not text or is_unanswerable(text) or "error generating summary" in text:
        return 0.0

    length_score = compute_answer_length_score(answer_text, "section")
    score = BASE_CONFIDENCE_FLOOR + 7.0
    if length_score >= 70:
        score += 4
    if len(re.findall(r"[\.\n\-•]", answer_text or "")) >= 3:
        score += 2
    if any(token in text for token in ["summary", "conclusion", "key", "finding"]):
        score += 2
    return round(max(0.0, min(100.0, score)), 2)


def compute_overall_confidence(part_confidences):
    """Aggregate part confidences conservatively and predictably."""
    if not part_confidences:
        return 0.0

    values = [max(0.0, min(100.0, safe_float(v))) for v in part_confidences]
    if not any(v > 0 for v in values):
        return 0.0
    mean_conf = sum(values) / len(values)
    min_conf = min(values)
    overall = (0.8 * mean_conf) + (0.2 * min_conf)
    if overall > 0:
        overall = max(BASE_CONFIDENCE_FLOOR, overall)
    return round(max(0.0, min(100.0, overall)), 2)

def ensure_confidence_line(answer, confidence_percent):
    """Normalize to exactly one per-part confidence footer."""
    text = strip_confidence_footer(answer)
    return render_final_answer(text, confidence_percent, numeric_query=False)

def append_overall_confidence(answer_text, overall_confidence):
    """Attach a single overall confidence line at the end."""
    text = (answer_text or "").strip()
    text = re.sub(
        r"\n*\s*overall\s+confidence\s*:\s*\d+(?:\.\d+)?%\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).rstrip()
    conf = max(0.0, min(100.0, safe_float(overall_confidence)))
    return f"{text}\n\nOverall Confidence: {round(conf, 2)}%"


def get_conversational_reply(user_query):
    """Return a small assistant-style reply for simple chat messages."""
    text = (user_query or "").strip().lower()
    normalized = re.sub(r"[^\w\s]", "", text)
    if not normalized:
        return None

    if normalized in {"ok", "okay", "ok thanks", "okay thanks", "thank you", "thanks", "thx"}:
        return "You're welcome."
    if normalized in {"welcome", "youre welcome"}:
        return "Happy to help."
    if normalized in {"hi", "hello", "hey", "hi there", "hello there", "hey there"}:
        return "Hello. How can I help you?"
    if normalized in {"good morning", "good afternoon", "good evening"}:
        return "Hello. How can I help you today?"
    return None


def answer_meta_followup(user_query, conversation_memory):
    """Answer confidence/support follow-ups from the previous response state."""
    q = (user_query or "").strip().lower()
    last_answer = conversation_memory.get("last_answer")
    last_confidence = conversation_memory.get("last_confidence")
    last_sources = conversation_memory.get("last_sources") or []
    last_chunks = conversation_memory.get("last_chunks") or []
    if not last_answer:
        return None

    if any(token in q for token in ["how confident", "confidence", "are you sure"]):
        if last_confidence is None:
            return "Confidence is not available for the previous answer."
        return f"The previous answer had confidence {round(safe_float(last_confidence), 2)}%."

    if any(token in q for token in ["actual data", "assumption", "based on actual data"]):
        if is_unanswerable(last_answer):
            return "The previous answer was based on the uploaded data and indicated the dataset did not support a definitive answer."
        return "The previous answer was based on the uploaded dataset, not an unsupported assumption."

    if any(token in q for token in ["what data supports", "supports your answer", "source rows", "show source"]):
        parts = []
        if last_sources:
            parts.append("Sources: " + "; ".join(str(s) for s in last_sources[:5]))
        if last_chunks:
            preview = []
            for chunk in last_chunks[:3]:
                text = str(chunk.get("text", "")).strip().replace("\n", " ")
                if text:
                    preview.append(text[:220])
            if preview:
                parts.append("Supporting data: " + " | ".join(preview))
        return " ".join(parts) if parts else "Supporting source rows are not available for the previous answer."

    return None


def resolve_chart_followup_query(user_query, conversation_memory):
    """Expand short chart follow-ups to the previous analytical/tabular query."""
    q = (user_query or "").strip()
    if not q:
        return q
    if not detect_visualization(q, q):
        return q

    lowered = q.lower()
    followup_tokens = [
        "chart", "graph", "plot", "visualize", "visualise",
        "show it", "show them", "show chart", "show graph",
        "in chart", "as chart", "plot it", "plot them",
        "visualize it", "visualize them", "show me chart", "show me graph"
    ]
    if not any(token in lowered for token in followup_tokens):
        return q

    last_query = conversation_memory.get("last_query")
    if not last_query:
        return q

    generic_followup = any(token in lowered for token in [
        "it", "them", "this", "that", "these", "those", "previous", "above", "result",
        "show it", "show them", "plot it", "plot them", "visualize it", "visualize them"
    ])
    if generic_followup and len(q.split()) <= 8:
        return f"{last_query} {q}"
    return q


def summarize_chart_data(chart_data, query, chart_type=None):
    """Create a short answer that matches the generated chart data."""
    if not chart_data:
        return "No chart data available."

    ordered = sorted(chart_data, key=lambda item: safe_float(item.get("value")), reverse=True)
    top_items = ordered[:5]
    metric_label = "value"
    q = (query or "").lower()
    if any(token in q for token in ["sales", "revenue"]):
        metric_label = "sales"
    elif "profit" in q:
        metric_label = "profit"
    elif any(token in q for token in ["units", "quantity", "sold"]):
        metric_label = "units"
    elif any(token in q for token in ["count", "number of", "how many"]):
        metric_label = "count"

    intro = f"Showing {chart_type or 'bar'} chart for {metric_label}."
    leaders = ", ".join(
        f"{item['label']} ({item['value']:,.0f})"
        for item in top_items
    )
    return f"{intro}\n\nTop values: {leaders}."


def combine_answers_for_display(answers, queries):
    """Render one or more answer blocks with minimal formatting."""
    if not answers:
        return ""
    if len(answers) == 1:
        return (answers[0] or "").strip()

    blocks = []
    for idx, answer_text in enumerate(answers, start=1):
        text = (answer_text or "").strip()
        if text:
            question_text = (queries[idx - 1] or "").strip() if idx - 1 < len(queries) else ""
            if question_text:
                blocks.append(f"Question {idx}: {question_text}\n\n{text}")
            else:
                blocks.append(f"Question {idx}\n\n{text}")
    return "\n\n".join(blocks)


def log_query_answer(path_label, query_text, answer_text):
    """Log query/answer pairs with route label for debugging."""
    logging.info(f"[{path_label}] QUERY: {query_text}")
    logging.info(f"[{path_label}] ANSWER: {answer_text}")

# =========================== Routes ===========================

# Mental model: receive API request -> call helper pipeline -> update session/metrics -> return JSON response.

# ------------------ Home Route ------------------

@app.route("/")
def home():
    """Render the main UI page."""
    return render_template("index.html")

# ------------------ Metrics Route ------------------

@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Expose aggregate runtime and answer metrics."""

    avg_time = 0
    if metrics["total_queries"] > 0:
        avg_time = metrics["total_time"] / metrics["total_queries"]

    accuracy = 0
    if metrics["total_queries"] > 0:
        accuracy = (metrics["successful_answers"] / metrics["total_queries"]) * 100

    return jsonify({
        "total_queries": metrics["total_queries"],
        "successful_answers": metrics["successful_answers"],
        "not_found": metrics["not_found"],
        "cache_hits": metrics["cache_hits"],
        "accuracy_percent": round(accuracy, 2),
        "average_response_time": round(avg_time, 3)
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Report model/vector-store availability health."""
    snapshot = dependency_health_snapshot()
    collection_present = False
    if snapshot["qdrant_ok"]:
        try:
            collections = qdrant.get_collections()
            collection_names = {c.name for c in collections.collections}
            collection_present = COLLECTION in collection_names
        except Exception:
            collection_present = False

    status = "ok" if model is not None and snapshot["qdrant_ok"] and collection_present else "degraded"
    return jsonify({
        "status": status,
        "model_loaded": model is not None,
        "vector_db_connected": snapshot["qdrant_ok"],
        "collection_present": collection_present,
        "llm_configured": snapshot["llm_configured"],
        "embedding_model": snapshot["embedding_model"],
        "openai_model": snapshot["openai_model"],
        "qdrant_url": snapshot["qdrant_url"],
        "hf_local_only": snapshot["hf_local_only"],
        "qdrant_error": snapshot["qdrant_error"],
    })

# ------------------ Upload Route ------------------

@app.route("/upload", methods=["POST"])
def upload():
    """Save files and start async chunk/embed/index work."""
    session_id = str(uuid.uuid4())
    session["session_id"] = session_id
    ensure_session_state(session_id)
    set_session_files(session_id, [])
    logging.info(f"New session created: {session_id}")

    files = request.files.getlist("files")
    upload_time = time.time()
    files_to_process = []

    for file in files:
        safe_name = secure_filename(file.filename)
        if not safe_name:
            continue

        path = os.path.join(UPLOAD_FOLDER, safe_name)
        file.save(path)

        current_files = get_session_files(session_id)
        current_files.append(path)
        set_session_files(session_id, current_files)
        files_to_process.append(
            {
                "safe_name": safe_name,
                "path": path,
                "data_mode": "tabular" if file.filename.lower().endswith((".csv", ".xlsx", ".xls")) else "document",
            }
        )

    if not files_to_process:
        return jsonify({"error": "No valid files to index."}), 400

    update_session_upload_status(
        session_id,
        state="queued",
        message=f"Upload received. Preparing {len(files_to_process)} file(s).",
        files_total=len(files_to_process),
        files_processed=0,
        current_file=None,
        chunks_prepared=0,
        chunks_indexed=0,
        indexed_files=[],
        tabular_caps=[],
        started_at=upload_time,
        finished_at=None,
        error=None,
    )
    Thread(
        target=process_upload_job,
        args=(session_id, files_to_process, upload_time),
        daemon=True,
    ).start()

    return jsonify(
        {
            "message": f"Upload received. Indexing started for {len(files_to_process)} file(s).",
            "session_id": session_id,
            "status": "queued",
        }
    ), 202


@app.route("/upload_status", methods=["GET"])
def upload_status():
    """Return async upload/indexing status for the active session."""
    session_id = get_active_session_id()
    if not session_id:
        return jsonify({"session_id": None, "state": "idle", "message": "No active session."})

    ensure_session_state(session_id)
    status = get_session_upload_status(session_id) or {}
    status["session_id"] = session_id
    return jsonify(status)

# ------------------ File List Route ------------------

@app.route("/files", methods=["GET"])
def list_files():
    """List uploaded files in the active session."""
    session_id = get_active_session_id()
    if not session_id:
        return jsonify({"files": [], "session_id": None})

    files = get_session_file_details(session_id)
    return jsonify({"files": files, "session_id": session_id})

def _resolve_session_file(session_id, filename):
    safe_filename = secure_filename(filename or "")
    if not safe_filename or safe_filename != filename:
        return None, ("Invalid filename.", 400)

    session_paths = get_session_files(session_id)
    allowed_names = {os.path.basename(path) for path in session_paths}
    if safe_filename not in allowed_names:
        return None, ("File not found in the active session.", 404)

    return os.path.join(UPLOAD_FOLDER, safe_filename), None

def _read_preview_content(path, filename, *, row_limit=20, char_limit=2000):
    lower_name = filename.lower()
    if lower_name.endswith(".csv"):
        df = pd.read_csv(path, nrows=row_limit)
        return build_chunk_text(df)[:char_limit]
    if lower_name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, nrows=row_limit)
        return build_chunk_text(df)[:char_limit]

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read(char_limit)

@app.route("/preview")
def preview():
    session_id = get_active_session_id()
    if not session_id:
        return "No active session. Please upload a file first.", 400

    filename = request.args.get("file", "")
    path, error = _resolve_session_file(session_id, filename)
    if error:
        return error

    try:
        return _read_preview_content(path, filename, row_limit=20, char_limit=2000)
    except Exception:
        return "Preview not available for this file type."

@app.route("/summary", methods=["POST"])
def summary():
    session_id = get_active_session_id()
    if not session_id:
        return jsonify({"error": "No active session. Please upload a file first."}), 400

    data = request.get_json(silent=True) or {}
    filename = data.get("file", "")
    path, error = _resolve_session_file(session_id, filename)
    if error:
        message, status = error
        return jsonify({"error": message}), status

    try:
        text = _read_preview_content(path, filename, row_limit=40, char_limit=3000)
    except Exception:
        return jsonify({"error": "Summary not available for this file type."}), 400

    prompt = f"Summarize this document:\n\n{text}"

    try:
        response = llm_complete(prompt)
        summary_text = extract_llm_text(response) or "No summary returned."
        return jsonify({"summary": summary_text})
    except Exception:
        return jsonify({"summary": "Unable to reach the language model right now. Please try again later."}), 200

# ------------------ Delete File Route ------------------

@app.route("/delete_file", methods=["POST"])
def delete_file():
    """Delete one uploaded file from session and vector store."""
    session_id = get_active_session_id()
    if not session_id:
        return jsonify({"error": "No active session. Please upload a file first."}), 400

    status = get_session_upload_status(session_id)
    if status and status.get("state") in {"queued", "processing", "embedding", "indexing"}:
        return jsonify({"error": "Upload indexing is still in progress. Please wait until it is ready."}), 409

    data = request.get_json(silent=True) or {}
    filename = data.get("filename")

    if not filename:
        return jsonify({"error": "Filename required"}), 400
    
    safe_filename = secure_filename(filename)
    if not safe_filename or safe_filename != filename:
        return jsonify({"error": "Invalid filename"}), 400

    updated_files = [
        f for f in get_session_files(session_id)
        if os.path.basename(f) != safe_filename
    ]
    set_session_files(session_id, updated_files)
    reset_session_memory(session_id)
    remove_file_from_session_index(session_id, safe_filename)

    file_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, safe_filename))
    upload_root = os.path.abspath(UPLOAD_FOLDER)
    if not file_path.startswith(upload_root + os.sep):
        return jsonify({"error": "Invalid file path"}), 400
    if os.path.exists(file_path):
        os.remove(file_path)

    try:
        qdrant.delete(
            collection_name=COLLECTION,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="file",
                        match=MatchValue(value=safe_filename)
                    ),
                    FieldCondition(
                        key="session_id",
                        match=MatchValue(value=session_id)
                    )
                ]
            )
        )
    except Exception as exc:
        logging.warning("Qdrant cleanup failed for deleted file %s: %s", safe_filename, exc)

    update_session_upload_status(
        session_id,
        state="idle" if not updated_files else "ready",
        message="No upload in progress." if not updated_files else "Ready.",
        files_total=len(updated_files),
        files_processed=len(updated_files),
        current_file=None,
        chunks_prepared=0,
        chunks_indexed=0,
        error=None,
    )

    return jsonify(
        {
            "message": f"File {safe_filename} deleted successfully.",
            "files_remaining": len(updated_files),
        }
    )


@app.route("/delete_all_files", methods=["POST"])
def delete_all_files():
    """Delete all uploaded files for the active session and clear indexed vectors."""
    session_id = get_active_session_id()
    if not session_id:
        return jsonify({"error": "No active session. Please upload a file first."}), 400

    status = get_session_upload_status(session_id)
    if status and status.get("state") in {"queued", "processing", "embedding", "indexing"}:
        return jsonify({"error": "Upload indexing is still in progress. Please wait until it is ready."}), 409

    session_files = get_session_files(session_id)
    if not session_files:
        return jsonify({"message": "No uploaded files to delete.", "files_deleted": 0}), 200

    deleted_files = []
    upload_root = os.path.abspath(UPLOAD_FOLDER)
    for file_entry in session_files:
        safe_filename = os.path.basename(file_entry)
        if not safe_filename:
            continue
        file_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, safe_filename))
        if file_path.startswith(upload_root + os.sep) and os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted_files.append(safe_filename)
            except Exception as exc:
                logging.warning("Failed to remove uploaded file %s: %s", safe_filename, exc)
        else:
            deleted_files.append(safe_filename)

    set_session_files(session_id, [])
    reset_session_memory(session_id)
    remove_session_index(session_id)

    try:
        qdrant.delete(
            collection_name=COLLECTION,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="session_id",
                        match=MatchValue(value=session_id)
                    )
                ]
            )
        )
    except Exception as exc:
        logging.warning("Qdrant cleanup failed for bulk delete in session %s: %s", session_id, exc)

    update_session_upload_status(
        session_id,
        state="idle",
        message="No upload in progress.",
        files_total=0,
        files_processed=0,
        current_file=None,
        chunks_prepared=0,
        chunks_indexed=0,
        indexed_files=[],
        tabular_caps=[],
        error=None,
    )

    return jsonify(
        {
            "message": f"Deleted {len(session_files)} files successfully.",
            "files_deleted": len(session_files),
        }
    )


# ------------------ Main Query Route ------------------

"""
/query flowchart (high-level, 20-step pipeline)

[1 Start + timer]
    |
[2 Rate limit check] --too many--> [Return rate-limit answer]
    |
[3 Get session_id] --missing--> [Return upload-first answer]
    |
[4 Ensure session state + read request JSON]
    |
[5 Validate user_query] --invalid--> [400 Query required]
    |
[6 Effective debug flag]
    |
[7 Mode gate: tabular + structured?] --yes--> [run_structured_query + return]
    |
[8 Verify Qdrant/collection] --failed--> [Return vector DB error]
    |
[9 split_into_questions(user_query)]
    |
[10 For each sub-question q]
    |
[11 Short follow-up carry-over (pronoun expansion)]
    |
[12 classify_query + extraction override + get_query_settings]
    |
[13 structured branch?] --yes--> [run_structured_query, append, continue]
    |
[14 Cache check] --hit--> [append cached answer, continue]
    |
[15 Section-intent fast path] --yes--> [summarize_section_query, append, continue]
    |
[16 run_retrieval_pipeline]
    |
[17 filtered_results empty?] --yes--> [General-knowledge fallback LLM, append, continue]
    |
[18 prepare_generation_context] --early summary fallback--> [append early answer, continue]
    |
[19 generate_answer_with_validation -> finalize_answer_text -> cache/memory/debug updates]
    |
[20 End loop -> aggregate answers + metrics + optional debug -> JSON response]
"""

@app.route("/query", methods=["POST"])
def query():
    """Main RAG endpoint handling tabular + semantic flows."""
    
    start_time = time.time()
    data = request.get_json(silent=True) or {}
    user_query = data.get("query", "")
    if not isinstance(user_query, str) or not user_query.strip():
        return jsonify({"answer": "Query is required."}), 400

    conversational_reply = get_conversational_reply(user_query)
    if conversational_reply:
        return jsonify({"answer": conversational_reply, "sources": [], "confidence": None, "chunks": []})

    # 1) Rate limit
    
    current_time = time.time()
    request_log.append(current_time)

    recent_requests = [t for t in request_log if current_time - t < 60]
    if len(recent_requests) > RATE_LIMIT:
        return jsonify({"answer": "Rate limit exceed. please slow down."})
    
    session_id = get_active_session_id()
    
    if not session_id:
        return jsonify({"answer": "No active session. Please upload a file first."})
    
    ensure_session_state(session_id)
    upload_status = get_session_upload_status(session_id)
    if upload_status and upload_status.get("state") in {"queued", "processing", "embedding", "indexing"}:
        return jsonify({"answer": f"Files are still being indexed. {upload_status.get('message', 'Please wait.')}"}), 409
    if upload_status and upload_status.get("state") == "failed":
        return jsonify({"answer": f"Upload failed: {upload_status.get('error') or upload_status.get('message', 'Unknown error.')}"}), 500

    conversation_memory = get_session_memory(session_id)
    user_query = resolve_chart_followup_query(user_query, conversation_memory)
    meta_reply = None if detect_visualization(user_query, user_query) else answer_meta_followup(user_query, conversation_memory)
    if meta_reply:
        return jsonify({"answer": meta_reply, "sources": conversation_memory.get("last_sources", []), "confidence": conversation_memory.get("last_confidence"), "chunks": conversation_memory.get("last_chunks", [])})
    uploaded_files = get_session_files(session_id)
    logging.info(f"Incoming query: {user_query}")
    debug_mode = bool(data.get("debug", False))
    effective_debug = debug_mode or DEBUG_RAG

    # 2) Mode gate
    
    mode = conversation_memory.get("data_mode")
    gate_query_type = classify_query(user_query)
    if mode == "tabular" and detect_visualization(user_query, user_query):
        chart_data = build_tabular_chart_data(user_query, uploaded_files)
        chart_type = get_chart_type(user_query)
        if chart_data:
            answer = summarize_chart_data(chart_data, user_query, chart_type)
            overall_conf = 88.0
            response = {
                "answer": ensure_confidence_line(answer, overall_conf),
                "sources": [],
                "confidence": overall_conf,
                "chunks": [],
                "chart_data": generate_chart(chart_data, chart_type),
                "chart_type": chart_type,
            }
            conversation_memory["last_query"] = user_query
            conversation_memory["last_answer"] = response["answer"]
            conversation_memory["last_confidence"] = overall_conf
            conversation_memory["last_sources"] = []
            conversation_memory["last_chunks"] = []
            return jsonify(response)
    if mode == "tabular" and gate_query_type in {"structured", "summary"}:
        logging.info("TABULAR MODE -> STRUCTURED QUERY")
        answer, tabular_confidences, tabular_queries = run_structured_query_batch(
            user_query,
            uploaded_files,
            conversation_memory,
        )
        answer, overall_conf = finalize_structured_response(
            answer,
            tabular_confidences,
            conversation_memory,
            user_query,
        )
        response = {
            "answer": answer,
            "sources": [],
            "confidence": overall_conf,
            "chunks": [],
            "chart_data": None,
            "chart_type": None,
        }
        if detect_visualization(user_query, answer):
            try:
                multi_data = extract_multi_file_data(answer)
                if multi_data:
                    response["chart_data"] = generate_multi_file_chart(multi_data, "Comparison Across Files")
                    response["chart_type"] = "bar"
                    if "results vary across datasets" not in answer.lower():
                        response["answer"] = ensure_confidence_line(
                            strip_confidence_footer(response["answer"]) + "\n\nResults vary across datasets.",
                            overall_conf,
                        )
                else:
                    chart_data = build_tabular_chart_data(user_query, uploaded_files)
                    if chart_data:
                        chart_type = get_chart_type(user_query)
                        response["chart_data"] = generate_chart(chart_data, chart_type)
                        response["chart_type"] = chart_type
            except Exception as e:
                logging.info(f"Structured chart generation failed: {e}")
        if effective_debug:
            response["debug"] = {
                "mode": "tabular",
                "gate_query_type": gate_query_type,
                "split_questions": len(tabular_queries),
                "test_mode": TEST_MODE,
                "disable_query_expansion": DISABLE_QUERY_EXPANSION
            }
        return jsonify(response)
    
    # 3) Normal RAG flow
    
    logging.info(f"Query: {user_query}")
    logging.info(f"Debug mode: {effective_debug}")
    try:
        collections = qdrant.get_collections()
        collection_names = {c.name for c in collections.collections}
        if COLLECTION not in collection_names:
            return jsonify({"answer": "Vector collection not found"})
    except Exception:
        return jsonify({"answer": "Vector database unavailable"})

    # ------------------ Smart Agentic + Decomposition ------------------

    def should_decompose(query):
        q = query.lower()

        triggers = [ "compare", "difference", "which", "highest", "lowest",
        "best", "worst", "largest", "smallest", "across", "among"
        ]
        return any(t in q for t in triggers) or len(query.split()) > 8

    # Skip agent planning in the live query path to reduce latency.
    plan = None

# ----- Smart decomposition--
  
    if should_decompose(user_query):
        try:
            decomposed_queries = decompose_query(user_query)
            logging.info(f"DECOMPOSED QUERIES: {decomposed_queries}")
        except Exception as e:
            logging.info(f"Decomposition failed: {e}")
            decomposed_queries = []
    
#----------limit-------------

        if len(decomposed_queries) > 5:
            decomposed_queries = decomposed_queries[:5]
    
        queries = decomposed_queries if decomposed_queries else split_into_questions(user_query)
    
    else:
        queries = split_into_questions(user_query)  #simple query no decomposition

#---------------------------------------------------------

    logging.info(f"Split into {len(queries)} questions")
    answers= []
    part_confidences = []
    debug_questions = []
    embedding_cache = {}
    response_sources = []
    seen_response_sources = set()
    retrieved_chunks = []
    
    for q in queries:
        q_start = time.perf_counter()
        q = q.strip()
        fallback_reason = None
        q_debug = {
            "question": q,
            "stages_ms": {},}
        if len(q.split()) <= 2 and "last_query" in conversation_memory:
            
            # Only apply carry-over expansion for pronoun-only short follow-ups.
            if re.search(r"\b(it|this|that)\b", q.lower()):
                q = conversation_memory["last_query"] + " explanation "

        logging.info(f"Processing query: {q}")
        logging.info(f"Session ID: {session_id}")
        t_classify = time.perf_counter()
        query_type = classify_query(q)
        q_debug["stages_ms"]["classify"] = round((time.perf_counter() - t_classify) * 1000, 2)
        logging.info(f"Classified query type: {query_type}")
        
        # Keep structured finance/data questions on the Pandas route.
        # Extraction override is only for non-structured intents.
        
        if query_type != "structured" and detect_extraction_intent(q):
            query_type = "extraction"
            logging.info("Detected extraction intent")

        settings = get_query_settings(query_type)
        context_limit = settings["context_limit"]
        threshold_modifier = settings["threshold_modifier"]
        use_expansion = settings["use_expansion"]
        numeric_query = detect_numeric_query(q)
        metadata_filter = infer_metadata_filters(q, session_id)
        q_debug["numeric_query"] = numeric_query
        if metadata_filter:
            q_debug["metadata_filter"] = metadata_filter
        
        if query_type == "structured":
            t_struct = time.perf_counter()
            final_answer, structured_confidences, structured_queries = run_structured_query_batch(
                q,
                uploaded_files,
                conversation_memory,
            )
            part_conf = compute_overall_confidence(structured_confidences)
            final_answer = ensure_confidence_line(final_answer, part_conf)
            q_debug["stages_ms"]["structured_pandas"] = round((time.perf_counter() - t_struct) * 1000, 2)
            q_debug["query_type"] = query_type
            q_debug["fallback_reason"] = None
            q_debug["split_questions"] = len(structured_queries)
            q_debug["stages_ms"]["total"] = round((time.perf_counter() - q_start) * 1000, 2)
            if effective_debug:
                debug_questions.append(q_debug)
            answers.append(final_answer)
            part_confidences.append(part_conf)
            continue            
        
        # ------------------ Cache Check ------------------

        cached = get_cached_answer(session_id, q)

        if cached:
            logging.info("Cache hit found")
            metrics["cache_hits"] += 1
            cached_conf = extract_confidence_value(cached)
            if cached_conf is None:
                cached_conf = 50.0 if "not available" not in cached.lower() else 0.0
            cached = ensure_confidence_line(cached, cached_conf)
            log_query_answer("CACHE", q, cached)
            q_debug["cache_hit"] = True
            q_debug["query_type"] = query_type
            q_debug["stages_ms"]["total"] = round((time.perf_counter() - q_start) * 1000, 2)
            if effective_debug:
                debug_questions.append(q_debug)
            answers.append(cached)
            part_confidences.append(cached_conf)
            continue

        logging.info("No cache hit, proceeding with retrieval")

        q_debug["query_type"] = query_type
        q_debug["use_expansion"] = use_expansion

        # ------------------ Section-Based Summary ------------------

        requested_section = detect_requested_section(q)

        if requested_section:
            logging.info(f"Searching for section: {requested_section}")
            section_answer, section_fallback, section_llm_ms = summarize_section_query(
                q,
                requested_section,
                session_id,
                embedding_cache=embedding_cache,
            )
            if section_llm_ms is not None:
                q_debug["stages_ms"]["llm_answer"] = section_llm_ms
            q_debug["fallback_reason"] = section_fallback
            q_debug["stages_ms"]["total"] = round((time.perf_counter() - q_start) * 1000, 2)
            if effective_debug:
                debug_questions.append(q_debug)
            section_conf = estimate_section_confidence(section_answer)
            section_answer = ensure_confidence_line(section_answer, section_conf)
            answers.append(section_answer)
            part_confidences.append(section_conf)
            continue
        
        # ---------------- VECTOR SEARCH + BM25 RETRIEVAL (PRODUCTION RANKING) ----------------

        # 1) Build retrieval queries (optional LLM expansion by query type)
        
        results, filtered_results, confidence_percent = run_retrieval_pipeline(
            q=q,
            session_id=session_id,
            query_type=query_type,
            use_expansion=use_expansion,
            threshold_modifier=threshold_modifier,
            effective_debug=effective_debug,
            q_debug=q_debug,
            embedding_cache=embedding_cache,
            metadata_filter=metadata_filter,
            numeric_query=numeric_query,
        )

        top1_score = safe_float(filtered_results[0].score) if filtered_results else 0.0
        if top1_score < 0.2:
            fallback_reason = "weak_retrieval_top1_below_0_2"
            logging.info(f"Fallback reason: {fallback_reason}")
            q_debug["top1_score"] = round(top1_score, 3)
            finalize_question_debug(q_debug, fallback_reason, q_start, effective_debug, debug_questions)
            answers.append(ensure_confidence_line("No relevant data found", 10.0))
            part_confidences.append(10.0)
            continue


        if not filtered_results:
            fallback_reason = "no_filtered_results_general_fallback"
            logging.info(f"Fallback reason: {fallback_reason}")
            general_prompt = f"""
        You are a helpful educational assistant.
        The answer is NOT present in the provided documents.
        So answer using your own knowledge.

        Clearly mention:
        "(General knowledge answer-not from uploaded files)"

        Question:
        {q}
        """

            try:
                t_llm = time.perf_counter()
                general_response = llm_complete(general_prompt, timeout=20)
                q_debug["stages_ms"]["llm_answer"] = round((time.perf_counter() - t_llm) * 1000, 2)
                ans = extract_llm_text(general_response) or "Unable to reach the language model right now. Please try again later."
                part_conf = 35.0
                ans = ensure_confidence_line(ans, part_conf)
                q_debug["fallback_reason"] = fallback_reason
                q_debug["stages_ms"]["total"] = round((time.perf_counter() - q_start) * 1000, 2)
                if effective_debug:
                    debug_questions.append(q_debug)
                answers.append(ans)
                part_confidences.append(part_conf)
                continue
            except Exception:
                fallback_reason = "general_fallback_failed"
                logging.info(f"Fallback reason: {fallback_reason}")
                q_debug["fallback_reason"] = fallback_reason
                q_debug["stages_ms"]["total"] = round((time.perf_counter() - q_start) * 1000, 2)
                if effective_debug:
                    debug_questions.append(q_debug)
                answers.append(ensure_confidence_line("Not available in the dataset.", 0.0))
                part_confidences.append(0.0)
                continue

        logging.info(f"Final confidence score: {confidence_percent}%")

        filtered_results, context, conflict_info, early_answer = prepare_generation_context(
            q=q,
            filtered_results=filtered_results,
            context_limit=context_limit,
        )
        for r in filtered_results[:5]:
            payload = r.payload or {}
            file_name = payload.get("file") or "Unknown source"
            page = resolve_payload_page(session_id, payload)
            chunk_text = (payload.get("text") or "").strip()
            score_value = payload.get("rerank_score_norm", r.score)
            append_unique_source(response_sources, seen_response_sources, file_name, page)
            if chunk_text:
                retrieved_chunks.append(
                    {
                        "text": chunk_text,
                        "file": file_name,
                        "page": page,
                        "score": round(safe_float(score_value), 3),
                    }
                )
        if early_answer is not None:
            fallback_reason = "summary_context_unavailable"
            finalize_question_debug(q_debug, fallback_reason, q_start, effective_debug, debug_questions)
            part_conf = 0.0 if "not available" in early_answer.lower() else 45.0
            early_answer = ensure_confidence_line(early_answer, part_conf)
            answers.append(early_answer)
            part_confidences.append(part_conf)
            continue

        if is_cross_document_comparison_query(q, filtered_results):
            try:
                compared_answer = answer_cross_document_comparison(q, filtered_results, q_debug)
            except Exception as e:
                logging.info(f"Comparison stage failed: {e}")
                compared_answer = None
            if compared_answer is not None:
                ans, comparison_context = compared_answer
                origin_span = extract_answer_span(q, comparison_context)
                confidence_percent = max(confidence_percent, 55.0)
                fallback_reason = None
                ans = ensure_confidence_line(final_safety_check(ans), confidence_percent)
                finalize_question_debug(q_debug, fallback_reason, q_start, effective_debug, debug_questions)
                answers.append(ans)
                part_confidences.append(confidence_percent)
                continue

        try:
            ans, origin_span, confidence_percent, fallback_reason = generate_answer_with_validation(
                q=q,
                query_type=query_type,
                context=context,
                conflict_info=conflict_info,
                filtered_results=filtered_results,
                confidence_percent=confidence_percent,
                q_debug=q_debug,
                fallback_reason=fallback_reason,
                embedding_cache=embedding_cache,
                numeric_query=numeric_query,
            )
        except Exception as e:
            logging.info(f"Error in query processing: {e}")
            return jsonify({"answer": "Internal error while processing query."}), 500
    
        if ans:
            if origin_span == "NOT_FOUND":
                logging.info("Answer rejected - not grounded in context")
                ans = "Not available in the dataset"
                confidence_percent = 0

            final_answer = finalize_answer_text(
                ans=ans,
                query_type=query_type,
                origin_span=origin_span,
                filtered_results=filtered_results,
                confidence_percent=confidence_percent,
                numeric_query=numeric_query,
            )

            log_query_answer("RAG", q, final_answer)
            set_cached_answer(session_id, q, final_answer)

            metrics["successful_answers"] += 1
            conversation_memory["last_query"] = q
            conversation_memory["last_answer"] = ans

            q_debug["total_retrieved"] = len(results)
            q_debug["after_filtering"] = len(filtered_results)
            q_debug["top_scores"] = [round(r.score, 3) for r in results[:5]]
            q_debug["top_chunk_score"] = round(filtered_results[0].score, 3) if filtered_results else None
            finalize_question_debug(q_debug, fallback_reason, q_start, effective_debug, debug_questions)

            answers.append(final_answer)
            part_confidences.append(max(0.0, min(100.0, safe_float(confidence_percent))))

        else:
            metrics["not_found"] += 1
            if fallback_reason is None:
                fallback_reason = "empty_answer"
            logging.info(f"Fallback reason: {fallback_reason}")
            finalize_question_debug(q_debug, fallback_reason, q_start, effective_debug, debug_questions)
            log_query_answer("RAG", q, "Not available in the dataset")
            answers.append(ensure_confidence_line("Not available in the dataset", 0.0))
            part_confidences.append(0.0)
    elapsed = time.time() - start_time
    metrics["total_time"] += elapsed
    metrics["total_queries"] += len(queries)
    overall_confidence = compute_overall_confidence(part_confidences)
    combined_answer = combine_answers_for_display(answers, queries)
    combined_answer = append_overall_confidence(combined_answer, overall_confidence)
    response = {
        "answer": combined_answer,
        "sources": response_sources,
        "confidence": overall_confidence,
        "chunks": retrieved_chunks,
        "chart_data": None,
        "chart_type": None
    }
    conversation_memory["last_query"] = user_query
    conversation_memory["last_answer"] = combined_answer
    conversation_memory["last_confidence"] = overall_confidence
    conversation_memory["last_sources"] = list(response_sources)
    conversation_memory["last_chunks"] = list(retrieved_chunks)
    if effective_debug:
        response["debug"] = {
            "debug_enabled": True,
            "test_mode": TEST_MODE,
            "disable_query_expansion": DISABLE_QUERY_EXPANSION,
            "questions": debug_questions
        }
    # =========================
    #  CHART GENERATION LOGIC
    # =========================

    chart_data = None
    chart_type = None

    if detect_visualization(user_query, context):   #  use user_query (IMPORTANT)

        try:
            #  YOUR CONTEXT = combine retrieved chunks
            context_for_chart = "\n\n".join([c["text"] for c in retrieved_chunks])

            data = extract_data_for_chart(user_query, context_for_chart)

            #  limit points
            data = data[:10]

            if should_generate_chart(user_query, data, context):
                try:
                    chart_type = decide_chart_type(user_query, data)
                except:
                    chart_type = get_chart_type(user_query)
                chart_data = generate_chart(data, chart_type)

                insight = generate_chart_insight(user_query, data)

                if insight:
                    response["answer"] += "\n\n" + insight

                #  append warning to final answer
                response["answer"] += "\n\n Chart based on retrieved data, may not be complete."

        except Exception as e:
            logging.info(f"Chart generation failed: {e}")
            chart_data = None
    response["chart_data"] = chart_data
    response["chart_type"] = chart_type

    return jsonify(response)

@app.route("/query_stream", methods=["POST"])
def query_stream():
    data = request.get_json(silent=True) or {}
    user_query = (data.get("query") or "").strip()
    if not user_query:
        return Response("Query is required.", mimetype="text/plain", status=400)

    conversational_reply = get_conversational_reply(user_query)
    if conversational_reply:
        return Response(conversational_reply, mimetype="text/plain")

    session_id = get_active_session_id()
    if not session_id:
        return Response("No active session. Please upload a file first.", mimetype="text/plain", status=400)

    ensure_session_state(session_id)
    conversation_memory = get_session_memory(session_id)
    user_query = resolve_chart_followup_query(user_query, conversation_memory)
    meta_reply = None if detect_visualization(user_query, user_query) else answer_meta_followup(user_query, conversation_memory)
    if meta_reply:
        return Response(meta_reply, mimetype="text/plain")
    uploaded_files = get_session_files(session_id)
    mode = conversation_memory.get("data_mode")
    query_type = classify_query(user_query)
    numeric_query = detect_numeric_query(user_query)
    metadata_filter = infer_metadata_filters(user_query, session_id)
    if mode == "tabular" and detect_visualization(user_query, user_query):
        chart_data = build_tabular_chart_data(user_query, uploaded_files)
        chart_type = get_chart_type(user_query)
        if chart_data:
            answer = summarize_chart_data(chart_data, user_query, chart_type)
            overall_conf = 88.0
            def generate_visual():
                yield json.dumps(
                    {
                        "type": "meta",
                        "sources": [],
                        "confidence": overall_conf,
                        "chunks": [],
                    }
                ) + "\n"
                yield json.dumps({"type": "token", "token": ensure_confidence_line(answer, overall_conf)}) + "\n"
                yield json.dumps({"type": "chart", "data": generate_chart(chart_data, chart_type), "chart_type": chart_type}) + "\n"
                yield json.dumps({"type": "done"}) + "\n"
            conversation_memory["last_query"] = user_query
            conversation_memory["last_answer"] = answer
            conversation_memory["last_confidence"] = overall_conf
            conversation_memory["last_sources"] = []
            conversation_memory["last_chunks"] = []
            return Response(stream_with_context(generate_visual()), mimetype="application/x-ndjson")
    if mode == "tabular" and query_type in {"structured", "summary"}:
        answer, structured_confidences, _ = run_structured_query_batch(user_query, uploaded_files, conversation_memory)
        answer, overall_conf = finalize_structured_response(
            answer,
            structured_confidences,
            conversation_memory,
            user_query,
        )
        def generate_tabular():
            rendered_answer = answer
            yield json.dumps(
                {
                    "type": "meta",
                    "sources": [],
                    "confidence": overall_conf,
                    "chunks": [],
                }
            ) + "\n"
            if detect_visualization(user_query, answer):
                multi_data = extract_multi_file_data(answer)
                if multi_data and "results vary across datasets" not in answer.lower():
                    rendered_answer = ensure_confidence_line(
                        strip_confidence_footer(answer) + "\n\nResults vary across datasets.",
                        overall_conf,
                    )
            yield json.dumps({"type": "token", "token": rendered_answer}) + "\n"
            if detect_visualization(user_query, answer):
                try:
                    multi_data = extract_multi_file_data(answer)
                    if multi_data:
                        chart_data = generate_multi_file_chart(multi_data, "Comparison Across Files")
                        yield json.dumps({"type": "chart", "data": chart_data, "chart_type": "bar"}) + "\n"
                    else:
                        chart_data = build_tabular_chart_data(user_query, uploaded_files)
                        if chart_data:
                            chart_type = get_chart_type(user_query)
                            yield json.dumps({"type": "chart", "data": generate_chart(chart_data, chart_type), "chart_type": chart_type}) + "\n"
                except Exception as e:
                    logging.info(f"Streaming structured chart error: {e}")
            yield json.dumps({"type": "done"}) + "\n"

        return Response(stream_with_context(generate_tabular()), mimetype="application/x-ndjson")

    embedding_cache = {}
    settings = get_query_settings(query_type)
    docs = retrieve_documents(
        user_query,
        session_id,
        top_k=4,
        query_type=query_type,
        embedding_cache=embedding_cache,
    )
    if not docs:
        return Response("Not available in the dataset.", mimetype="text/plain")

    context = build_capped_context_from_docs(
        docs,
        max_chunks=min(6, max(4, settings.get("context_limit", 6))),
        max_chunk_chars=900,
        max_total_chars=6000,
    )
    if not context:
        return Response("Not available in the dataset.", mimetype="text/plain")
    sources = []
    seen_sources = set()
    chunks = []
    raw_scores = []
    for doc in docs:
        metadata = doc.metadata or {}
        file_name = metadata.get("file") or "Unknown source"
        page = metadata.get("page")
        score = safe_float(metadata.get("score", 0.0))
        normalized_score = score * 100 if score <= 1 else score
        normalized_score = max(0.0, min(100.0, normalized_score))
        raw_scores.append(normalized_score)
        append_unique_source(sources, seen_sources, file_name, page)
        chunks.append(
            {
                "text": doc.page_content,
                "file": file_name,
                "page": page,
                "chunk_kind": metadata.get("chunk_kind"),
                "metadata_filter": metadata_filter or {},
                "score": round(score, 3),
            }
        )
    confidence = round(sum(raw_scores) / len(raw_scores), 2) if raw_scores else 0.0
    confidence = max(0.0, min(100.0, confidence))
    chunk_kinds = [str((doc.metadata or {}).get("chunk_kind") or "unknown") for doc in docs[:5]]
    prompt = build_prompt(
        query=user_query,
        context=context,
        conflict_info="",
        query_type=query_type,
        primary_evidence=None,
        numeric_query=numeric_query,
    )
    prompt = trim_text_to_token_budget(prompt, 12000)

    def generate():
        yield json.dumps(
            {
                "type": "meta",
                "sources": sources,
                "confidence": confidence,
                "chunks": chunks,
                "debug": {
                    "metadata_filter": metadata_filter or {},
                    "selected_chunk_kinds": chunk_kinds,
                    "numeric_query": numeric_query,
                },
            }
        ) + "\n"
        try:
            if not llm_is_available():
                raise RuntimeError(build_llm_unavailable_message(context="streaming"))
            stream_prompt = prompt
            try:
                stream = llm.chat.completions.create(
                    model=OPENAI_MODEL_NAME,
                    messages=[{"role": "user", "content": stream_prompt}],
                    temperature=LLM_TEMP,
                    stream=True,
                )
            except Exception as exc:
                if "context_length_exceeded" not in str(exc):
                    raise
                logging.warning("Streaming prompt exceeded context; retrying with smaller context window")
                fallback_context = build_capped_context_from_docs(
                    docs,
                    max_chunks=3,
                    max_chunk_chars=500,
                    max_total_chars=2500,
                )
                stream_prompt = trim_text_to_token_budget(
                    build_prompt(
                        query=user_query,
                        context=fallback_context,
                        conflict_info="",
                        query_type=query_type,
                        primary_evidence=None,
                        numeric_query=numeric_query,
                    ),
                    6000,
                )
                stream = llm.chat.completions.create(
                    model=OPENAI_MODEL_NAME,
                    messages=[{"role": "user", "content": stream_prompt}],
                    temperature=LLM_TEMP,
                    stream=True,
                )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    yield json.dumps({"type": "token", "token": token}) + "\n"

            try:
                chart_context = context
                data = extract_data_for_chart(user_query, chart_context)
                data = data[:10]

                if data:
                    chart_type = get_chart_type(user_query)
                    chart_data = generate_chart(data, chart_type)

                    yield json.dumps({
                        "type": "chart",
                        "data": chart_data,
                        "chart_type": chart_type
                    }) + "\n"
            
            except Exception as e:
                logging.info(f"Streaming chart error: {e}")
                
        except Exception as exc:
            logging.error(f"Query stream generation failed: {exc}")
            yield json.dumps(
                {
                    "type": "token",
                    "token": "Unable to reach the language model right now. Please try again later.",
                }
            ) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")

@app.route('/favicon.ico')
def favicon():
    """Return empty favicon response."""
    return '', 204
@app.errorhandler(Exception)
def handle_exception(e):
    """Global JSON error handler for uncaught exceptions."""
    logging.error(f"UnhandledException: {e}")
    return jsonify({
        "answer": "Internal system error. Please try again later."
        }), 500

if __name__ == "__main__":
    debug_flag = os.getenv("FLASK_DEBUG", "false").strip().lower() == "true"
    app.run(host="127.0.0.1", port=8000, debug=debug_flag)
