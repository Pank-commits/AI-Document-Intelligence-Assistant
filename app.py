# ---------- SILENCE WARNINGS ----------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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
import pandas as pd
from flask import (
    Flask, 
    render_template, 
    request, 
    jsonify,
    session)
from werkzeug.utils import secure_filename
from qdrant_client.models import (
    PointStruct, 
    VectorParams, 
    Distance,
    Filter,
    FieldCondition,
    MatchValue)
from loader import load_file
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
import logging
import numpy as np
from threading import Lock
from retriever import HybridIndex, reranker
from retrieval import hybrid_retrieve
from collections import deque
from langchain_core.documents import Document
from tabular_engine import answer_tabular, to_number
from citation_builder import build_citations
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------ Initial Setup ------------------
ANSWER_SPAN_NOT_FOUND = "NOT_FOUND"

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
session_state_lock = Lock()
metrics = {
    "total_queries": 0,
    "cache_hits": 0,
    "total_time": 0.0,
    "successful_answers": 0,
    "not_found": 0
}

# ------------------ Model Initialization ------------------

model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(url="http://localhost:6333")
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")
llm = OpenAI(api_key=openai_api_key)

COLLECTION = os.getenv("COLLECTION_NAME", "all_files")
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONF", 15))
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

# ------------------ Session State Helpers ------------------

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


def get_session_files(session_id):
    """Get a copy of files tracked for the session."""
    with session_state_lock:
        return list(session_uploaded_files.get(session_id, []))


def set_session_files(session_id, files):
    """Replace the list of files tracked for the session."""
    with session_state_lock:
        session_uploaded_files[session_id] = list(files)


def get_session_memory(session_id):
    """Return mutable per-session conversational memory."""
    with session_state_lock:
        return session_conversation_memory.setdefault(session_id, {})


def set_session_hybrid_index(session_id, payload_list):
    """Build and store the session BM25 hybrid index from payloads."""
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


def remove_file_from_session_index(session_id, filename):
    """Remove a file's docs from the session hybrid index."""
    with session_state_lock:
        docs = session_index_docs.get(session_id, [])
        docs = [d for d in docs if d.metadata.get("file") != filename]
        session_index_docs[session_id] = docs
        session_hybrid_indexes[session_id] = HybridIndex(docs) if docs else None

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


def llm_complete(prompt, *, temperature=None, timeout=20):
    """Centralized LLM completion call for consistent settings."""
    return llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMP if temperature is None else temperature,
        timeout=timeout,
    )


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


def summarize_section_query(query_text, requested_section, session_id, embedding_cache=None):
    """Answer section-summary queries directly from section chunks."""
    if embedding_cache is None:
        section_vector = model.encode(requested_section, normalize_embeddings=True).tolist()
    else:
        section_vector = get_embedding_cached(requested_section, embedding_cache).tolist()
    section_results = qdrant.search(
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
        return response.choices[0].message.content.strip(), None, llm_ms
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
    logging.info("Collection and index created successfully")

except UnexpectedResponse as e:
    if "already exists" in str(e):
        logging.info("Collection already exists")
    else:
        raise e
    
# ------------------ Data Helpers ------------------

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
        "chunk_index": chunk_idx
    }
    
    

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


def answer_calculation(query, df, memory=None):
    """Handle rule-based financial calculations on tabular data."""
    col = None
    year = None
    memory = memory if memory is not None else {}
    
    q = query.lower()
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

    # Handle short carry-over fragments from split queries.
    if re.match(r"^and\s+\w+", q) and memory.get("last_structured_intent"):
        q = f"{memory['last_structured_intent']} {q.replace('and', '').strip()}"

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

    for word in query_tokens:
        for col_name in cols_lower:
            if word in col_name:
                col = cols_lower[col_name]
                break
        if col:
            break
    
    if not col:
        col = calc_numeric_cols[0]
        
    
    year_match = re.search(r"(20\d{2})", q)
    if year_match:
        year = int(year_match.group())
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

        final_answer = "Conflict detected:\n"
        for i, ans in enumerate(pandas_answer, 1):
            final_answer += f"File {i}: {ans}\n"
        return final_answer

    return "Not available in the dataset"


def grounding_score(answer, context):
    """
    Measures how much answer text comes from context
    Prevents hallucinations
    """
    import re
    if isinstance(context, list):
        context = " ".join(str(c) for c in context)
    elif context is None:
        context = ""
    else:
        context = str(context)

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "of", "to", "in", "on", "for", "and", "or", "as", "by", "with", "at",
        "from", "that", "this", "it", "its", "into", "about", "which", "what",
    }
    answer_words = {
        w for w in re.findall(r"\w+", str(answer).lower())
        if len(w) > 2 and w not in stop_words
    }
    context_words = {
        w for w in re.findall(r"\w+", context.lower())
        if len(w) > 2 and w not in stop_words
    }

    if not answer_words:
        return 0.0

    overlap = answer_words.intersection(context_words)
    lexical = len(overlap) / len(answer_words)

    # Add semantic support so paraphrases are not unfairly scored as zero.
    semantic = 0.0
    try:
        pair = [(str(answer)[:500], context[:1200])]
        score = reranker.predict(pair)
        raw = float(score[0]) if hasattr(score, "__len__") else float(score)
        semantic = 1 / (1 + np.exp(-raw))
    except Exception:
        semantic = 0.0

    return (0.65 * lexical) + (0.35 * semantic)

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
    finance_words = [
        "revenue", "profit", "loss", "margin", "expense",
        "financial", "ratio", "dividend", "growth", "trend", "company", "invest",
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


    # ---- FINANCE ----
    if any(word in q for word in finance_words):
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

def calculate_confidence(filtered_results):
    """Convert top retrieval scores into a confidence percentage."""
    if not filtered_results:
        return 0.0
    
    top_k = filtered_results[:5]
    scores = [max(0.0, min(safe_float(r.score), 1.0)) for r in top_k]
    top1 = scores[0]
    avg = sum(scores) / len(scores)
    # Emphasize top evidence while still considering consistency.
    blended = (0.65 * top1) + (0.35 * avg)
    return round(blended * 100, 2)

#---------------compute trust score-------------------

def compute_trust_score(retrieval_conf, semantic_sim, grounding, verifier_ok):
    """Blend retrieval, grounding, semantic and verifier signals."""

    retrieval_conf = safe_float(retrieval_conf) / 100
    semantic_sim = safe_float(semantic_sim)
    grounding = safe_float(grounding)

    score = 0
    retrieval_conf = min(max(retrieval_conf, 0), 1)
    grounding = min(max(grounding, 0), 1)
    semantic_sim = min(max(semantic_sim, 0), 1)

    score += retrieval_conf * 0.25
    score += grounding * 0.35
    score += semantic_sim * 0.25

    if verifier_ok:
        score += 0.15

    # Boost cases where both semantic and grounding are reasonably strong.
    if semantic_sim >= 0.75 and grounding >= 0.25:
        score += 0.10

    return round(min(score, 1.0) * 100, 2)

# ------------------ Retry/Reflection ------------------

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

        return response.choices[0].message.content.strip()

    except Exception as e:
        logging.info(f"Error regenerating answer: {e}")
        return previous_answer

# ------------------ Query Expansion ------------------

def expand_query_with_llm(query):
    """
    generate three alternatives search queries using llm
    improve retreival recall
    """
    prompt = f"""
rewrite the following question into three different search variations.
keep the meaning same.
return each variation on a new line.

Question:
{query}
"""
    
    try:
        response = llm_complete(prompt, temperature=EXPANSION_TEMP)

        expanded = response.choices[0].message.content.strip()
        variations = [
            line.strip("- ").strip()
            for line in expanded.split("\n")
            if line.strip()
        ]
        return [query] + variations[:3]
    
    except Exception as e:
        logging.info(f"Error expanding query: {e}")
        return [query]
    
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
        return response.choices[0].message.content.strip()
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
        return response.choices[0].message.content.strip()
    except:
        return None
    
# ------------------ Prompt Building ------------------

def build_prompt(query, context, conflict_info, query_type, primary_evidence=None):
    """Build query-type specific instruction prompt for answering."""

    # ---------------- OUTPUT FORMATS ----------------

    if query_type == "structured":
        output_instruction = """
Return JSON only:
{
  "answer": "...",
  "value": "...",
  "year": "...",
  "source_file": "..."
}
If fields missing, keep value null.
"""

    elif query_type == "summary":
        output_instruction = """
Write a concise structured summary:

Main Topic:
Key Points:
Key Findings:
Conclusion:
"""

    elif query_type == "extraction":
        output_instruction = """
Extract and present the exact fact/equation from the context.

Rules:
- Match the wording precisely
- Include all relevant details
- If not found exactly, say: Not available in the dataset
"""

    elif query_type == "analytical":
        output_instruction = """
Provide a reasoning-based analysis using the context.

Rules:
- Do NOT just list raw facts
- Explain causes/effects and implications from the context
- If data is insufficient, say: Insufficient evidence in the dataset
"""

    else:  # semantic
        output_instruction = """
Explain the concept clearly in your own words using the context.
Paraphrasing is allowed.
Do NOT require exact sentence match.
Target length: 4-6 sentences.
Include:
- a clear definition,
- 1-2 key supporting details from context,
- and a short concluding line.
Avoid one-line answers unless the user explicitly asks for brevity.
Use direct, confident wording for definitions.
Do not use hedging phrases like "can be inferred", "it seems", or "possibly" when context is sufficient.
"""

    # ---------------- CORE RAG RULES ----------------

    core_rules = """
GROUNDING RULES:

You must base the answer on the provided context.

Allowed:
- Rephrase
- Infer definitions
- Combine multiple sentences
- Convert paragraph -> explanation
- Recognize equations written differently

Not Allowed:
- Invent new facts
- Use outside knowledge unrelated to context

If the topic truly does not exist in the context:
Return exactly: Not available in the dataset
"""

    # ---------------- FINAL PROMPT ----------------

    prompt = f"""
You are an expert document question-answering AI.

{core_rules}

{output_instruction}

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
):
    """Generate answer, then verify/ground/retry before returning."""
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
    )

    t_llm = time.perf_counter()
    response = llm_complete(prompt)
    q_debug["stages_ms"]["llm_answer"] = round((time.perf_counter() - t_llm) * 1000, 2)

    ans = response.choices[0].message.content.strip()
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

    if "not available" not in ans.lower():
        if query_type in ["semantic", "analytical", "summary"] and not is_symbolic_answer(ans):
            if embedding_cache is None:
                answer_embedding = model.encode(ans, normalize_embeddings=True)
            else:
                answer_embedding = get_embedding_cached(ans, embedding_cache)
            similarity = float(answer_embedding @ query_embedding.T)
            logging.info(f"Semantic similarity: {similarity}")
            if similarity < 0.25:
                logging.info("Low semantic similarity detected")
                similarity = 0.15
        else:
            logging.info("Symbolic answer detected")
            similarity = 0.85

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
        verdict = verification_response.choices[0].message.content.strip().lower()
        logging.info(f"Verification verdict: {verdict}")
    else:
        logging.info("Verification not needed")

    grounding = 0.0
    answer_span = None
    verification_context = context
    context_list = []

    if "not available" not in ans.lower() and filtered_results:
        best_chunk = filtered_results[0].payload.get("text") or ""
        answer_span = extract_answer_span(q, best_chunk)
        if answer_span and answer_span != ANSWER_SPAN_NOT_FOUND:
            verification_context = answer_span
        if is_chemical_equation(ans):
            context_list = [r.payload.get("text") or "" for r in filtered_results]
            grounding = max(calibrated_grounding(ans, context_list), 0.85)
        else:
            grounding = safe_float(grounding_score(ans, verification_context))

    logging.info(f"Grounding score: {grounding}")
    verifier_ok = verdict == "yes"
    semantic_alignment = safe_float(grounding_score(ans, verification_context))
    trust = compute_trust_score(
        retrieval_conf=confidence_percent,
        semantic_sim=semantic_alignment,
        grounding=grounding,
        verifier_ok=verifier_ok,
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
            grounding = safe_float(grounding_score(ans, verification_context))
        semantic_alignment = safe_float(grounding_score(ans, verification_context))
        trust = compute_trust_score(
            retrieval_conf=confidence_percent,
            semantic_sim=semantic_alignment,
            grounding=grounding,
            verifier_ok=verifier_ok,
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
                grounding = safe_float(grounding_score(ans, retry_context))
            trust = compute_trust_score(confidence_percent, similarity, grounding, verifier_ok)
            logging.info(f"New trust score: {trust}")

    if trust < 30:
        logging.info("Low trust score, attempting self correction")
        improved = regenerate_answer_with_reflection(q, context, ans)
        new_ground = safe_float(grounding_score(improved, context))
        if embedding_cache is None:
            improved_embedding = model.encode(improved, normalize_embeddings=True)
        else:
            improved_embedding = get_embedding_cached(improved, embedding_cache)
        new_sim = float(improved_embedding @ query_embedding.T)
        new_trust = compute_trust_score(confidence_percent, new_sim, new_ground, True)

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
    if verdict == "no" and trust < 40:
        logging.info("Verification failed, setting not available")
        fallback_reason = "verification_failed_low_trust"
        logging.info(f"Fallback reason: {fallback_reason}")
        ans = "Not available in the dataset"

    return ans, origin_span, confidence_percent, fallback_reason


def finalize_answer_text(ans, query_type, origin_span, filtered_results, confidence_percent):
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
    final_answer += f"\n\nConfidence: {confidence_percent}%"
    return final_answer


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


def run_retrieval_pipeline(
    q,
    session_id,
    query_type,
    use_expansion,
    threshold_modifier,
    effective_debug,
    q_debug,
    embedding_cache=None,
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
        vector_limit=25,
        final_top_k=8,
        collect_debug=effective_debug,
        encode_query_fn=(lambda txt: get_embedding_cached(txt, embedding_cache)) if embedding_cache is not None else None,
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
        q_debug["keyword_hits"] = retrieval_debug.get("keyword_hits", 0)

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
    confidence = calculate_confidence(filtered_results)
    confidence_percent = confidence
    logging.info(f"Initial confidence: {confidence_percent}%")
    filtered_results = sanitize_results(filtered_results)
    if filtered_results:
        logging.debug("TOP CHUNK: %s", filtered_results[0].payload["text"][:120])
    q_debug["stages_ms"]["filter"] = round((time.perf_counter() - t_filter) * 1000, 2)
    logging.info("Hybrid filter applied")

    filtered_results = sorted(filtered_results, key=lambda r: r.score, reverse=True)

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

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        logging.info("Low confidence; running fallback search with relaxed filtering")
        fallback_query_vector = (
            get_embedding_cached(q, embedding_cache).tolist()
            if embedding_cache is not None
            else model.encode(q, normalize_embeddings=True).tolist()
        )
        fallback_results = qdrant.search(
            collection_name=COLLECTION,
            query_vector=fallback_query_vector,
            limit=25,
            with_payload=True,
            query_filter={"must": [{"key": "session_id", "match": {"value": session_id}}]},
        )
        logging.info("Fallback search completed")

        filtered_results.extend(fallback_results[:5])
        filtered_results = rerank_results(q, filtered_results)
        filtered_results = sanitize_results(filtered_results)
        for d in filtered_results:
            payload = getattr(d, "payload", None)
            if payload is not None and payload.get("rerank_score_norm") is None:
                payload["rerank_score_norm"] = 0.0
        if filtered_results:
            logging.debug("top rerank: %s", filtered_results[0].payload.get("rerank_score_norm"))
        similarity = calibrated_similarity(filtered_results)
        logging.info(f"Calibrated similarity: {similarity}")
        logging.info(f"Updated confidence: {confidence_percent}%")

    return results, filtered_results, confidence_percent


def prepare_generation_context(q, filtered_results, context_limit):
    """Deduplicate chunks and prepare context/conflict summary."""
    seen_texts = set()
    clean_chunks = []
    for r in filtered_results:
        text = r.payload["text"]
        if text[:200] not in seen_texts:
            clean_chunks.append(r)
            seen_texts.add(text[:200])
    filtered_results = clean_chunks

    file_chunks = defaultdict(list)
    file_versions = {}
    for r in filtered_results:
        fname = r.payload["file"]
        text = r.payload["text"]
        file_chunks[fname].append(text)
        if fname not in file_versions:
            file_versions[fname] = r.payload["version"]
    logging.info("Results grouped by file")

    sorted_files = sorted(file_chunks.keys(), key=lambda f: file_versions[f], reverse=True)

    context_chunks = []
    used_files = set()
    for r in filtered_results:
        fname = r.payload["file"]
        text = r.payload["text"]
        if fname not in used_files or len(context_chunks) < context_limit:
            context_chunks.append(text)
            used_files.add(fname)
    context = "\n\n".join(context_chunks[:context_limit])
    logging.info("Context compressed")

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
        context = "\n".join(pdf_chunks[:15])
        if not context.strip():
            logging.info("No context available for summary")
            return filtered_results, None, "", "Not available in the dataset"
        context = context[:6000]
    else:
        context = "\n\n".join(context_chunks[:context_limit])[:6000]
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


def log_query_answer(path_label, query_text, answer_text):
    """Log query/answer pairs with route label for debugging."""
    logging.info(f"[{path_label}] QUERY: {query_text}")
    logging.info(f"[{path_label}] ANSWER: {answer_text}")

# =========================== Routes ===========================

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
    collection_present = False
    try:
        collections = qdrant.get_collections()
        collection_names = {c.name for c in collections.collections}
        collection_present = COLLECTION in collection_names
        vector_db_connected = True
    except Exception:
        vector_db_connected = False

    status = "ok" if model is not None and vector_db_connected and collection_present else "degraded"
    return jsonify({
        "status": status,
        "model_loaded": model is not None,
        "vector_db_connected": vector_db_connected,
        "collection_present": collection_present
    })

# ------------------ Upload Route ------------------

@app.route("/upload", methods=["POST"])
def upload():
    """Upload files, chunk/embed, and index them for retrieval."""
    session_id = str(uuid.uuid4())
    session["session_id"] = session_id
    ensure_session_state(session_id)
    set_session_files(session_id, [])
    logging.info(f"New session created: {session_id}")

    files = request.files.getlist("files")

    all_payloads = []


    upload_time = time.time()

    for file in files:
        safe_name = secure_filename(file.filename)
        if not safe_name:
            continue

        path = os.path.join(UPLOAD_FOLDER, safe_name)
        file.save(path)

        if file.filename.lower().endswith(".csv"):
            conversation_memory = get_session_memory(session_id)
            conversation_memory["data_mode"] = "tabular"
        else:
            conversation_memory = get_session_memory(session_id)
            conversation_memory["data_mode"] = "document"
        
        current_files = get_session_files(session_id)
        current_files.append(path)
        set_session_files(session_id, current_files)

        chunks: list[Document] = load_file(path)

        sections = {}
        if not safe_name.lower().endswith((".csv", ".xlsx", ".xls")):
            full_text = "\n\n".join(
                c.page_content for c in chunks if getattr(c, "page_content", None)
            )
            sections = extract_sections(full_text)
            logging.info(f"Extracted {len(sections)} sections from {safe_name}")
        else:
            logging.info(f"Skipped section extraction for tabular file: {safe_name}")

        for sec_idx, (sec_name, sec_text) in enumerate(sections.items()):
            section_chunk_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{session_id}|{safe_name}|section|{sec_name}|{sec_idx}"
            ))
            all_payloads.append({
                "text": f"[SECTION:{sec_name.upper()}]\n{sec_text}",
                "file": safe_name,
                "version": upload_time,
                "session_id": session_id,
                "source": path,
                "file_type": os.path.splitext(safe_name)[1].lower().replace(".", ""),
                "section": sec_name.upper(),
                "row": None,
                "page": None,
                "chunk_id": section_chunk_id
            })

        for chunk_idx, c in enumerate(chunks):
            meta = normalize_metadata(getattr(c, "metadata", {}), path, chunk_idx)

            chunk_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{session_id}|{meta['source']}|{meta['page']}|{meta['row']}|{meta['chunk_index']}"
                ))
            all_payloads.append({
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
                "chunk_id": chunk_id
            })

    all_texts = [p["text"] for p in all_payloads if p.get("text")]
    all_payloads = [p for p in all_payloads if p.get("text")]

    logging.info(f"Loaded {len(all_texts)} text chunks from {len(set(p['file'] for p in all_payloads))} files")
    if not all_payloads:
        return jsonify({"error": "No valid files to index."}), 400

    vectors = model.encode(
        all_texts, 
        batch_size=120,
        normalize_embeddings=True,
        show_progress_bar=False
        )
                           

    points =[
        PointStruct(
            id=all_payloads[i]["chunk_id"],
            vector = vectors[i], 
            payload=all_payloads[i]
        ) 
        for i in range (len(all_payloads))      
    ]
    
    qdrant.upsert(collection_name=COLLECTION, points=points)
    set_session_hybrid_index(session_id, all_payloads)

    return jsonify({"message": f"{len(files)} file(s) indexed successfully!",
                    "session_id": session_id
                    })

# ------------------ File List Route ------------------

@app.route("/files", methods=["GET"])
def list_files():
    """List uploaded files in the active session."""
    session_id = get_active_session_id()
    if not session_id:
        return jsonify({"files": [], "session_id": None})

    files = []
    for path in get_session_files(session_id):
        files.append(os.path.basename(path))
        
    return jsonify({"files": files, "session_id": session_id})

# ------------------ Delete File Route ------------------

@app.route("/delete_file", methods=["POST"])
def delete_file():
    """Delete one uploaded file from session and vector store."""
    session_id = get_active_session_id()
    if not session_id:
        return jsonify({"error": "No active session. Please upload a file first."}), 400

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
    remove_file_from_session_index(session_id, safe_filename)
    
    qdrant.delete(
        collection_name=COLLECTION,
        filter=Filter (
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

    file_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, safe_filename))
    upload_root = os.path.abspath(UPLOAD_FOLDER)
    if not file_path.startswith(upload_root + os.sep):
        return jsonify({"error": "Invalid file path"}), 400
    if os.path.exists(file_path):
        os.remove(file_path)

    return jsonify({"message": f"File {safe_filename} deleted successfully."})


# ------------------ Main Query Route ------------------

@app.route("/query", methods=["POST"])
def query():
    """Main RAG endpoint handling tabular + semantic flows."""
    
    start_time = time.time()

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
    conversation_memory = get_session_memory(session_id)
    uploaded_files = get_session_files(session_id)
    data = request.get_json(silent=True) or {}
    user_query = data.get("query", "")
    if not isinstance(user_query, str) or not user_query.strip():
        return jsonify({"answer": "Query is required."}), 400
    logging.info(f"Incoming query: {user_query}")
    debug_mode = bool(data.get("debug", False))
    effective_debug = debug_mode or DEBUG_RAG

    # 2) Mode gate
    mode = conversation_memory.get("data_mode")

    gate_query_type = classify_query(user_query)
    if mode == "tabular" and gate_query_type == "structured":
        logging.info("TABULAR MODE -> STRUCTURED QUERY")
        answer = run_structured_query(user_query, uploaded_files, conversation_memory)
        log_query_answer("TABULAR", user_query, answer)
        response = {"answer": answer}
        if effective_debug:
            response["debug"] = {
                "mode": "tabular",
                "gate_query_type": gate_query_type,
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

    queries = split_into_questions(user_query)
    logging.info(f"Split into {len(queries)} questions")
    answers= []
    debug_questions = []
    embedding_cache = {}
    
    for q in queries:
        q_start = time.perf_counter()
        q = q.strip()
        fallback_reason = None
        q_debug = {
            "question": q,
            "stages_ms": {},
        }

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
        

        if query_type == "structured":
            t_struct = time.perf_counter()
            final_answer = run_structured_query(q, uploaded_files, conversation_memory)
            log_query_answer("STRUCTURED", q, final_answer)
            q_debug["stages_ms"]["structured_pandas"] = round((time.perf_counter() - t_struct) * 1000, 2)
            q_debug["query_type"] = query_type
            q_debug["fallback_reason"] = None
            q_debug["stages_ms"]["total"] = round((time.perf_counter() - q_start) * 1000, 2)
            if effective_debug:
                debug_questions.append(q_debug)
            answers.append(final_answer)
            continue            
        
        # ------------------ Cache Check ------------------

        cached = get_cached_answer(session_id, q)
        
        if cached:
            logging.info("Cache hit found")
            metrics["cache_hits"] += 1
            log_query_answer("CACHE", q, cached)
            q_debug["cache_hit"] = True
            q_debug["query_type"] = query_type
            q_debug["stages_ms"]["total"] = round((time.perf_counter() - q_start) * 1000, 2)
            if effective_debug:
                debug_questions.append(q_debug)
            answers.append(cached)
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
            answers.append(section_answer)
            continue
        
        # ------------------ Hybrid Search Block ------------------
        
        # ---------------- HYBRID RETRIEVAL (PRODUCTION RANKING) ----------------

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
        )


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
                ans = general_response.choices[0].message.content.strip()
                q_debug["fallback_reason"] = fallback_reason
                q_debug["stages_ms"]["total"] = round((time.perf_counter() - q_start) * 1000, 2)
                if effective_debug:
                    debug_questions.append(q_debug)
                answers.append(ans)
                continue
            except Exception:
                fallback_reason = "general_fallback_failed"
                logging.info(f"Fallback reason: {fallback_reason}")
                q_debug["fallback_reason"] = fallback_reason
                q_debug["stages_ms"]["total"] = round((time.perf_counter() - q_start) * 1000, 2)
                if effective_debug:
                    debug_questions.append(q_debug)
                answers.append("Not available in the dataset.")
                continue

        logging.info(f"Final confidence score: {confidence_percent}%")

        filtered_results, context, conflict_info, early_answer = prepare_generation_context(
            q=q,
            filtered_results=filtered_results,
            context_limit=context_limit,
        )
        if early_answer is not None:
            fallback_reason = "summary_context_unavailable"
            finalize_question_debug(q_debug, fallback_reason, q_start, effective_debug, debug_questions)
            answers.append(early_answer)
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

        else:
            metrics["not_found"] += 1
            if fallback_reason is None:
                fallback_reason = "empty_answer"
            logging.info(f"Fallback reason: {fallback_reason}")
            finalize_question_debug(q_debug, fallback_reason, q_start, effective_debug, debug_questions)
            log_query_answer("RAG", q, "Not available in the dataset")
            answers.append("Not available in the dataset")
    elapsed = time.time() - start_time
    metrics["total_time"] += elapsed
    metrics["total_queries"] += len(queries)
    response = {
        "answer": "\n\n".join(answers),
        "time_taken": round(elapsed, 3)
    }
    if effective_debug:
        response["debug"] = {
            "debug_enabled": True,
            "test_mode": TEST_MODE,
            "disable_query_expansion": DISABLE_QUERY_EXPANSION,
            "questions": debug_questions
        }
    return jsonify(response)

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

