import os
import re
import hashlib
import pandas as pd
import pymupdf4llm
from docx import Document
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document as LangDocument
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter, 
    MarkdownHeaderTextSplitter,)

SECTION_ALIASES = {
    "abstract": "ABSTRACT",
    "introduction": "INTRODUCTION",
    "background": "INTRODUCTION",
    "method": "METHODS",
    "methods": "METHODS",
    "methodology": "METHODS",
    "materials and methods": "METHODS",
    "results": "RESULTS",
    "discussion": "DISCUSSION",
    "conclusion": "CONCLUSION",
    "conclusions": "CONCLUSION",
    "references": "REFERENCES",
}

SECTION_REGEX = re.compile(
    r'^\s*(abstract|introduction|background|methodology|methods?|materials and methods|results?|discussion|conclusions?|references)\s*$',
    re.IGNORECASE
)

def inject_section_markers(text: str) -> str:
    """
    convert raw pdf text into rag sections
    required for your  /query section retrieval to work
    """
    lines = text.split("\n")
    new_lines = []
    
    for line in lines:
        clean = line.strip().lower()

        if SECTION_REGEX.match(clean):
            section_name = SECTION_ALIASES.get(clean, clean).upper()
            new_lines.append(f"[SECTION:{section_name}]")

        new_lines.append(line)

    return "\n".join(new_lines)

def normalize_text(text:str) -> str:
    """normalize whitespace for stable embeddings"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Keep paragraph boundaries for section heading detection.
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.replace("\x00", "")
    return text.strip()

def made_chunk_id(file_path: str, content: str, page=None) -> str:
    """Stable deterministic ID for updates"""
    base = f"{file_path}|{page}|{content[:200]}"
    return hashlib.sha256(base.encode()).hexdigest()[:32]

def attach_metadata(docs: list[LangDocument], file_path: str, ext: str) -> list[LangDocument]:
    base =os.path.basename(file_path)
    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata["file"] = base
        d.metadata["source"] = file_path
        d.metadata["file_type"] = ext
    return docs
def load_pdf(file_path: str) -> list[LangDocument]:
    """Layout aware pdf loader"""
    md = pymupdf4llm.to_markdown(file_path, page_chunks=True)
    docs = []
    for page in md:
        if not isinstance(page, dict):
            continue
       
        text = normalize_text(page.get("text", ""))
        text = inject_section_markers(text)
        if not text:
            continue
        page_no = page.get("page", page.get("page_number", page.get("number", -1)))
        docs.append(
            LangDocument(
                page_content=text,
                metadata={"page": page_no}
            )
        )
    return docs

def load_docx(file_path: str) -> list[LangDocument]:
    doc = Document(file_path)
    text = "\n".join(p.text for p in doc.paragraphs)
    return [LangDocument(page_content=normalize_text(text))]


def load_csv_excel(file_path: str, ext: str) -> list[LangDocument]:
    """Group rows into logical blocks instead of row spam"""
    df = pd.read_csv(file_path) if ext == "csv" else pd.read_excel(file_path)

    blocks = []
    chunk_rows = 20  # production tuning

    for i in range(0, len(df), chunk_rows):
        part = df.iloc[i:i+chunk_rows].astype(str)
        table_text = "\n".join(" | ".join(row) for row in part.values)
        blocks.append(
            LangDocument(
                page_content=normalize_text(table_text),
                metadata={"rows": f"{i}-{i+len(part)-1}"}
            )
        )

    return blocks

# MAIN ENTRY

def load_and_chunk_file(file_path: str) -> list[LangDocument]:


    ext = os.path.splitext(file_path)[1].lower().replace(".", "")

    # -------- Load --------
    if ext == "pdf":
        docs = load_pdf(file_path)
    elif ext == "txt":
        docs = TextLoader(file_path, encoding="utf-8").load()
    elif ext == "docx":
        docs = load_docx(file_path)
    elif ext in ["csv", "xls", "xlsx"]:
        docs = load_csv_excel(file_path, ext)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    docs = attach_metadata(docs, file_path, ext)

    # CHUNKING STRATEGY
    
    if ext == "pdf":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 900,
            chunk_overlap = 150,
            separators = [
                "\n[SECTION:",
                "\n\n",
                "\n",
                ". ",
                " "
            ]
        )
        chunks = splitter.split_documents(docs)
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

    
    # FINALIZE (IDs + CLEAN)
    
    final_chunks = []
    seen = set()
    for c in chunks:
        content = normalize_text(c.page_content)
        if not content:
            continue
        page = c.metadata.get("page")
        cid = made_chunk_id(file_path, content, page)
        if cid in seen:
            continue
        seen.add(cid)
        c.page_content = content
        c.metadata["chunk_id"] = cid
        final_chunks.append(c)
    print(f"[INGEST] {os.path.basename(file_path)} -> {len(final_chunks)} chunks")
    return final_chunks


def load_file(file_path: str) -> list[LangDocument]:
    """Backward-compatible entrypoint used by app.py."""
    return load_and_chunk_file(file_path)




    
