# rig_app.py
# ---------------------------------------------------------------------------
# Section: imports
# ---------------------------------------------------------------------------
from __future__ import annotations

import os
import re
import sys
import json
import time
import shutil
import hashlib
import logging
import traceback
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import faiss  # faiss-cpu
import streamlit as st
import fitz  # PyMuPDF
from docx import Document as DocxDocument  # python-docx

try:
    import textract  # optional for .doc
    TEXTRACT_OK = True
except Exception:
    TEXTRACT_OK = False

from openai import OpenAI

print("Imports ready.")

# ---------------------------------------------------------------------------
# Section: user config
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

# Paths and defaults
DEFAULT_PROJECT_ROOT: str = r"C:\Users\Andrew\PycharmProjects\Data Enablement Chatbot"
DEFAULT_DOCS_DIR: str = os.getenv(
    "DOCS_DIR",
    os.path.join(DEFAULT_PROJECT_ROOT, "docs")
)
DEFAULT_INDEX_SUBDIR: str = "_rig_index"

# Models and parameters
CHAT_MODEL: str = "gpt-4o-mini"
EMBED_MODEL: str = "text-embedding-3-large"  # 3072-dim
EMBED_DIM: int = 3072
TOP_K_DEFAULT: int = 6
EMBED_BATCH_SIZE: int = 64
GEN_TEMPERATURE: float = 0.2
GEN_MAX_TOKENS: int = 800

# Chunking
CHUNK_TARGET_CHARS: int = 1800
CHUNK_OVERLAP_CHARS: int = 250

# Filenames
FAISS_INDEX_NAME: str = "faiss.index"
DOCSTORE_JSONL_NAME: str = "docstore.jsonl"
MANIFEST_JSON_NAME: str = "manifest.json"
IDS_JSON_NAME: str = "ids.json"

# API key pulled from .env
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")

print("User config ready.")


# ---------------------------------------------------------------------------
# Section: logging setup
# ---------------------------------------------------------------------------
LOG_PATH: str = os.path.join(DEFAULT_PROJECT_ROOT, "rig_app.log")
os.makedirs(DEFAULT_PROJECT_ROOT, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("rig_app")
print("Logging setup ready.")

# ---------------------------------------------------------------------------
# Section: data models
# ---------------------------------------------------------------------------
@dataclass
class ChunkMeta:
    file_path: str
    file_name: str
    file_ext: str
    file_type: str
    page: Optional[int]  # 1-based for PDFs, else None
    chunk_idx: int       # 1-based index within file
    char_start: int
    char_end: int
    sha256: str

@dataclass
class ChunkRecord:
    chunk_id: int        # int64 for FAISS
    text: str
    meta: ChunkMeta

@dataclass
class FileManifestEntry:
    sha256: str
    mtime: float
    chunk_ids: List[int]

print("Data models ready.")

# ---------------------------------------------------------------------------
# Section: utilities
# ---------------------------------------------------------------------------
CONTROL_CHAR_RE = re.compile(r"[\u0000-\u0008\u000B-\u001F\u007F\u200B\u200C\u200D\uFEFF]")
TIMESTAMP_RE = re.compile(r"(\[?\b\d{1,2}:\d{2}:\d{2}\b\]?)")
SPEAKER_ARROW_RE = re.compile(r"(?m)^(>{2,}\s*)")
WHITESPACE_RUN_RE = re.compile(r"[ \t]{2,}")
PUNCT_RUN_RE = re.compile(r"([!?.,]){3,}")

def safe_mkdir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        logger.error("Failed to create directory: %s | %s", path, e)

def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def clean_text(text: str) -> str:
    text = CONTROL_CHAR_RE.sub("", text)
    text = TIMESTAMP_RE.sub("", text)
    text = SPEAKER_ARROW_RE.sub("", text)
    text = PUNCT_RUN_RE.sub(lambda m: m.group(1) * 2, text)
    lines = [WHITESPACE_RUN_RE.sub(" ", ln) for ln in text.splitlines()]
    return "\n".join(lines).strip()

def paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]

def chunk_text(text: str, target: int, overlap: int) -> List[Tuple[int, int, str]]:
    """Return list of (start_char, end_char, chunk_text)."""
    paras = paragraphs(text)
    chunks: List[Tuple[int, int, str]] = []
    cur: List[str] = []
    start = 0
    for p in paras:
        if sum(len(x) + 2 for x in cur) + len(p) > target and cur:
            chunk_str = "\n\n".join(cur)
            end = start + len(chunk_str)
            chunks.append((start, end, chunk_str))
            if overlap > 0:
                keep = chunk_str[-overlap:]
                cur = [keep]
                start = end - len(keep)
            else:
                cur = []
                start = end
        cur.append(p)
    if cur:
        chunk_str = "\n\n".join(cur)
        end = start + len(chunk_str)
        chunks.append((start, end, chunk_str))
    return chunks

def stable_int64_id(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)

def ensure_index_dir(docs_dir: str) -> str:
    idx_dir = os.path.join(docs_dir, DEFAULT_INDEX_SUBDIR)
    safe_mkdir(idx_dir)
    return idx_dir

def read_json(path: str, default: dict) -> dict:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error("Failed to read JSON %s: %s", path, e)
    return default

def write_json(path: str, data: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("Failed to write JSON %s: %s", path, e)

def read_docstore_jsonl(path: str) -> Dict[int, ChunkRecord]:
    records: Dict[int, ChunkRecord] = {}
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    if not ln.strip():
                        continue
                    obj = json.loads(ln)
                    meta = ChunkMeta(**obj["meta"])
                    rec = ChunkRecord(chunk_id=int(obj["chunk_id"]), text=obj["text"], meta=meta)
                    records[rec.chunk_id] = rec
    except Exception as e:
        logger.error("Failed to read docstore %s: %s", path, e)
    return records

def write_docstore_jsonl(path: str, records: Dict[int, ChunkRecord]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            for rec in records.values():
                obj = {"chunk_id": rec.chunk_id, "text": rec.text, "meta": asdict(rec.meta)}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error("Failed to write docstore %s: %s", path, e)

print("Utilities ready.")

# ---------------------------------------------------------------------------
# Section: indexing
# ---------------------------------------------------------------------------
def load_or_create_faiss(index_path: str, dim: int) -> faiss.IndexIDMap2:
    if os.path.exists(index_path):
        try:
            index = faiss.read_index(index_path)
            if not isinstance(index, faiss.IndexIDMap2):
                index = faiss.IndexIDMap2(index)
            return index  # type: ignore
        except Exception as e:
            logger.error("Failed to read FAISS index, recreating: %s", e)
    base = faiss.IndexFlatIP(dim)
    return faiss.IndexIDMap2(base)

def save_faiss(index: faiss.IndexIDMap2, index_path: str) -> None:
    try:
        faiss.write_index(index, index_path)
    except Exception as e:
        logger.error("Failed to save FAISS index: %s", e)

def remove_chunk_ids(index: faiss.IndexIDMap2, ids: List[int]) -> None:
    if not ids:
        return
    try:
        sel = faiss.IDSelectorBatch(np.array(ids, dtype=np.int64))
        index.remove_ids(sel)
    except Exception as e:
        logger.error("Failed to remove ids: %s", e)

def list_supported_files(docs_dir: str) -> List[str]:
    paths: List[str] = []
    for root, _, files in os.walk(docs_dir):
        if os.path.basename(root) == DEFAULT_INDEX_SUBDIR:
            continue
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in (".pdf", ".docx", ".txt", ".doc"):
                paths.append(os.path.join(root, fn))
    return paths

def incremental_plan(docs_dir: str, manifest_path: str) -> Tuple[Dict[str, FileManifestEntry], List[str], List[str]]:
    manifest_raw = read_json(manifest_path, {})
    manifest: Dict[str, FileManifestEntry] = {}
    for k, v in manifest_raw.items():
        try:
            manifest[k] = FileManifestEntry(sha256=v["sha256"], mtime=float(v["mtime"]), chunk_ids=list(v.get("chunk_ids", [])))
        except Exception:
            continue
    files = list_supported_files(docs_dir)
    new_or_changed: List[str] = []
    unchanged: List[str] = []
    for path in files:
        try:
            mtime = os.path.getmtime(path)
            sha = compute_sha256(path)
            if path not in manifest or manifest[path].sha256 != sha or abs(manifest[path].mtime - mtime) > 1e-6:
                new_or_changed.append(path)
            else:
                unchanged.append(path)
        except Exception as e:
            logger.error("Stat/hash failed for %s: %s", path, e)
    return manifest, new_or_changed, unchanged

print("Indexing scaffolding ready.")

# ---------------------------------------------------------------------------
# Section: extraction
# ---------------------------------------------------------------------------
def extract_pdf(path: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    try:
        with fitz.open(path) as doc:
            for i, page in enumerate(doc, start=1):
                txt = page.get_text("text") or ""
                out.append((i, clean_text(txt)))
    except Exception as e:
        logger.error("PDF extract failed %s: %s", path, e)
    return out

def extract_docx(path: str) -> str:
    try:
        doc = DocxDocument(path)
        parts: List[str] = []
        for p in doc.paragraphs:
            parts.append(p.text)
        for tbl in doc.tables:
            for row in tbl.rows:
                for cell in row.cells:
                    parts.append(cell.text)
        return clean_text("\n".join(parts))
    except Exception as e:
        logger.error("DOCX extract failed %s: %s", path, e)
        return ""

def extract_doc(path: str) -> str:
    if not TEXTRACT_OK:
        logger.warning(".doc support requires textract; skipping %s", path)
        return ""
    try:
        raw = textract.process(path)
        return clean_text(raw.decode("utf-8", errors="ignore"))
    except Exception as e:
        logger.error("DOC extract failed %s: %s", path, e)
        return ""

def extract_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return clean_text(f.read())
    except Exception as e:
        logger.error("TXT read failed %s: %s", path, e)
        return ""

def build_chunks_for_file(path: str) -> List[ChunkRecord]:
    ext = os.path.splitext(path)[1].lower()
    fname = os.path.basename(path)
    sha = compute_sha256(path)
    records: List[ChunkRecord] = []

    if ext == ".pdf":
        pages = extract_pdf(path)
        for page_num, page_text in pages:
            if not page_text.strip():
                continue
            spans = chunk_text(page_text, CHUNK_TARGET_CHARS, CHUNK_OVERLAP_CHARS)
            for idx, (start, end, chunk) in enumerate(spans, start=1):
                cid = stable_int64_id(f"{path}|{sha}|pdf|{page_num}|{idx}")
                meta = ChunkMeta(
                    file_path=path,
                    file_name=fname,
                    file_ext=ext,
                    file_type="pdf",
                    page=page_num,
                    chunk_idx=idx,
                    char_start=start,
                    char_end=end,
                    sha256=sha,
                )
                records.append(ChunkRecord(chunk_id=cid, text=chunk, meta=meta))
    elif ext == ".docx":
        text = extract_docx(path)
        spans = chunk_text(text, CHUNK_TARGET_CHARS, CHUNK_OVERLAP_CHARS)
        for idx, (start, end, chunk) in enumerate(spans, start=1):
            cid = stable_int64_id(f"{path}|{sha}|docx|{idx}")
            meta = ChunkMeta(
                file_path=path,
                file_name=fname,
                file_ext=ext,
                file_type="docx",
                page=None,
                chunk_idx=idx,
                char_start=start,
                char_end=end,
                sha256=sha,
            )
            records.append(ChunkRecord(chunk_id=cid, text=chunk, meta=meta))
    elif ext == ".doc":
        text = extract_doc(path)
        if text.strip():
            spans = chunk_text(text, CHUNK_TARGET_CHARS, CHUNK_OVERLAP_CHARS)
            for idx, (start, end, chunk) in enumerate(spans, start=1):
                cid = stable_int64_id(f"{path}|{sha}|doc|{idx}")
                meta = ChunkMeta(
                    file_path=path,
                    file_name=fname,
                    file_ext=ext,
                    file_type="doc",
                    page=None,
                    chunk_idx=idx,
                    char_start=start,
                    char_end=end,
                    sha256=sha,
                )
                records.append(ChunkRecord(chunk_id=cid, text=chunk, meta=meta))
    elif ext == ".txt":
        text = extract_txt(path)
        spans = chunk_text(text, CHUNK_TARGET_CHARS, CHUNK_OVERLAP_CHARS)
        for idx, (start, end, chunk) in enumerate(spans, start=1):
            cid = stable_int64_id(f"{path}|{sha}|txt|{idx}")
            meta = ChunkMeta(
                file_path=path,
                file_name=fname,
                file_ext=ext,
                file_type="txt",
                page=None,
                chunk_idx=idx,
                char_start=start,
                char_end=end,
                sha256=sha,
            )
            records.append(ChunkRecord(chunk_id=cid, text=chunk, meta=meta))
    else:
        logger.info("Unsupported file skipped: %s", path)

    return records

print("Extraction ready.")

# ---------------------------------------------------------------------------
# Section: OpenAI helpers
# ---------------------------------------------------------------------------
def get_client() -> OpenAI:
    try:
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.error("OpenAI client init failed: %s", e)
        raise

def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    out: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            for item in resp.data:
                out.append(item.embedding)
        except Exception as e:
            logger.error("Embedding batch failed: %s", e)
            raise
    arr = np.array(out, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    arr = arr / norms
    return arr

def chat_answer(client: OpenAI, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
    try:
        resp = client.responses.create(
            model=CHAT_MODEL,
            temperature=temperature,
            max_output_tokens=max_tokens,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.output_text  # type: ignore
    except Exception as e:
        logger.error("Chat failed: %s", e)
        raise

print("OpenAI helpers ready.")

# ---------------------------------------------------------------------------
# Section: retrieval+answer
# ---------------------------------------------------------------------------
def ensure_all_ready(docs_dir: str) -> Tuple[faiss.IndexIDMap2, Dict[int, ChunkRecord], Dict[str, FileManifestEntry], str, str, str, str]:
    idx_dir = ensure_index_dir(docs_dir)
    faiss_path = os.path.join(idx_dir, FAISS_INDEX_NAME)
    docstore_path = os.path.join(idx_dir, DOCSTORE_JSONL_NAME)
    manifest_path = os.path.join(idx_dir, MANIFEST_JSON_NAME)
    ids_path = os.path.join(idx_dir, IDS_JSON_NAME)
    index = load_or_create_faiss(faiss_path, EMBED_DIM)
    docstore = read_docstore_jsonl(docstore_path)
    manifest_raw = read_json(manifest_path, {})
    manifest: Dict[str, FileManifestEntry] = {}
    for k, v in manifest_raw.items():
        try:
            manifest[k] = FileManifestEntry(sha256=v["sha256"], mtime=float(v["mtime"]), chunk_ids=list(v.get("chunk_ids", [])))
        except Exception:
            continue
    return index, docstore, manifest, faiss_path, docstore_path, manifest_path, ids_path

def reindex(docs_dir: str, progress_cb=None) -> Tuple[int, int]:
    index, docstore, manifest, faiss_path, docstore_path, manifest_path, _ = ensure_all_ready(docs_dir)
    client = get_client()
    start_time = time.time()

    manifest, changed, unchanged = incremental_plan(docs_dir, manifest_path)
    total_new_chunks = 0
    total_removed_chunks = 0

    for path in changed:
        if path in manifest and manifest[path].chunk_ids:
            remove_chunk_ids(index, manifest[path].chunk_ids)
            total_removed_chunks += len(manifest[path].chunk_ids)
            for cid in manifest[path].chunk_ids:
                if cid in docstore:
                    del docstore[cid]

    for i, path in enumerate(changed, start=1):
        if progress_cb:
            progress_cb(f"Processing {i}/{len(changed)}: {os.path.basename(path)}")
        try:
            records = build_chunks_for_file(path)
            if not records:
                manifest[path] = FileManifestEntry(sha256=compute_sha256(path), mtime=os.path.getmtime(path), chunk_ids=[])
                continue
            embeddings = embed_texts(client, [r.text for r in records])
            ids = np.array([r.chunk_id for r in records], dtype=np.int64)
            index.add_with_ids(embeddings, ids)
            for rec in records:
                docstore[rec.chunk_id] = rec
            manifest[path] = FileManifestEntry(
                sha256=compute_sha256(path),
                mtime=os.path.getmtime(path),
                chunk_ids=[r.chunk_id for r in records],
            )
            total_new_chunks += len(records)
        except Exception as e:
            logger.error("File index failed %s: %s", path, e)
            logger.debug(traceback.format_exc())

    save_faiss(index, faiss_path)
    write_docstore_jsonl(docstore_path, docstore)
    write_json(manifest_path, {k: asdict(v) for k, v in manifest.items()})

    logger.info("Reindex done in %.2fs | new=%d removed=%d unchanged_files=%d", time.time() - start_time, total_new_chunks, total_removed_chunks, len(unchanged))
    return total_new_chunks, total_removed_chunks

def clear_index(docs_dir: str) -> None:
    idx_dir = ensure_index_dir(docs_dir)
    try:
        shutil.rmtree(idx_dir, ignore_errors=True)
    except Exception as e:
        logger.error("Clear index failed: %s", e)
    safe_mkdir(idx_dir)

def format_citation(meta: ChunkMeta) -> str:
    if meta.file_type == "pdf" and meta.page:
        return f"[{meta.file_name} p. {meta.page}]"
    return f"[{meta.file_name} chunk {meta.chunk_idx}]"

def build_context_block(recs: List[ChunkRecord]) -> Tuple[str, List[str]]:
    lines: List[str] = []
    cits: List[str] = []
    for rec in recs:
        cit = format_citation(rec.meta)
        cits.append(cit)
        preview = rec.text.strip()
        lines.append(f"{cit}\n{preview}")
    return "\n\n---\n\n".join(lines), cits

def search(index: faiss.IndexIDMap2, q_emb: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    if index.ntotal == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    D, I = index.search(q_emb, top_k)
    return I[0], D[0]

def generate_answer(client: OpenAI, query: str, candidates: List[ChunkRecord], temperature: float, max_tokens: int) -> str:
    sys_prompt = "Use only the provided snippets. If insufficient, say so plainly."
    context_block, _ = build_context_block(candidates)
    user_prompt = (
        "Question:\n"
        f"{query}\n\n"
        "Context snippets (use only these; cite each claim inline using the required format):\n"
        f"{context_block}\n\n"
        "Cite PDFs like [filename.pdf p. N]. Cite DOCX/TXT/DOC like [filename.ext chunk K]."
    )
    ans = chat_answer(client, sys_prompt, user_prompt, temperature, max_tokens)
    if not re.search(r"\[[^\[\]]+\]", ans):
        _, cits = build_context_block(candidates)
        srcs = "\n".join(f"- {c}" for c in dict.fromkeys(cits))
        ans = ans.rstrip() + "\n\nSources:\n" + srcs
    return ans

print("Retrieval+answer ready.")

# ---------------------------------------------------------------------------
# Section: Streamlit UI
# ---------------------------------------------------------------------------
def ui_write_requirements(project_root: str) -> str:
    path = os.path.join(project_root, "requirements_rig.txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(REQUIREMENTS_SNIPPET.strip() + "\n")
        return path
    except Exception as e:
        logger.error("Write requirements failed: %s", e)
        return ""

def handle_uploads(target_dir: str, files: List[Any]) -> List[str]:
    saved: List[str] = []
    for uf in files:
        try:
            ext = os.path.splitext(uf.name)[1].lower()
            if ext not in (".pdf", ".docx", ".txt", ".doc"):
                continue
            dest = os.path.join(target_dir, uf.name)
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
            saved.append(dest)
        except Exception as e:
            logger.error("Upload save failed %s: %s", getattr(uf, 'name', 'unknown'), e)
    return saved

def main_app() -> None:
    st.set_page_config(page_title="Local Docs RAG", layout="wide")
    st.title("Local, Docs-Only RAG Chatbot")

    st.sidebar.header("Controls")
    docs_dir = st.sidebar.text_input("Documents folder", value=DEFAULT_DOCS_DIR)
    if not docs_dir:
        st.sidebar.error("Provide a documents folder.")
        st.stop()
    safe_mkdir(docs_dir)

    with st.sidebar.expander("Upload files"):
        files = st.file_uploader("Drop PDF/DOC/DOCX/TXT here", type=["pdf", "docx", "txt", "doc"], accept_multiple_files=True)
        if files:
            saved = handle_uploads(docs_dir, files)
            if saved:
                st.success(f"Saved {len(saved)} file(s). Click Reindex.")

    top_k = st.sidebar.slider("Top-k", min_value=1, max_value=12, value=TOP_K_DEFAULT, step=1)
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=GEN_TEMPERATURE, step=0.05)
    max_tokens = st.sidebar.slider("Max tokens", min_value=200, max_value=2000, value=GEN_MAX_TOKENS, step=50)
    strict_mode = st.sidebar.checkbox("Strict: append Sources if no citations", value=True)
    show_debug = st.sidebar.checkbox("Show debug info", value=False)

    c1, c2, c3 = st.sidebar.columns(3)
    if c1.button("Reindex"):
        msg = st.sidebar.empty()
        def cb(text: str) -> None:
            msg.info(text)
        newc, rmvc = reindex(docs_dir, progress_cb=cb)
        st.sidebar.success(f"Reindex complete. New: {newc}, Removed: {rmvc}")
    if c2.button("Clear index"):
        clear_index(docs_dir)
        st.sidebar.warning("Index cleared.")
    if c3.button("Write requirements"):
        p = ui_write_requirements(DEFAULT_PROJECT_ROOT)
        if p:
            st.sidebar.success(f"Wrote {p}")

    q = st.text_area("Ask a question about your documents:", height=120, placeholder="Type your question...")
    if st.button("Ask"):
        if not q.strip():
            st.error("Enter a question.")
            st.stop()
        index, docstore, _, _, _, _, _ = ensure_all_ready(docs_dir)
        if index.ntotal == 0 or not docstore:
            st.warning("No index found. Click Reindex after adding files.")
            st.stop()

        try:
            client = get_client()
            q_emb = embed_texts(client, [q])
            I, D = search(index, q_emb, top_k)
            if I.size == 0:
                st.write("I couldn’t find that in the provided documents.")
                st.stop()
            records: List[ChunkRecord] = []
            for cid in I.tolist():
                rec = docstore.get(int(cid))
                if rec:
                    records.append(rec)
            if not records:
                st.write("I couldn’t find that in the provided documents.")
                st.stop()

            ans = generate_answer(client, q, records, temperature, max_tokens)
            if strict_mode and not re.search(r"\[[^\[\]]+\]", ans):
                _, cits = build_context_block(records)
                srcs = "\n".join(f"- {c}" for c in dict.fromkeys(cits))
                ans = ans.rstrip() + "\n\nSources:\n" + srcs

            st.markdown("**Answer**")
            st.write(ans)

            with st.expander("Top sources"):
                for rec in records:
                    st.markdown(f"- {format_citation(rec.meta)}")
                    st.code(rec.text[:1200] + ("..." if len(rec.text) > 1200 else ""))

            if show_debug:
                st.subheader("Debug")
                st.json({
                    "docs_dir": docs_dir,
                    "index_size": int(index.ntotal),
                    "retrieved_ids": [int(x) for x in I.tolist()],
                    "scores": [float(x) for x in D.tolist()],
                })
        except Exception as e:
            st.error(f"Error: {e}")
            logger.error("Ask failed: %s", e)
            logger.debug(traceback.format_exc())

print("Streamlit UI ready.")

# ---------------------------------------------------------------------------
# Section: requirements snippet
# ---------------------------------------------------------------------------
REQUIREMENTS_SNIPPET = """
faiss-cpu==1.8.0.post1
numpy==1.26.4
PyMuPDF==1.24.9
streamlit==1.36.0
openai==1.51.0
python-docx==1.1.2
tiktoken==0.7.0
textract==1.6.5
pyinstaller==6.10.0
"""

print("Requirements snippet ready.")

# ---------------------------------------------------------------------------
# Section: final notes
# ---------------------------------------------------------------------------
def _in_streamlit_runtime() -> bool:
    """Return True only when executed by Streamlit, False when run via plain Python."""
    try:
        import streamlit.runtime as _rt  # type: ignore
        return bool(getattr(_rt, "exists", lambda: False)())
    except Exception:
        # Fallback env flags used by various Streamlit versions
        return bool(
            os.environ.get("STREAMLIT_SERVER_RUNNING")
            or os.environ.get("STREAMLIT_RUNTIME")
        )

if _in_streamlit_runtime():
    # Streamlit is running this script; render the app.
    main_app()
elif __name__ == "__main__":
    # Prevent infinite relaunch. Tell the user how to start the app.
    print("Start the app with:  streamlit run rig_app.py")

print("Final notes ready.")


rig_app.py
---------------------------------------------------------------------------
Section: imports
---------------------------------------------------------------------------