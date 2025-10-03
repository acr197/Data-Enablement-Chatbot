# app.py — Global RAG over a local docs/ folder (no per-user upload).
# Runtime secrets: set OPENAI_API_KEY in your Space (Settings → Secrets).
# Optional: set DOCS_DIR (e.g. "/home/user/app/docs" or "./docs").

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------
# Config "knobs" (safe defaults)
# ---------------------------
SYSTEM = (
    "You are a company onboarding chatbot. Answer ONLY using the Context below. "
    "If the answer is not in the context, reply exactly: 'Not in the docs.' Be concise."
)
EMBED_MODEL = "text-embedding-3-large"   # ↑ better recall. Swap to 3-small to save cost.
GEN_MODEL   = "gpt-4o"                    # ↑ best for doc QA. Can fall back to gpt-4o-mini.
CHUNK_SIZE, CHUNK_OVERLAP = 1200, 200     # ↑ larger chunks capture full thoughts; overlap keeps continuity.

# ---------------------------
# Secrets / client
# ---------------------------
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY (add in HF Space Settings → Secrets).")
client = OpenAI(api_key=OPENAI_KEY)

# Where to read docs from. Default: ./docs relative to this file.
DOCS_DIR = Path(os.getenv("DOCS_DIR", Path(__file__).parent / "docs")).resolve()

# ---------------------------
# File reading
# ---------------------------
# optional fallback extractor for stubborn PDFs
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _read_pdf_pages(path: Path) -> List[Tuple[str, Dict]]:
    """Return list[(text, {'page': int})] per page; fallback to whole-doc with pdfminer."""
    import fitz  # PyMuPDF
    out: List[Tuple[str, Dict]] = []
    with fitz.open(str(path)) as doc:
        if getattr(doc, "needs_pass", False):
            return out
        for i, p in enumerate(doc, start=1):
            txt = p.get_text("text") or ""
            if not txt.strip():
                try:
                    blocks = p.get_text("blocks") or []
                    txt = "\n".join(
                        b[4] for b in blocks
                        if isinstance(b, (list, tuple)) and len(b) > 4
                    ) or ""
                except Exception:
                    pass
            if txt.strip():
                out.append((txt, {"page": i}))

    # Fallback: if page-wise got nothing, try pdfminer whole-doc extraction
    if not out and pdfminer_extract_text:
        try:
            txt = (pdfminer_extract_text(str(path)) or "").strip()
            if txt:
                out = [(txt, {"page": None})]
        except Exception:
            pass

    return out


def load_file_with_meta(path: Path) -> List[Tuple[str, Dict]]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf_pages(path)                         # [(text, {'page': n}) …]
    # treat txt/md/anything-else as whole-text (no page num)
    return [(_read_text(path), {"page": None})]

# ---------------------------
# Chunking
# ---------------------------
def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    text = text.replace("\r\n", "\n")
    out, i, n = [], 0, len(text)
    while i < n:
        j = min(i + size, n)
        chunk = text[i:j].strip()
        if j < n:
            k = chunk.rfind(". ")
            if k > len(chunk) * 0.5:      # try to end on a sentence boundary
                chunk = chunk[: k + 1]
        if len(chunk) > 30:               # drop tiny fragments
            out.append(chunk)
        i = max(0, j - overlap)
    return out

# ---------------------------
# Embeddings (batched) + cosine sim
# ---------------------------
def embed_batch(texts: List[str], batch_size=64) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        r = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs.append(np.asarray([d.embedding for d in r.data], dtype=np.float32))
    return np.vstack(vecs)

def cosine_topk(matrix: np.ndarray, query: np.ndarray, k: int) -> List[int]:
    m = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    q = query / np.linalg.norm(query)
    sims = (m @ q).astype(np.float32)
    idx = np.argsort(-sims)[:k]
    return idx.tolist()

# ---------------------------
# Global index (embeddings + metadata) built at startup
# ---------------------------
G_EMB: np.ndarray = None         # shape (N, D)
G_META: List[Dict] = []          # [{doc, page, text_preview}, …]
G_STATUS: str = ""               # human-readable status text

def build_index_from_local() -> Tuple[str, np.ndarray, List[Dict]]:
    """Scan DOCS_DIR for .pdf/.txt/.md, chunk, embed, return (status, emb, meta)."""
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Docs folder not found: {DOCS_DIR}")

    files = []
    for ext in ("*.pdf", "*.txt", "*.md"):
        files.extend(sorted(DOCS_DIR.glob(ext)))

    if not files:
        return f"No documents found in {DOCS_DIR}", None, []

    all_chunks, all_meta = [], []
    skipped_pages = 0
    for path in files:
        try:
            blobs = load_file_with_meta(path)  # list[(text, {'page':...})]
        except Exception as e:
            # skip unreadable file and continue
            continue
        for text, extra in blobs:
            if not text.strip():
                skipped_pages += 1
                continue
            for c in chunk_text(text):
                all_chunks.append(c)
                all_meta.append({
                    "doc": path.name,
                    "page": extra.get("page"),
                    "text_preview": c[:400].replace("\n", " ")
                })

    if not all_chunks:
        msg = f"Found {len(files)} file(s) but no readable text (scanned PDFs?)."
        if skipped_pages:
            msg += f" Skipped {skipped_pages} blank/scanned page(s)."
        return msg, None, []

    emb = embed_batch(all_chunks, batch_size=64)
    status = f"Indexed {len(all_chunks):,} chunks from {len(files):,} file(s) in {DOCS_DIR.name}."
    if skipped_pages:
        status += f" Skipped {skipped_pages} blank/scanned page(s)."
    return status, emb, all_meta

def retrieve(query: str, top_k=6) -> List[Dict]:
    global G_EMB, G_META
    if G_EMB is None or not G_META:
        return []
    q = client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
    q = np.asarray(q, dtype=np.float32)
    idxs = cosine_topk(G_EMB, q, top_k)
    out = []
    # compute scores for display (optional)
    m = G_EMB / np.linalg.norm(G_EMB, axis=1, keepdims=True)
    qn = q / np.linalg.norm(q)
    sims = (m @ qn).astype(np.float32)
    for i in idxs:
        out.append({"score": float(sims[i]), "meta": G_META[i]})
    return out

def format_ctx(snips: List[Dict]) -> Tuple[str, List[str]]:
    if not snips:
        return "", []
    lines, srcs = [], set()
    for s in snips[:4]:  # keep prompt lean
        m, sc = s["meta"], s["score"]
        page = f" p.{m['page']}" if m.get("page") else ""
        lines.append(f"[{m['doc']}{page}] {m['text_preview']}")
        srcs.add(m["doc"])
    return "\n\n".join(lines), sorted(srcs)

def answer_from_context(question: str) -> str:
    snips = retrieve(question, top_k=6)
    if not snips:
        return "Not in the docs."
    ctx, srcs = format_ctx(snips)
    msgs = [
        {"role": "system", "content": SYSTEM},  # strong grounding
        {"role": "user",   "content": f"Context:\n{ctx}\n\nQuestion: {question}"}
    ]
    r = client.chat.completions.create(
        model=GEN_MODEL,
        messages=msgs,
        max_tokens=350,
        temperature=0.0,  # deterministic; reduces hallucinations
    )
    out = r.choices[0].message.content.strip()
    if srcs:
        out += "\n\nSources: " + ", ".join(srcs)
    return out

# ---------------------------
# UI
# ---------------------------
with gr.Blocks(title="Data Enablement Chatbot") as demo:
    index_status = gr.Markdown("Index status: building …")
    chat = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask a question about your docs…")
    send = gr.Button("Send")
    reindex = gr.Button("Reindex now")

    def _send(user_text, history):
        if not user_text or not user_text.strip():
            return gr.update(value="Please enter a question."), history
        reply = answer_from_context(user_text)
        history = (history or []) + [(user_text, reply)]
        return "", history

    def _do_reindex():
        global G_EMB, G_META, G_STATUS
        try:
            status, emb, meta = build_index_from_local()
            G_EMB, G_META, G_STATUS = emb, meta, status
        except Exception as e:
            G_EMB, G_META, G_STATUS = None, [], f"Index error: {type(e).__name__}: {e}"
        return f"Index status: {G_STATUS}"

    # wire up
    send.click(_send, inputs=[msg, chat], outputs=[msg, chat])
    msg.submit(_send, inputs=[msg, chat], outputs=[msg, chat])
    reindex.click(_do_reindex, outputs=[index_status])

    # build index once at app start and show status
    def _on_start():
        return _do_reindex()

    demo.load(_on_start, outputs=[index_status])

if __name__ == "__main__":
    demo.launch()
