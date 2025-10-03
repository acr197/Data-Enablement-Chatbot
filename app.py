# app.py — Preloaded RAG chatbot (indexes local ./docs on startup)
# Requires: gradio>=4.44.1, openai>=1.40.0, python-dotenv, pymupdf, numpy

import os, json, uuid, time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

# ---------------- Config "knobs" (safe defaults) ----------------
SYSTEM = (
    "You are a company onboarding chatbot. Answer ONLY using the Context below. "
    "If the answer is not in the context, reply exactly: 'Not in the docs.' Be concise."
)
EMBED_MODEL = "text-embedding-3-large"   # Better recall; use "text-embedding-3-small" to save cost
GEN_MODEL   = "gpt-4o"                    # Strongest for doc QA; "gpt-4o-mini" is cheaper
CHUNK_SIZE, CHUNK_OVERLAP = 1200, 200     # Larger chunk + overlap gives fewer misses
DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))  # Where we read files from

# ---------------- Init OpenAI client ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY (set in env or HF Space Secrets).")
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------- File readers ----------------
def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _read_pdf_pages(path: Path) -> List[Tuple[str, Dict]]:
    """Return list of (page_text, {'page': int}) so we can cite pages."""
    import fitz  # PyMuPDF
    out: List[Tuple[str, Dict]] = []
    with fitz.open(str(path)) as doc:
        if getattr(doc, "needs_pass", False):
            raise ValueError(f"PDF is password-protected: {path.name}")
        for i, p in enumerate(doc, start=1):
            txt = p.get_text("text") or ""
            if not txt.strip():
                try:  # second extractor attempt
                    blocks = p.get_text("blocks") or []
                    txt = "\n".join(b[4] for b in blocks if isinstance(b, (list, tuple)) and len(b) > 4) or ""
                except Exception:
                    pass
            if txt.strip():  # skip scanned pages with no text layer
                out.append((txt, {"page": i}))
    return out

def load_file_with_meta(path: Path) -> List[Tuple[str, Dict]]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf_pages(path)
    return [(_read_text(path), {"page": None})]  # txt/md/etc

# ---------------- Chunking ----------------
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
            if k > len(chunk) * 0.5:
                chunk = chunk[: k + 1]  # try to end on a sentence
        if len(chunk) > 30:
            out.append(chunk)
        i = max(0, j - overlap)
    return out

# ---------------- Embeddings ----------------
def embed_batch(texts: List[str], batch_size=64) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        part = texts[i:i + batch_size]
        r = client.embeddings.create(model=EMBED_MODEL, input=part)
        vecs.append(np.asarray([d.embedding for d in r.data], dtype=np.float32))
    return np.vstack(vecs)

# ---------------- Index build (from ./docs) ----------------
EMB_MATRIX: np.ndarray = np.array([])
META: List[Dict] = []
INDEX_SUMMARY = "No index."

def build_index_from_docs() -> str:
    global EMB_MATRIX, META
    start = time.time()
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Docs folder not found: {DOCS_DIR.resolve()}")

    files = [p for p in DOCS_DIR.rglob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"}]
    if not files:
        raise FileNotFoundError(f"No PDF/TXT/MD files found under {DOCS_DIR.resolve()}")

    all_chunks, all_meta = [], []
    total_pages_skipped = 0
    for path in files:
        name = path.name
        blobs = load_file_with_meta(path)  # [(text, {'page':..})]
        for text, extra in blobs:
            if not text.strip():
                total_pages_skipped += 1
                continue
            for c in chunk_text(text):
                all_chunks.append(c)
                all_meta.append({
                    "doc": name,
                    "page": extra.get("page"),
                    "chunk_id": str(uuid.uuid4()),
                    "text_preview": c[:400].replace("\n", " ")
                })

    if not all_chunks:
        raise ValueError("No readable text found (likely scanned PDFs without OCR).")

    EMB_MATRIX = embed_batch(all_chunks, batch_size=64)
    META = all_meta
    dur = time.time() - start
    return (
        f"Indexed {len(files)} file(s), {len(META)} chunks in {dur:.1f}s."
        + (f" Skipped {total_pages_skipped} blank/scanned page(s)." if total_pages_skipped else "")
    )

# Build index at startup
try:
    INDEX_SUMMARY = build_index_from_docs()
except Exception as e:
    INDEX_SUMMARY = f"Index error: {type(e).__name__}: {e}"

# ---------------- Retrieval & Chat ----------------
def retrieve(query: str, k=6) -> List[Dict]:
    if EMB_MATRIX.size == 0 or not META:
        return []
    q = embed_batch([query])[0]
    embs_n = EMB_MATRIX / np.linalg.norm(EMB_MATRIX, axis=1, keepdims=True)
    q_n = q / np.linalg.norm(q)
    sims = (embs_n @ q_n).astype(np.float32)
    idxs = np.argsort(-sims)[:k]
    return [{"score": float(sims[i]), "meta": META[i]} for i in idxs]

def _format_ctx(snips: List[Dict]) -> Tuple[str, List[str]]:
    if not snips:
        return "", []
    lines, srcs = [], set()
    for s in snips[:4]:
        m = s["meta"]
        page = f" p.{m['page']}" if m.get("page") else ""
        lines.append(f"[{m['doc']}{page}] {m['text_preview']}")
        srcs.add(m["doc"])
    return "\n\n".join(lines), sorted(srcs)

def chat_fn(message, history):
    try:
        snips = retrieve(message, k=6)
        if not snips:
            return "Not in the docs."
        ctx, srcs = _format_ctx(snips)
        msgs = [
            {"role": "system", "content": SYSTEM},                                # Strong grounding
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {message}"} # Only-from-context
        ]
        r = client.chat.completions.create(
            model=GEN_MODEL, messages=msgs, max_tokens=350, temperature=0.0
        )
        out = r.choices[0].message.content.strip()
        if srcs:
            out += "\n\nSources: " + ", ".join(srcs)
        return out
    except Exception as e:
        return f"[Error] {type(e).__name__}: {e}"

# ---------------- UI (chat-only; index summary shown) ----------------
with gr.Blocks(title="Data Enablement Chatbot") as demo:
    gr.Markdown(f"**Index status:** {INDEX_SUMMARY}")
    chat = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask a question about the docs…")
    send = gr.Button("Send")

    def on_send(text, history):
        if not text or not text.strip():
            return gr.update(value="Please enter a question."), history
        reply = chat_fn(text, history)
        history = (history or []) + [(text, reply)]
        return "", history

    send.click(on_send, inputs=[msg, chat], outputs=[msg, chat])
    msg.submit(on_send, inputs=[msg, chat], outputs=[msg, chat])

if __name__ == "__main__":
    demo.launch()
