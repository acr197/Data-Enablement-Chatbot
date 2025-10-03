# app.py — Ephemeral, per-session RAG chatbot (no docs in repo)
# Requires: gradio>=4.44.1, openai>=1.40.0, python-dotenv, pymupdf, numpy, tiktoken (optional)

import os, json, uuid, shutil
from pathlib import Path
from typing import List, Tuple, Dict
from dotenv import load_dotenv

import numpy as np
import gradio as gr
from openai import OpenAI

# ---- Secrets / client ----
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY (HF → Settings → Secrets).")
client = OpenAI(api_key=OPENAI_KEY)

# ---- Behavior knobs (safe defaults; see comments to change later) ----
SYSTEM = (
    "You are a company onboarding chatbot. Answer ONLY using the Context below. "
    "If the answer is not in the context, reply exactly: 'Not in the docs.' "
    "Be concise."
)
EMBED_MODEL = "text-embedding-3-large"   # ↑ Better recall. Swap to 3-small to save $.
GEN_MODEL   = "gpt-4o"                    # ↑ Best for doc QA. Can fall back to gpt-4o-mini.
CHUNK_SIZE, CHUNK_OVERLAP = 1200, 200     # ↑ Larger chunks capture full thoughts; overlap keeps continuity.
SESSIONS_ROOT = Path("/tmp/sessions")     # ↑ Ephemeral; wiped on restart. Keeps docs private per user.

# ---------- File reading (returns text WITH page metadata) ----------
def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _read_pdf_pages(path: Path):
    import fitz  # PyMuPDF
    out = []
    with fitz.open(str(path)) as doc:
        if getattr(doc, "needs_pass", False):
            raise ValueError(f"PDF is password-protected: {path.name}")
        for i, p in enumerate(doc, start=1):
            txt = p.get_text("text") or ""
            if not txt.strip():  # try a second extractor
                try:
                    blocks = p.get_text("blocks") or []
                    txt = "\n".join(b[4] for b in blocks if isinstance(b, (list, tuple)) and len(b) > 4) or ""
                except Exception:
                    pass
            if txt.strip():  # skip fully scanned pages with no text layer
                out.append((txt, {"page": i}))
    return out

def embed_batch(texts, batch_size=64):
    vecs = []
    for i in range(0, len(texts), batch_size):
        part = texts[i:i + batch_size]
        r = client.embeddings.create(model=EMBED_MODEL, input=part)
        vecs.append(np.asarray([d.embedding for d in r.data], dtype=np.float32))
    return np.vstack(vecs)


def load_file_with_meta(path: Path) -> List[Tuple[str, Dict]]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf_pages(path)
    # treat txt/md/anything-else as whole-text (no page)
    return [(_read_text(path), {"page": None})]

# ---------- Chunking ----------
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
                chunk = chunk[: k + 1]  # end on sentence if possible
        if len(chunk) > 30:
            out.append(chunk)
        i = max(0, j - overlap)
    return out

# ---------- Per-session numpy index (cosine sim) ----------
def sess_dir(session_id: str) -> Path:
    d = SESSIONS_ROOT / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_index(session_id: str, embeddings: np.ndarray, meta: list):
    d = sess_dir(session_id)
    np.save(d / "embeddings.npy", embeddings)
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def load_index(session_id: str):
    d = sess_dir(session_id)
    e, m = d / "embeddings.npy", d / "meta.json"
    if not e.exists() or not m.exists():
        return None, None
    return np.load(e), json.loads(m.read_text(encoding="utf-8"))

def embed(text: str) -> np.ndarray:
    r = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(r.data[0].embedding, dtype=np.float32)

def retrieve(session_id: str, query: str, k=6) -> List[Dict]:
    embs, meta = load_index(session_id)
    if embs is None or not len(meta):
        return []
    q = embed(query)
    embs_n = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    q_n    = q    / np.linalg.norm(q)
    sims = (embs_n @ q_n).astype(np.float32)
    idxs = np.argsort(-sims)[:k]
    return [{"score": float(sims[i]), "meta": meta[i]} for i in idxs]

# ---------- Ingest / Clear (returns REAL error text if something fails) ----------
def ingest_files(files, session_id: str):
    try:
        if not files:
            return "Upload one or more files (PDF/TXT/MD)."
        all_chunks, all_meta = [], []
        skipped_pages = 0

        for f in files:
            p = Path(getattr(f, "name", str(f)))
            name = p.name
            blobs = load_file_with_meta(p)  # list[(text, {'page':...})]
            for text, extra in blobs:
                chunks = chunk_text(text)
                if not text.strip():
                    skipped_pages += 1
                for c in chunks:
                    all_chunks.append(c)
                    all_meta.append({
                        "doc": name,
                        "page": extra.get("page"),
                        "chunk_id": str(uuid.uuid4()),
                        "text_preview": c[:400].replace("\n", " ")
                    })

        if not all_chunks:
            msg = "No readable text found (likely scanned/unencrypted PDFs)."
            if skipped_pages:
                msg += f" Skipped pages without text: {skipped_pages}."
            return msg

        emb_matrix = embed_batch(all_chunks, batch_size=64)
        save_index(session_id, emb_matrix, all_meta)

        # best-effort cleanup of temp uploads
        for f in files:
            try:
                Path(getattr(f, "name", str(f))).unlink(missing_ok=True)
            except Exception:
                pass

        return f"Ingested {len(all_chunks)} chunks from {len(files)} file(s)." + \
               (f" Skipped {skipped_pages} blank/scanned page(s)." if skipped_pages else "")
    except Exception as e:
        return f"Ingest error: {type(e).__name__}: {e}"


def clear_session(session_id: str):
    d = sess_dir(session_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    return "Session index cleared."

# ---------- Chat (STRICTLY from context; temperature=0) ----------
def _format_ctx(snips: List[Dict]) -> Tuple[str, List[str]]:
    if not snips:
        return "", []
    lines, srcs = [], set()
    for s in snips[:4]:  # keep prompt lean
        m, sc = s["meta"], s["score"]
        page = f" p.{m['page']}" if m.get("page") else ""
        lines.append(f"[{m['doc']}{page}] {m['text_preview']}")
        srcs.add(m["doc"])
    return "\n\n".join(lines), sorted(srcs)

def chat_fn(message, history, session_id):
    try:
        snips = retrieve(session_id, message, k=6)
        if not snips:
            return "Not in the docs."
        ctx, srcs = _format_ctx(snips)

        msgs = [
            {"role": "system", "content": SYSTEM},                           # strong grounding
            {"role": "user",   "content": f"Context:\n{ctx}\n\nQuestion: {message}"}  # only-docs prompting
        ]
        r = client.chat.completions.create(
            model=GEN_MODEL,
            messages=msgs,
            max_tokens=350,
            temperature=0.0  # ↑ deterministic, avoids hallucination when context is clear
        )
        out = r.choices[0].message.content.strip()
        if srcs:
            out += "\n\nSources: " + ", ".join(srcs)
        return out
    except Exception as e:
        return f"[Error] {type(e).__name__}: {e}"

# ---------- UI (per-user session_id; kept in gr.State) ----------
with gr.Blocks(title="Data Enablement Chatbot") as demo:
    session_id = gr.State()  # None until first interaction; we create & return it from handlers

    gr.Markdown("### Data Enablement Chatbot — upload docs (private to **your session**), click **Ingest**, then ask questions.")

    def _sid(sid):  # ensure every user gets a unique, private session
        return sid or str(uuid.uuid4())

    with gr.Row():
        with gr.Column(scale=1):
            up          = gr.File(label="Upload docs (PDF / MD / TXT)", file_count="multiple")
            ingest_btn  = gr.Button("Ingest uploaded files")
            ingest_out  = gr.Textbox(label="Ingest status")
            clear_btn   = gr.Button("Clear my session index")
            clear_out   = gr.Textbox(label="Clear status")
            gr.Markdown("Docs & embeddings are stored **ephemerally** under `/tmp/sessions/<session>` and are cleared on restart or when you press **Clear**.")
        with gr.Column(scale=2):
            chat = gr.Chatbot()
            msg  = gr.Textbox(placeholder="Ask a question about your docs…")
            send = gr.Button("Send")

    # Handlers return updated session_id so each user keeps their own
    def ingest_handler(files, sid):
        sid = _sid(sid)
        return ingest_files(files, sid), sid

    def clear_handler(sid):
        sid = _sid(sid)
        return clear_session(sid), sid

    def send_handler(text, history, sid):
        sid = _sid(sid)
        if not text or not text.strip():
            return gr.update(value="Please enter a question."), history, sid
        reply = chat_fn(text, history, sid)
        history = (history or []) + [(text, reply)]
        return "", history, sid

    ingest_btn.click(ingest_handler, inputs=[up, session_id],   outputs=[ingest_out, session_id])
    clear_btn.click(clear_handler,   inputs=[session_id],        outputs=[clear_out, session_id])
    send.click(send_handler,         inputs=[msg, chat, session_id], outputs=[msg, chat, session_id])
    msg.submit(send_handler,         inputs=[msg, chat, session_id], outputs=[msg, chat, session_id])

if __name__ == "__main__":
    demo.launch()
