# app.py — Ephemeral, per-session RAG chatbot (no docs in repo)
# Requirements: gradio, openai>=1.40.0, python-dotenv, pymupdf, numpy

import os, json, uuid, shutil
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import gradio as gr
from openai import OpenAI

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY (HF → Settings → Secrets).")
client = OpenAI(api_key=OPENAI_KEY)

SYSTEM = ("You are a helpful company onboarding chatbot. Prefer answers grounded "
          "in the uploaded docs; if not found, say: “I don’t see this in our training docs.”")
EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL   = os.getenv("GEN_MODEL", "gpt-3.5-turbo")
CHUNK_SIZE, CHUNK_OVERLAP = 900, 150
SESSIONS_ROOT = Path("/tmp/sessions")  # ephemeral storage

# ---------- file reading ----------
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_pdf(path: Path) -> str:
    import fitz  # PyMuPDF
    with fitz.open(str(path)) as doc:
        return "\n\n".join(p.get_text("text") for p in doc)

def load_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}: return read_text(path)
    if ext == ".pdf":           return read_pdf(path)
    return read_text(path)

# ---------- chunking ----------
def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text: return []
    text = text.replace("\r\n", "\n")
    out, i, n = [], 0, len(text)
    while i < n:
        j = min(i + size, n)
        chunk = text[i:j].strip()
        if j < n:
            k = chunk.rfind(". ")
            if k > len(chunk) * 0.5: chunk = chunk[:k+1]
        if len(chunk) > 30: out.append(chunk)
        i = max(0, j - overlap)
    return out

# ---------- per-session index (numpy, cosine) ----------
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
    if not e.exists() or not m.exists(): return None, None
    return np.load(e), json.loads(m.read_text(encoding="utf-8"))

def embed(text: str) -> np.ndarray:
    r = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(r.data[0].embedding, dtype=np.float32)

def retrieve(session_id: str, query: str, k=4):
    embs, meta = load_index(session_id)
    if embs is None or not len(meta): return []
    q = embed(query)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    q    = q / np.linalg.norm(q)
    sims = (embs @ q).astype(np.float32)
    idxs = np.argsort(-sims)[:k]
    return [{"score": float(sims[i]), "meta": meta[i]} for i in idxs]

# ---------- ingest / clear ----------
def ingest_files(files, session_id: str):
    if not files: return "Upload one or more files (PDF/TXT/MD)."
    all_chunks, all_meta = [], []
    for f in files:
        p = Path(getattr(f, "name", str(f)))
        name = p.name
        try:
            text = load_file(p)
        except Exception as e:
            return f"Error reading {name}: {e}"
        for c in chunk_text(text):
            all_chunks.append(c)
            all_meta.append({"doc": name, "chunk_id": str(uuid.uuid4()), "text_preview": c[:400].replace("\n"," ")})
    if not all_chunks: return "No readable text found."
    emb_list = [embed(c) for c in all_chunks]
    save_index(session_id, np.vstack(emb_list).astype(np.float32), all_meta)
    # best-effort wipe uploaded temp files
    for f in files:
        try:
            p = Path(getattr(f, "name", str(f)))
            p.unlink(missing_ok=True)
        except: pass
    return f"Ingested {len(all_chunks)} chunks from {len(files)} file(s)."

def clear_session(session_id: str):
    d = sess_dir(session_id)
    if d.exists(): shutil.rmtree(d, ignore_errors=True)
    return "Session index cleared."

# ---------- chat ----------
def format_snips(snips):
    if not snips: return "", []
    lines, srcs = [], set()
    for s in snips:
        m, sc = s["meta"], s["score"]
        lines.append(f"[{m['doc']}] (score={sc:.3f})\n{m['text_preview']}")
        srcs.add(m["doc"])
    return "\n\n".join(lines), sorted(srcs)

def chat_fn(message, history, session_id):
    try:
        snips = retrieve(session_id, message, k=4)
        ctx, srcs = format_snips(snips)
        msgs = [{"role":"system","content":SYSTEM}]
        if ctx:
            msgs.append({"role":"system","content":f"Relevant documents:\n{ctx}"} )
        for h,b in (history or []):
            msgs += [{"role":"user","content":h},{"role":"assistant","content":b}]
        msgs.append({"role":"user","content":message})
        r = client.chat.completions.create(model=GEN_MODEL, messages=msgs, max_tokens=300, temperature=0.1)
        out = r.choices[0].message.content
        if srcs: out += "\n\nSources: " + ", ".join(srcs)
        return out
    except Exception as e:
        return f"[Error] {type(e).__name__}: {e}"

# ---------- UI ----------
with gr.Blocks(title="Data Enablement Chatbot") as demo:
    session_id = gr.State(lambda: str(uuid.uuid4()))  # unique per user session

    gr.Markdown("### Data Enablement Chatbot — upload docs (private to your session), click **Ingest**, then ask questions.")
    with gr.Row():
        with gr.Column(scale=1):
            up = gr.File(label="Upload docs (PDF / MD / TXT)", file_count="multiple")
            ingest_btn = gr.Button("Ingest uploaded files")
            ingest_out = gr.Textbox(label="Ingest status")
            clear_btn = gr.Button("Clear my session index")
            clear_out = gr.Textbox(label="Clear status")
            gr.Markdown("Docs & embeddings are stored **ephemerally** under `/tmp/sessions/<session>` and are cleared on restart or when you press Clear.")
        with gr.Column(scale=2):
            chat = gr.Chatbot()
            msg = gr.Textbox(placeholder="Ask a question about your docs...")
            send = gr.Button("Send")

    ingest_btn.click(lambda f, sid: ingest_files(f, sid), inputs=[up, session_id], outputs=[ingest_out])
    clear_btn.click(lambda sid: clear_session(sid), inputs=[session_id], outputs=[clear_out])

    def on_send(text, history, sid):
        if not text or not text.strip(): return gr.update(value="Please enter a question."), history
        reply = chat_fn(text, history, sid)
        history = (history or []) + [(text, reply)]
        return "", history

    send.click(on_send, inputs=[msg, chat, session_id], outputs=[msg, chat])
    msg.submit(on_send, inputs=[msg, chat, session_id], outputs=[msg, chat])

if __name__ == "__main__":
    demo.launch()
