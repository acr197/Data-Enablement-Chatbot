# app.py — Drop-in RAG chatbot for Hugging Face Space
# Requirements: gradio, openai>=1.40.0, python-dotenv, tiktoken, pymupdf, numpy
# Put OPENAI_API_KEY as secret in Hugging Face (exact name: OPENAI_API_KEY)

import os
import json
import uuid
import math
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import gradio as gr
from openai import OpenAI

# ----------------------------
# Config
# ----------------------------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY (add in HF Space Settings → Secrets).")

client = OpenAI(api_key=OPENAI_KEY)

SYSTEM = (
    "You are a helpful company onboarding chatbot. Use provided document context first, "
    "then answer concisely and clearly. If you can't find an answer in the docs, say "
    "'I don't see this in our training docs.'"
)

INDEX_DIR = Path("vector_data")
EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-3.5-turbo")  # fallback safe model
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# ----------------------------
# Utilities: text extraction
# ----------------------------
def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_pdf(path: Path) -> str:
    try:
        import fitz  # PyMuPDF
    except Exception:
        raise RuntimeError("PyMuPDF (fitz) is required to read PDFs. Add pymupdf to requirements.")
    doc = fitz.open(str(path))
    pages = []
    for p in doc:
        pages.append(p.get_text("text"))
    return "\n\n".join(pages)

def load_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".txt", ".md"]:
        return read_text_file(path)
    if ext == ".pdf":
        return read_pdf(path)
    # fallback: try reading as text
    return read_text_file(path)

# ----------------------------
# Chunking + helpers
# ----------------------------
def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        # extend back to sentence boundary if possible
        if end < length:
            # try to backtrack to last sentence end
            last_period = chunk.rfind(". ")
            if last_period > int(len(chunk) * 0.5):
                chunk = chunk[: last_period + 1]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= length:
            break
    # filter small
    return [c for c in chunks if len(c) > 30]

# ----------------------------
# Index management (simple numpy index)
# ----------------------------
def ensure_index_dir():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

def index_exists():
    return INDEX_DIR.exists() and (INDEX_DIR / "embeddings.npy").exists() and (INDEX_DIR / "meta.json").exists()

def save_index(embeddings: np.ndarray, meta: list):
    ensure_index_dir()
    np.save(INDEX_DIR / "embeddings.npy", embeddings)
    with open(INDEX_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_index():
    emb_path = INDEX_DIR / "embeddings.npy"
    meta_path = INDEX_DIR / "meta.json"
    if not emb_path.exists() or not meta_path.exists():
        return None, None
    embeddings = np.load(emb_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return embeddings, meta

# ----------------------------
# Embeddings & retrieval
# ----------------------------
def create_embedding(text: str):
    # call OpenAI embeddings API
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray):
    # a: (n, d) b: (d,)
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b)
    return (a_norm @ b_norm).astype(np.float32)

def retrieve(query: str, top_k=4):
    embeddings, meta = load_index()
    if embeddings is None or len(meta) == 0:
        return []
    q_emb = create_embedding(query)
    sims = cosine_sim(embeddings, q_emb)
    idx = np.argsort(-sims)[:top_k]
    results = []
    for i in idx:
        results.append({"score": float(sims[i]), "meta": meta[i]})
    return results

# ----------------------------
# Ingest: ingest uploaded files
# ----------------------------
def clear_index():
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)

def ingest_files(files):
    """
    files: list of file objects from Gradio upload (temp paths)
    """
    if not files:
        return "Upload one or more files (PDF/TXT/MD)."
    ensure_index_dir()
    all_chunks = []
    all_meta = []
    for file_obj in files:
        # Gradio gives a temp file path in file_obj.name or file_obj
        path = Path(file_obj.name) if hasattr(file_obj, "name") else Path(file_obj)
        name = path.name
        try:
            text = load_file(path)
        except Exception as e:
            return f"Error reading {name}: {e}"
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            uid = str(uuid.uuid4())
            meta = {"doc": name, "chunk_id": uid, "text_preview": c[:400].replace("\n", " ")}
            all_chunks.append(c)
            all_meta.append(meta)
    if not all_chunks:
        return "No text found in uploaded files."
    # compute embeddings in a loop to avoid large single request
    emb_list = []
    for ch in all_chunks:
        emb = create_embedding(ch)
        emb_list.append(emb)
    embeddings = np.vstack(emb_list).astype(np.float32)
    save_index(embeddings, all_meta)
    return f"Ingested {len(all_chunks)} chunks from {len(files)} file(s)."

# ----------------------------
# Chat handler
# ----------------------------
def format_context_snippets(snippets):
    lines = []
    seen_docs = set()
    for s in snippets:
        meta = s["meta"]
        score = s["score"]
        lines.append(f"[{meta['doc']}] (score={score:.3f})\n{meta['text_preview']}\n")
        seen_docs.add(meta['doc'])
    return "\n\n".join(lines), list(seen_docs)

def chat_fn(message, history):
    try:
        # retrieve context
        retrieved = retrieve(message, top_k=4)
        context_text, sources = format_context_snippets(retrieved) if retrieved else ("", [])
        # build messages
        msgs = [{"role": "system", "content": SYSTEM}]
        if context_text:
            msgs.append({"role": "system", "content": f"Relevant documents:\n{context_text}"})
        # replay chat history
        for h, b in (history or []):
            msgs.append({"role": "user", "content": h})
            msgs.append({"role": "assistant", "content": b})
        msgs.append({"role": "user", "content": message})

        # call completion
        resp = client.chat.completions.create(
            model=GEN_MODEL,
            messages=msgs,
            max_tokens=300,
            temperature=0.1
        )
        answer = resp.choices[0].message.content

        # append citation note
        if sources:
            src_line = "Sources: " + ", ".join(sources)
            answer = answer.strip() + "\n\n" + src_line
        return answer
    except Exception as e:
        # show helpful error for debugging
        return f"[Error] {type(e).__name__}: {e}"

# ----------------------------
# Gradio UI
# ----------------------------
def clear_index_button():
    clear_index()
    return "Index cleared."

with gr.Blocks(title="Data Enablement Chatbot") as demo:
    gr.Markdown("### Data Enablement Chatbot — upload docs, click Ingest, then ask questions.")
    with gr.Row():
        with gr.Column(scale=1):
            uploader = gr.File(label="Upload docs (PDF, MD, TXT)", file_count="multiple")
            ingest_btn = gr.Button("Ingest uploaded files")
            ingest_out = gr.Textbox(label="Ingest status")
            clear_btn = gr.Button("Clear index")
            clear_out = gr.Textbox(label="Clear index status")
            info = gr.Markdown("Index stored at `vector_data/` in the Space. Re-deploy to reset.")
        with gr.Column(scale=2):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(placeholder="Ask a question about your docs...")
            submit = gr.Button("Send")

    ingest_btn.click(fn=ingest_files, inputs=[uploader], outputs=[ingest_out])
    clear_btn.click(fn=clear_index_button, inputs=None, outputs=[clear_out])

    def submit_and_update(msg_text, chat_history):
        if not msg_text or msg_text.strip() == "":
            return gr.update(value="Please enter a question."), chat_history
        reply = chat_fn(msg_text, chat_history)
        # append to UI history
        chat_history = chat_history or []
        chat_history.append((msg_text, reply))
        return "", chat_history

    submit.click(fn=submit_and_update, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(fn=submit_and_update, inputs=[msg, chatbot], outputs=[msg, chatbot])

# ----------------------------
# Launch
# ----------------------------
if __name__ == "__main__":
    demo.launch()
