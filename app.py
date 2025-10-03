import os
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY (HF → Settings → Secrets).")

client = OpenAI(api_key=api_key)
SYSTEM = "You are a helpful company chatbot. Answer clearly and concisely."

def chat_fn(message, history):
    try:
        msgs = [{"role": "system", "content": SYSTEM}]
        for h, b in (history or []):
            msgs.append({"role": "user", "content": h})
            msgs.append({"role": "assistant", "content": b})
        msgs.append({"role": "user", "content": message})

        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",   # start safe; switch to gpt-4o-mini later
            messages=msgs,
            max_tokens=200,
            timeout=30,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[OpenAI error] {type(e).__name__}: {e}"

demo = gr.ChatInterface(fn=chat_fn, title="Data Enablement Chatbot")  # no type=... needed
demo.launch()
