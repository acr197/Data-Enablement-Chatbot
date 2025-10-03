import os
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

SYSTEM = "You are a helpful company chatbot. Answer clearly and concisely."

def chat_fn(messages, _history):
    # messages is a list of dicts: [{"role":"user"/"assistant"/"system","content": "..."}]
    msgs = [{"role":"system","content": SYSTEM}] + [m for m in messages if m["role"] != "system"]
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=msgs, max_tokens=300)
    return resp.choices[0].message.content

demo = gr.ChatInterface(
    fn=chat_fn,
    type="messages",               # fixes the warning
    title="Data Enablement Chatbot",
)

demo.launch()
