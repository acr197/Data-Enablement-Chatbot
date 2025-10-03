import os
import gradio as gr
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

SYSTEM = "You are a helpful company chatbot. Answer clearly and concisely."

def chat_fn(messages, _history):
    try:
        msgs = [{"role": "system", "content": SYSTEM}] + [m for m in messages if m["role"] != "system"]
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",   # or "gpt-4o" if you want
            messages=msgs,
            max_tokens=200
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

demo = gr.ChatInterface(
    fn=chat_fn,
    type="messages",
    title="Data Enablement Chatbot"
)
demo.launch()
