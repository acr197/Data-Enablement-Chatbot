import os
import gradio as gr
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_bot(message, history):
    messages = [{"role": "system", "content": "You are a helpful company chatbot. Answer clearly and concisely."}]
    for human, bot in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": bot})
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=300
    )
    return response.choices[0].message.content

demo = gr.ChatInterface(ask_bot, title="Data Enablement Chatbot")
demo.launch()
