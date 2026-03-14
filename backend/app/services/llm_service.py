import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

def generate_answer(question, context):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)
    context_text = "\n\n".join(context) if context else "No context available."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the user's question using only the provided context. If the context doesn't contain enough information, say so."
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {question}"
            }
        ]
    )
    return response.choices[0].message.content
