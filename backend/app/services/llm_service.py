import os
from pathlib import Path
from openai import OpenAI, OpenAIError, RateLimitError
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

def generate_answer(question, context):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set")
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    context_text = "\n\n".join(context) if context else "No context available."
    try:
        response = client.chat.completions.create(
            model=model_name,
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
    except RateLimitError:
        if context:
            return f"The Groq API key has reached its quota limit. I cannot generate a model answer right now. Most relevant document text:\n\n{context_text[:1500]}"
        return "The Groq API key has reached its quota limit, and no matching document context was found to answer from."
    except OpenAIError as exc:
        if context:
            return f"The language model request failed: {exc}. Most relevant document text:\n\n{context_text[:1500]}"
        return f"The language model request failed: {exc}"
