import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def generate_answer(question, context):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
