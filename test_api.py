import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Check your .env file."
    )

# Use Groq's API through the existing OpenAI SDK.
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)

response = client.responses.create(
    model=model,
    input="In simple words, explain what a RAG chatbot does.",
)

print(response.output_text)