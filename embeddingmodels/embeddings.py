import os

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=64,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://models.inference.ai.azure.com",
)

vector = embeddings.embed_query("You are going to learn GenAI")

print(vector)