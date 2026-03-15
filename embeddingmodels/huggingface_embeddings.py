from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
)

texts = [
    "You are going to learn GenAI",
    "GenAI is the future of technology",
    "GenAI can help solve complex problems"
]

vectors = embedding.embed_documents(texts)
print(vectors)