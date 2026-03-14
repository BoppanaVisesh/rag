from sentence_transformers import SentenceTransformer
from app.config import MODEL_NAME

model = SentenceTransformer(MODEL_NAME)

def create_embeddings(chunks):
    return model.encode(chunks)

def embed_query(query):
    return model.encode([query])[0]
