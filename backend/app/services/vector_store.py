import faiss
import numpy as np

index = faiss.IndexFlatL2(384)
stored_chunks = []

def store_embeddings(chunks, embeddings):
    global stored_chunks
    stored_chunks.extend(chunks)
    vectors = np.array(embeddings)
    index.add(vectors)

def search_vectors(query_embedding):
    query = np.array([query_embedding])
    distances, indices = index.search(query, 5)
    results = []
    for i in indices[0]:
        if i < len(stored_chunks):
            results.append(stored_chunks[i])
    return results
