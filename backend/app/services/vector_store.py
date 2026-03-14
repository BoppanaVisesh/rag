import numpy as np

stored_chunks = []
stored_vectors = np.empty((0, 384), dtype=np.float32)

def store_embeddings(chunks, embeddings):
    global stored_chunks, stored_vectors
    stored_chunks.extend(chunks)
    vectors = np.array(embeddings, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if stored_vectors.size == 0:
        stored_vectors = vectors
    else:
        stored_vectors = np.vstack((stored_vectors, vectors))

def search_vectors(query_embedding):
    if stored_vectors.shape[0] == 0:
        return []
    query = np.array(query_embedding, dtype=np.float32)
    if query.ndim > 1:
        query = query[0]
    distances = np.sum((stored_vectors - query) ** 2, axis=1)
    k = min(5, len(distances))
    indices = np.argsort(distances)[:k]
    results = []
    for i in indices:
        if i < len(stored_chunks):
            results.append(stored_chunks[i])
    return results
