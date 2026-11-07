import ollama
import numpy as np

MODEL = "nomic-embed-text"


def get_embedding(text):
    return np.array(ollama.embed(model=MODEL, input=text)["embeddings"][0])


def get_chunk_embedding(chunks):
    """Embed a list of chunks in one call, return [(chunk, vector), ...]."""
    vectors = ollama.embed(model=MODEL, input=chunks)["embeddings"]
    return [(chunk, np.array(vec, dtype=np.float32)) for chunk, vec in zip(chunks, vectors)]
