import ollama
import numpy as np

MODEL = "nomic-embed-text"


def get_embedding(text):
    return np.array(ollama.embed(model=MODEL, input=text)["embeddings"][0])
