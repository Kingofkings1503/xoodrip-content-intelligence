import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_sim(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors
    """
    return cosine_similarity([vec1], [vec2])[0][0]
