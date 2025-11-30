from sentence_transformers import SentenceTransformer
import numpy as np
from .config import settings

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model

def embed_texts(texts):
    model = get_model()
    # encode returns numpy array
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    # ensure 2D
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, 0)
    return embeddings
