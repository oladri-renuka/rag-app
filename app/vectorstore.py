import faiss
import numpy as np
import json
from pathlib import Path
from .config import settings

class FaissStore:
    def __init__(self, dim, index_path=settings.FAISS_INDEX_PATH, metadata_path=settings.METADATA_PATH):
        self.dim = dim
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self._index = None
        self.metadata = []
        if self.index_path.exists() and self.metadata_path.exists():
            self._load()

    def _create_index(self):
        # simple flat index; change to IndexIVFFlat for larger datasets
        self._index = faiss.IndexFlatL2(self.dim)

    def add_embeddings(self, embeddings: np.ndarray, metadatas: list):
        if self._index is None:
            self._create_index()
        self._index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadatas)
        self._save()

    def search(self, q_embedding: np.ndarray, top_k=5):
        if self._index is None:
            return []
        D, I = self._index.search(q_embedding.astype(np.float32), top_k)
        results = []
        for i_list, dist_list in zip(I.tolist(), D.tolist()):
            for idx, dist in zip(i_list, dist_list):
                if idx < len(self.metadata):
                    results.append({"id": idx, "dist": dist, "meta": self.metadata[idx]})
        return results

    def _save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _load(self):
        self._index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
