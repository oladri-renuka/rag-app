import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from .index_data import index_folder
from .vectorstore import FaissStore
from .config import settings
from .agent import run_agent
from .embeddings import embed_texts

app = FastAPI(title="RAG Prod - FastAPI")

# initialize store lazily
_store = None

def get_store():
    global _store
    if _store is None:
        # attempt to read index to infer dim
        # if doesn't exist, error for queries
        if not Path(settings.FAISS_INDEX_PATH).exists():
            raise RuntimeError("Index not found. Run /index or index_data.py first.")
        # read index to get dimension
        import faiss
        idx = faiss.read_index(str(settings.FAISS_INDEX_PATH))
        dim = idx.d
        _store = FaissStore(dim)
    return _store

@app.get("/")
def health():
    return {"status":"ok"}

@app.post("/index")
async def upload_and_index(file: UploadFile = File(...)):
    saved_dir = Path("data/uploads")
    saved_dir.mkdir(parents=True, exist_ok=True)
    file_path = saved_dir / file.filename
    content = await file.read()
    file_path.write_bytes(content)
    # call indexer (synchronous for simplicity)
    index_folder(saved_dir)
    return {"message": f"Saved {file.filename} and indexed folder."}

@app.post("/query")
async def query(payload: dict):
    q = payload.get("query") or payload.get("q")
    if not q:
        raise HTTPException(status_code=400, detail="`query` is required")
    store = get_store()
    result = run_agent(q, store, top_k=3)
    return result
