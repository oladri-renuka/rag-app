import argparse
from pathlib import Path
from .embeddings import embed_texts
from .vectorstore import FaissStore
from .config import settings
import json
from PyPDF2 import PdfReader
from tqdm import tqdm

def extract_text_from_pdf(path: Path):
    reader = PdfReader(str(path))
    texts = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            texts.append(txt.strip())
    # split into chunks if large
    return "\n".join(texts)

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += (chunk_size - overlap)
    return chunks

def index_folder(folder, index_path=None):
    folder = Path(folder)
    corpus = []
    metadatas = []
    for f in folder.iterdir():
        if f.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(f)
        elif f.suffix.lower() in [".txt",""]:
            text = f.read_text(encoding="utf-8")
        else:
            continue
        chunks = chunk_text(text)
        for c in chunks:
            corpus.append(c)
            metadatas.append({"source": str(f), "text": c})
    # embeddings
    embeddings = embed_texts(corpus)
    dim = embeddings.shape[1]
    store = FaissStore(dim)
    store.add_embeddings(embeddings, metadatas)
    print(f"Indexed {len(corpus)} chunks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder with PDFs or text files to index")
    args = parser.parse_args()
    index_folder(args.folder)
