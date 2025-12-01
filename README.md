
# Open-Source LangGraph RAG Production System

A fully production-ready **Retrieval-Augmented Generation (RAG) system** built using **LangGraph, FastAPI, FAISS, MLflow, and Docker**.  
This system allows you to **index documents (PDFs/text), query using LLMs (Ollama), and retrieve context-aware answers**. It also supports integration with Streamlit for a simple UI.

---

## Features

- Index PDFs and text documents into **FAISS vector store**.
- Use **SentenceTransformers embeddings** for semantic retrieval.
- **FastAPI endpoints** for querying and indexing.
- **LangGraph agent** for orchestrating retrieval and LLM generation.
- **Ollama LLM** integration (local or containerized).
- Logging of queries and results using **MLflow**.
- **Dockerized setup** for local development or deployment on Hugging Face Spaces.
- Optional **Streamlit frontend** for interactive Q&A.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/oladri-renuka/rag-app.git
cd rag-prod
````

2. Create a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Local Development (FastAPI + Ollama)

1. **Start Ollama LLM container** (or local Ollama server):

```bash
docker-compose up -d
```

2. **Index your documents**:

```bash
python app/index_data.py data/   # replace 'data/' with your PDFs/text folder
```

3. **Run the FastAPI app**:

```bash
uvicorn app.main:app --reload
```

4. **Endpoints**:

* `GET /` - Health check
* `POST /index` - Upload and index new documents
* `POST /query` - Query the RAG system

---

## Streamlit Interface

Run the Streamlit app for a quick demo:

```bash
streamlit run app.py
```

* Enter a question in the input box.
* The system retrieves relevant documents from FAISS and generates an answer using the LLM.

---

## FAISS Index Management

Build the FAISS index using:

```bash
python faiss_build.py
```

* FAISS index is saved locally in `faiss_index/`.
* Embeddings are handled with **SentenceTransformers**.

---

## MLflow Logging

* All queries, retrieved documents, and answers are logged via **MLflow**.
* Set your tracking URI in `app/config.py` (default: `file:./mlruns`).

---

## Hugging Face Spaces Deployment

1. Create a **new Space** on Hugging Face (choose Docker or Streamlit SDK).
2. Push your repository to the Space.
3. Ensure `app.py` is the entrypoint for Streamlit-based deployment.
4. Adjust `requirements.txt` and Dockerfile if needed.

---

## Configuration

All settings are in `app/config.py`:

* `OLLAMA_URL` – Local Ollama API URL
* `OLLAMA_MODEL` – Model name (e.g., `llama3.1`)
* `EMBEDDING_MODEL` – SentenceTransformers model
* `FAISS_INDEX_PATH` – Path to FAISS index
* `METADATA_PATH` – Path to metadata JSON
* `MLFLOW_URI` – MLflow tracking URI

---

## License

MIT License

