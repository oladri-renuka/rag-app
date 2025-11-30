from pydantic import BaseSettings

class Settings(BaseSettings):
    OLLAMA_URL: str = "http://localhost:11434"   # overridden by env in docker-compose
    OLLAMA_MODEL: str = "llama2"                 # change to the model you pull in ollama
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    FAISS_INDEX_PATH: str = "models/faiss.index"
    METADATA_PATH: str = "models/metadata.json"
    MLFLOW_URI: str = "file:./mlruns"

settings = Settings()
