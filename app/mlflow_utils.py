import mlflow
from .config import settings

mlflow.set_tracking_uri(settings.MLFLOW_URI)
def log_rag_run(query, answer, retrieved):
    with mlflow.start_run():
        mlflow.log_param("query", query)
        mlflow.log_text(answer, "answer.txt")
        mlflow.log_text(str(retrieved), "retrieved.json")
