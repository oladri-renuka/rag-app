import requests
from .config import settings

def generate_with_ollama(prompt: str, model: str = None, max_tokens: int = 512, temperature: float = 0.0):
    model = model or settings.OLLAMA_MODEL
    url = f"{settings.OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()
