from langgraph import Graph, Step
from .vectorstore import FaissStore
from .embeddings import embed_texts
from .ollama_client import generate_with_ollama
from .config import settings

def build_prompt(query, contexts):
    system = "You are an assistant that answers based only on the provided context. If the answer is not in the context, say 'I don't know based on the provided documents.'"
    context_text = "\n\n".join([f"Document {i+1}:\n{c['meta']['text']}" for i,c in enumerate(contexts)])
    prompt = f"{system}\n\nContext:\n{context_text}\n\nUser question:\n{query}\n\nAnswer concisely:"
    return prompt

def run_agent(query, store: FaissStore, top_k=3):
    # 1. embed the query
    q_emb = embed_texts([query])
    # 2. retrieve
    hits = store.search(q_emb, top_k=top_k)
    # 3. build prompt
    prompt = build_prompt(query, hits)
    # 4. call LLM
    llm_resp = generate_with_ollama(prompt)
    # 5. parse response (Ollama generate typically returns `text` under nested structure)
    # This code assumes the JSON includes 'completion' or 'text' fields; adapt if needed.
    # We'll try to be robust:
    text = None
    if isinstance(llm_resp, dict):
        # new Ollama shapes: {"id":...,"model":"...","result":[{"role":"assistant","content":"..."}], ...}
        if 'result' in llm_resp:
            # join content from parts
            parts = []
            for part in llm_resp['result']:
                if isinstance(part, dict) and 'content' in part:
                    parts.append(part['content'])
            text = "\n".join(parts)
        elif 'text' in llm_resp:
            text = llm_resp['text']
        else:
            # fallback to str representation
            text = str(llm_resp)
    else:
        text = str(llm_resp)
    return {"answer": text, "retrieved": hits}
