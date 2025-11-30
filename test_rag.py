from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
import streamlit as st


# Load FAISS index
vectorstore = FAISS.load_local(
    "faiss_index",
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)

llm = Ollama(model="llama3.1")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

st.title("RAG QA System")
query = st.text_input("Ask a question:")
if query:
    answer = qa_chain.run(query)
    st.write("Answer:", answer)


# # Load FAISS index
# vectorstore = FAISS.load_local("faiss_index", HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

# # LLM
# llm = OllamaLLM(model="llama3.1")

# # QA chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectorstore.as_retriever(),
# )

# st.title("RAG System Demo")
# query = st.text_input("Enter your question:")

# if query:
#     answer = qa_chain.run(query)
#     st.write("Answer:", answer)

