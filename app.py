import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()

qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")

client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# Conecte ao Vector Store
vector_store = QdrantVectorStore(
    client=client,
    collection_name="sobrapo_collection",  # Nome da coleção salva
    embedding=embeddings,
)

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})

from rag import answer_question
import streamlit as st

st.title("💬 Chatbot Sobrapo")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "O que você gostaria de saber sobre os artigos da SBPO 2023?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    results = answer_question(prompt, retriever)
    msg = results['answer']
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
