import streamlit as st
from rag_functions import rag, vectorize_search

vectorstore = vectorize_search("all-MiniLM-L6-v2(chunk)", "sentence-transformers/all-MiniLM-L6-v2")
retriever = vectorstore.as_retriever()

st.title("💬 Chatbot Sobrapo")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "O que você gostaria de saber sobre os artigos da SBPO 2023?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    results = rag(prompt, retriever, vectorstore)
    msg = results['answer']
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
