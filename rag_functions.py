from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ['GROQ_API_KEY']

from langchain_groq import ChatGroq
llm=ChatGroq(model="llama-3.1-70b-versatile")


def vectorize_search(doc, model="sentence-transformers/all-mpnet-base-v2"): 
    embeddings = HuggingFaceEmbeddings(model_name=model)
    faiss_index = FAISS.load_local(
        "./assets",
        embeddings,
        doc,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return faiss_index

def rag(question, retriever, vectorstore, llm=llm): # futuro: adicionar opção de escolher o modelo de llm, prompt, fluxos do langchain
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    # português
    system_prompt = (
        "Você é um assistente para tarefas de perguntas e respostas. "
        "Use os seguintes trechos de contexto recuperados para responder "
        "a pergunta. Se você não souber a resposta, diga que você não sabe. "
        "Use no máximo três frases e mantenha a resposta concisa."
        "Responda somente em português, ainda que a pergunta esteja em inglês."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    results = rag_chain.invoke({"input": question})
    return results