import os
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

api_key = os.environ['GROQ_API_KEY']

from langchain_groq import ChatGroq
llm=ChatGroq(model="llama-3.3-70b-versatile")


def answer_question(question, retriever):

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
