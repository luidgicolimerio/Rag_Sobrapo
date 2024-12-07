import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")

client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
    timeout=60.0,
)

collection_name = "sobrapo_collection"


if not client.collection_exists(collection_name):

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=768,                       # Dimensão dos vetores
            distance=models.Distance.COSINE # Métrica de distância
        )
    )
    print(f"Coleção '{collection_name}' criada com sucesso!")
else:
    print(f"A coleção '{collection_name}' já existe.")



def indexer(files, folder, method='document', chunk_size=2000, chunk_overlap=200):
    docs = []
    for file in files:
        loader = PyPDFLoader(os.path.join(folder, file))
        doc = loader.load()

        if method == 'document':
            complete_text = " ".join([page.page_content for page in doc])
            docs.append(Document(page_content = complete_text))

        elif method == 'chunk':
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            splits = text_splitter.split_documents(doc)
            docs.extend(splits)

        elif method == 'page':
            docs.extend(doc)

        elif method == 'paragraph':
            for page in doc:
                for paragraph in page.paragraphs:
                    docs.append(Document(page_content = paragraph))
    return docs


vector_store = QdrantVectorStore(
    client=client,
    collection_name="sobrapo_collection",
    embedding=embeddings,
)


if __name__ == "__main__":
    folder = r"C:\workspace\rag_sobrapo_st\assets"
    docs = indexer(os.listdir(folder), folder, method='chunk')

    vector_store.add_documents(docs)

    print(f"{len(docs)} documentos foram vetorados e salvos no Qdrant.")