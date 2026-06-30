import retriever
from retriever.chroma_retriever import retriever



def retrieve_documents(retriever,query):
    docs = retriever.invoke(query)
    return docs
    